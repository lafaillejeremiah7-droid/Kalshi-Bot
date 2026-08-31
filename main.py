from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.config import Settings
from xau_company.data import TwelveDataClient
from xau_company.frequency import TradeFrequencyGuard
from xau_company.orchestrator import BossAgent
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.telegram import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xau-company")


def _pick_frame(frames: dict[str, pd.DataFrame], preferences: tuple[str, ...]) -> pd.DataFrame | None:
    for timeframe in preferences:
        frame = frames.get(timeframe)
        if frame is not None and not frame.empty:
            return frame
    return next((df for df in frames.values() if df is not None and not df.empty), None)


def _frame_timestamp(frame: pd.DataFrame | None):
    if frame is not None and not frame.empty and "datetime" in frame.columns:
        value = frame["datetime"].iloc[-1]
        if pd.notna(value):
            return value
    return datetime.now(timezone.utc)


def run() -> None:
    cfg = Settings()
    cfg.validate()
    market = TwelveDataClient(cfg.twelve_data_api_key)
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=cfg.max_candidates,
        spread_bps=cfg.spread_bps,
        walk_forward_folds=cfg.walk_forward_folds,
        catalog_size=cfg.research_catalog_size,
        min_walk_forward_folds=cfg.min_walk_forward_folds,
        slippage_bps=cfg.slippage_bps,
        backtest_stop_atr=cfg.backtest_stop_atr,
        backtest_reward_risk=cfg.backtest_reward_risk,
        enable_evolution=cfg.enable_strategy_evolution,
        strategy_library_path=cfg.strategy_library_path,
        discoveries_per_cycle=cfg.discoveries_per_cycle,
        discovery_library_size=cfg.discovery_library_size,
        overfit_min_adjusted_score=cfg.overfit_min_adjusted_score,
        overfit_min_profit_factor=cfg.overfit_min_profit_factor,
        overfit_min_avg_r=cfg.overfit_min_avg_r,
        overfit_min_trades=cfg.overfit_min_trades,
        overfit_max_walk_forward_std=cfg.overfit_max_walk_forward_std,
        overfit_max_train_valid_gap=cfg.overfit_max_train_valid_gap,
        overfit_max_drawdown_r=cfg.overfit_max_drawdown_r,
        overfit_max_loss_streak=cfg.overfit_max_loss_streak,
    )
    outcomes = OutcomeCalibrationAgent(
        db_path=cfg.outcome_db_path,
        max_age_hours=cfg.outcome_max_age_hours,
        bin_width=cfg.calibration_bin_width,
        prior_strength=cfg.calibration_prior_strength,
    )
    frequency = TradeFrequencyGuard(
        timezone_name=cfg.trade_timezone,
        max_trades_per_day=cfg.max_trades_per_day,
    )
    boss = BossAgent(
        lab,
        cfg.min_confidence,
        cfg.min_consensus,
        cfg.high_impact_events_utc,
        cfg.news_block_minutes,
    )
    telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)

    cycle = 0
    last_fingerprint: tuple | None = None
    dxy = None
    yields = None

    while True:
        try:
            frames = market.multi_timeframe(cfg.symbol, cfg.timeframes, cfg.context_output_size)
            if not frames:
                raise RuntimeError("No XAU/USD timeframes available")

            resolution_df = _pick_frame(frames, ("1min", "5min", "15min", "1h", "4h"))
            execution_df = _pick_frame(frames, ("5min", "1min", "15min"))
            signal_setup_at = _frame_timestamp(execution_df)

            if resolution_df is not None:
                resolved = outcomes.resolve_open(resolution_df)
                if sum(resolved.values()):
                    summary = outcomes.summary()
                    log.info(
                        "Outcome desk resolved wins=%s losses=%s expired=%s total_resolved=%s forward_win_rate=%s brier=%s",
                        resolved["wins"],
                        resolved["losses"],
                        resolved["expired"],
                        summary["resolved"],
                        f"{summary['win_rate']:.1%}" if isinstance(summary["win_rate"], float) else "n/a",
                        f"{summary['brier_score']:.4f}" if isinstance(summary["brier_score"], float) else "n/a",
                    )

            if cycle == 0 or cycle % cfg.research_every_cycles == 0:
                research_df = market.candles(cfg.symbol, cfg.research_interval, cfg.output_size)
                frames[cfg.research_interval] = research_df
                top = lab.run(research_df)
                log.info(
                    "Strategy lab universe=%s evaluated=%s live_catalog=%s experimental_catalog=%s top=%s dynamic_library=%s discovered=%s promoted=%s quarantined=%s walk_forward_folds=%s spread_bps=%.2f slippage_bps=%.2f",
                    lab.last_universe_size,
                    lab.last_evaluated,
                    len(lab.catalog),
                    lab.last_experimental_catalog_size,
                    len(top),
                    lab.dynamic_library_size,
                    lab.last_discovered,
                    lab.last_promoted,
                    lab.last_quarantined,
                    lab.walk_forward_folds,
                    cfg.spread_bps,
                    cfg.slippage_bps,
                )

            if cycle == 0 or cycle % 5 == 0:
                dxy = market.safe_candles(cfg.dxy_symbol, cfg.macro_interval, cfg.context_output_size)
                yields = market.safe_candles(cfg.yield_symbol, cfg.macro_interval, cfg.context_output_size)

            signal = boss.decide(cfg.symbol, frames, dxy=dxy, yield_df=yields)
            if signal:
                raw_confidence = float(signal.confidence)
                calibration = outcomes.calibrate(
                    raw_confidence,
                    strategy=signal.selected_strategy,
                    regime=signal.regime,
                )
                signal.strategy_stats = dict(signal.strategy_stats or {})
                signal.strategy_stats.update(
                    {
                        "selection_confidence_raw": raw_confidence,
                        "calibrated_confidence": calibration.probability,
                        "calibration_samples": calibration.samples,
                        "calibration_wins": calibration.wins,
                        "calibration_brier_score": calibration.brier_score,
                    }
                )
                signal.confidence = calibration.probability

                if signal.confidence < cfg.min_confidence:
                    log.info(
                        "Calibration veto: raw %.1f%% -> calibrated %.1f%% from %s forward outcomes",
                        raw_confidence * 100,
                        signal.confidence * 100,
                        calibration.samples,
                    )
                else:
                    setup_key = OutcomeCalibrationAgent.utc_iso(signal_setup_at)
                    fingerprint = (
                        setup_key,
                        signal.direction.value,
                        signal.selected_strategy,
                        round(signal.entry, 1),
                        round(signal.stop_loss, 1),
                        round(signal.take_profit, 1),
                    )
                    already_recorded = outcomes.exists(signal, signal_setup_at)
                    if already_recorded:
                        log.info("Duplicate signal suppressed by persistent outcome ledger")
                    elif fingerprint != last_fingerprint:
                        emitted_at = datetime.now(timezone.utc)
                        day_start, day_end = frequency.day_bounds_utc(emitted_at)
                        trades_today = outcomes.count_emitted_between(day_start, day_end)
                        frequency_decision = frequency.evaluate(emitted_at, trades_today)

                        if not frequency_decision.allowed:
                            log.info(
                                "Frequency veto: %s date=%s trades_today=%s max=%s timezone=%s",
                                frequency_decision.reason,
                                frequency_decision.local_date,
                                frequency_decision.trades_today,
                                cfg.max_trades_per_day,
                                cfg.trade_timezone,
                            )
                        else:
                            signal.strategy_stats.update(
                                {
                                    "trades_today_before_signal": frequency_decision.trades_today,
                                    "daily_trade_cap": cfg.max_trades_per_day,
                                    "trade_timezone": cfg.trade_timezone,
                                }
                            )
                            log.info(
                                "Signal %s using %s raw_confidence %.1f%% calibrated_confidence %.1f%% samples=%s daily_slot=%s/%s",
                                signal.direction.value,
                                signal.selected_strategy,
                                raw_confidence * 100,
                                signal.confidence * 100,
                                calibration.samples,
                                frequency_decision.trades_today + 1,
                                cfg.max_trades_per_day,
                            )
                            if cfg.paper_mode:
                                log.info("PAPER_MODE: %s", telegram.format_signal(signal).replace("\n", " | "))
                            else:
                                telegram.send(signal)
                            outcomes.record(
                                signal,
                                emitted_at,
                                selection_confidence=raw_confidence,
                                setup_at=signal_setup_at,
                            )
                            last_fingerprint = fingerprint
            else:
                log.info("No strategy passed research + market-context + risk thresholds")
        except Exception:
            log.exception("Cycle failed")
        cycle += 1
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
