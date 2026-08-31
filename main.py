from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from xau_company.config import Settings
from xau_company.data import TwelveDataClient
from xau_company.orchestrator import BossAgent
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.research import StrategyResearchAgent
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
    lab = StrategyResearchAgent(
        max_candidates=cfg.max_candidates,
        spread_bps=cfg.spread_bps,
        walk_forward_folds=cfg.walk_forward_folds,
        catalog_size=cfg.research_catalog_size,
        min_walk_forward_folds=cfg.min_walk_forward_folds,
        slippage_bps=cfg.slippage_bps,
        backtest_stop_atr=cfg.backtest_stop_atr,
        backtest_reward_risk=cfg.backtest_reward_risk,
    )
    outcomes = OutcomeCalibrationAgent(
        db_path=cfg.outcome_db_path,
        max_age_hours=cfg.outcome_max_age_hours,
        bin_width=cfg.calibration_bin_width,
        prior_strength=cfg.calibration_prior_strength,
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
                    "Strategy lab universe=%s evaluated=%s robust_catalog=%s top=%s walk_forward_folds=%s spread_bps=%.2f slippage_bps=%.2f",
                    lab.last_universe_size,
                    lab.last_evaluated,
                    len(lab.catalog),
                    len(top),
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
                    if fingerprint != last_fingerprint and not already_recorded:
                        emitted_at = datetime.now(timezone.utc)
                        log.info(
                            "Signal %s using %s raw_confidence %.1f%% calibrated_confidence %.1f%% samples=%s",
                            signal.direction.value,
                            signal.selected_strategy,
                            raw_confidence * 100,
                            signal.confidence * 100,
                            calibration.samples,
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
                    elif already_recorded:
                        log.info("Duplicate signal suppressed by persistent outcome ledger")
            else:
                log.info("No strategy passed research + market-context + risk thresholds")
        except Exception:
            log.exception("Cycle failed")
        cycle += 1
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
