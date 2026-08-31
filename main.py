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
from xau_company.quality import MarketDataQualityAgent
from xau_company.runtime_quality import fetch_resolution_history, revalidate_optional_macro
from xau_company.telegram import TelegramNotifier, TelegramRejectedError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xau-company")


def _pick_frame(frames: dict[str, pd.DataFrame], preferences: tuple[str, ...]) -> pd.DataFrame | None:
    for timeframe in preferences:
        frame = frames.get(timeframe)
        if frame is not None and not frame.empty:
            return frame
    return None


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
    quality = MarketDataQualityAgent(
        max_stale_multiplier=cfg.max_stale_multiplier,
        timezone_name="America/Chicago",
    )
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
        enable_invention=cfg.enable_strategy_invention,
        invention_library_path=cfg.invention_library_path,
        invented_families_per_cycle=cfg.invented_families_per_cycle,
        invented_variants_per_family=cfg.invented_variants_per_family,
        invention_library_size=cfg.invention_library_size,
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
        research_interval=cfg.research_interval,
    )
    telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)

    cycle = 0
    last_research_success: int | None = None
    dxy: pd.DataFrame | None = None
    yields: pd.DataFrame | None = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            if not quality.market_is_open(now):
                log.info("Market-quality veto: XAU/USD session is closed")
                cycle += 1
                time.sleep(cfg.poll_seconds)
                continue

            raw_frames = market.multi_timeframe(cfg.symbol, cfg.timeframes, cfg.context_output_size)
            if not raw_frames:
                raise RuntimeError("No XAU/USD timeframes available")
            frames, reports = quality.clean_frames(
                raw_frames,
                now=now,
                required_context=(cfg.research_interval, "1h", "4h"),
                execution_choices=("1min", "5min"),
            )
            required_names = {cfg.research_interval, "1h", "4h", "execution"}
            failures = [r for r in reports if not r.ok and r.timeframe in required_names]
            if failures:
                log.warning(
                    "Market-quality veto: %s",
                    "; ".join(f"{r.timeframe}: {r.reason}" for r in failures),
                )
                cycle += 1
                time.sleep(cfg.poll_seconds)
                continue

            research_live_df = frames.get(cfg.research_interval)
            signal_setup_at = _frame_timestamp(research_live_df)

            # Resolve forward outcomes from the finest healthy feed available.
            # A 5m fallback prevents calibration from freezing when 1m is unavailable.
            resolution_df, resolution_minutes, resolution_interval = fetch_resolution_history(
                market,
                quality,
                cfg.symbol,
                cfg.outcome_max_age_hours,
                cfg.context_output_size,
                now,
            )
            if resolution_df is not None and resolution_minutes is not None:
                resolved = outcomes.resolve_open(resolution_df, interval_minutes=resolution_minutes)
                if sum(resolved.values()):
                    summary = outcomes.summary()
                    log.info(
                        "Outcome desk resolved interval=%s wins=%s losses=%s expired=%s ambiguous=%s total_resolved=%s forward_win_rate=%s brier=%s",
                        resolution_interval,
                        resolved["wins"],
                        resolved["losses"],
                        resolved["expired"],
                        resolved["ambiguous"],
                        summary["resolved"],
                        f"{summary['win_rate']:.1%}" if isinstance(summary["win_rate"], float) else "n/a",
                        f"{summary['brier_score']:.4f}" if isinstance(summary["brier_score"], float) else "n/a",
                    )

            research_due = (
                last_research_success is None
                or cycle - last_research_success >= cfg.research_every_cycles
            )
            if research_due:
                raw_research = market.candles(cfg.symbol, cfg.research_interval, cfg.output_size)
                research_df, research_report = quality.clean_frame(
                    raw_research,
                    cfg.research_interval,
                    now=now,
                    require_fresh=True,
                )
                if not research_report.ok:
                    raise RuntimeError(f"Research data rejected: {research_report.reason}")
                frames[cfg.research_interval] = research_df
                research_live_df = research_df
                signal_setup_at = _frame_timestamp(research_live_df)
                top = lab.run(research_df)
                last_research_success = cycle
                log.info(
                    "Strategy lab universe=%s evaluated=%s lifetime_trials=%s live_catalog=%s experimental_catalog=%s invented_catalog=%s top=%s dynamic_library=%s invention_library=%s discovered=%s promoted=%s quarantined=%s invented_new_families=%s invented_new_variants=%s invention_promoted=%s invention_quarantined=%s invented_family_total=%s invented_family_promoted=%s walk_forward_folds=%s spread_bps=%.2f slippage_bps=%.2f",
                    lab.last_universe_size,
                    lab.last_evaluated,
                    lab.last_lifetime_trials,
                    len(lab.catalog),
                    lab.last_experimental_catalog_size,
                    lab.last_invented_catalog_size,
                    len(top),
                    lab.dynamic_library_size,
                    lab.invention_library_size,
                    lab.last_discovered,
                    lab.last_promoted,
                    lab.last_quarantined,
                    lab.last_invented_families,
                    lab.last_invented_variants,
                    lab.last_invention_promoted,
                    lab.last_invention_quarantined,
                    lab.invention_family_count,
                    lab.invention_promoted_family_count,
                    lab.walk_forward_folds,
                    cfg.spread_bps,
                    cfg.slippage_bps,
                )

            # Macro API calls are rate-limited to every five cycles, but cached
            # frames are revalidated for freshness on every decision below.
            if cycle == 0 or cycle % 5 == 0:
                dxy = revalidate_optional_macro(
                    quality,
                    market.safe_candles(cfg.dxy_symbol, cfg.macro_interval, cfg.context_output_size),
                    cfg.macro_interval,
                    now,
                )
                yields = revalidate_optional_macro(
                    quality,
                    market.safe_candles(cfg.yield_symbol, cfg.macro_interval, cfg.context_output_size),
                    cfg.macro_interval,
                    now,
                )

            decision_now = pd.Timestamp(datetime.now(timezone.utc))
            setup_start = pd.Timestamp(signal_setup_at)
            if setup_start.tzinfo is None:
                setup_start = setup_start.tz_localize("UTC")
            else:
                setup_start = setup_start.tz_convert("UTC")
            setup_end = setup_start + quality.interval_delta(cfg.research_interval)
            signal_delay = decision_now - setup_end
            if signal_delay < pd.Timedelta(0):
                log.info("Signal timing veto: research candle is not complete")
                cycle += 1
                time.sleep(cfg.poll_seconds)
                continue
            if signal_delay > pd.Timedelta(minutes=cfg.max_signal_delay_minutes):
                log.info(
                    "Signal timing veto: setup is %s old after research candle close (max %s min)",
                    signal_delay,
                    cfg.max_signal_delay_minutes,
                )
                cycle += 1
                time.sleep(cfg.poll_seconds)
                continue

            live_price = market.safe_price(cfg.symbol)
            if live_price is None:
                log.info("Market-quality veto: no fresh executable reference price")
                cycle += 1
                time.sleep(cfg.poll_seconds)
                continue

            # Cached context must still be fresh now, not merely when it was fetched.
            dxy_for_decision = revalidate_optional_macro(quality, dxy, cfg.macro_interval, decision_now)
            yields_for_decision = revalidate_optional_macro(quality, yields, cfg.macro_interval, decision_now)

            signal = boss.decide(
                cfg.symbol,
                frames,
                dxy=dxy_for_decision,
                yield_df=yields_for_decision,
                entry_price=live_price,
            )
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
                        "setup_candle_start": OutcomeCalibrationAgent.utc_iso(signal_setup_at),
                        "setup_candle_end": OutcomeCalibrationAgent.utc_iso(setup_end),
                        "signal_delay_seconds": max(0.0, signal_delay.total_seconds()),
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
                    emitted_at = datetime.now(timezone.utc)
                    # Keep the original research holding horizon intact. The outcome
                    # ledger anchors it to setup_candle_end, so send latency no longer
                    # shifts the timeout window later than the backtest.
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
                        reservation = outcomes.reserve_if_under_cap(
                            signal,
                            emitted_at,
                            selection_confidence=raw_confidence,
                            setup_at=signal_setup_at,
                            day_start=day_start,
                            day_end=day_end,
                            max_per_day=cfg.max_trades_per_day,
                        )
                        if not reservation.reserved:
                            log.info("Signal reservation veto: %s", reservation.reason)
                        else:
                            signal.strategy_stats["trades_today_before_signal"] = reservation.trades_today
                            log.info(
                                "Signal %s using %s raw_confidence %.1f%% calibrated_confidence %.1f%% samples=%s daily_slot=%s/%s",
                                signal.direction.value,
                                signal.selected_strategy,
                                raw_confidence * 100,
                                signal.confidence * 100,
                                calibration.samples,
                                reservation.trades_today + 1,
                                cfg.max_trades_per_day,
                            )
                            if cfg.paper_mode:
                                log.info("PAPER_MODE: %s", telegram.format_signal(signal).replace("\n", " | "))
                                outcomes.mark_delivery_state(signal, signal_setup_at, "SENT", "paper")
                            else:
                                try:
                                    message_id = telegram.send(signal)
                                except TelegramRejectedError:
                                    outcomes.mark_delivery_state(signal, signal_setup_at, "FAILED")
                                    raise
                                except Exception:
                                    outcomes.mark_delivery_state(signal, signal_setup_at, "UNKNOWN")
                                    raise
                                else:
                                    outcomes.mark_delivery_state(signal, signal_setup_at, "SENT", message_id)
            else:
                log.info("No strategy passed research + market-context + risk thresholds")
        except Exception:
            log.exception("Cycle failed")
        cycle += 1
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
