from __future__ import annotations

import logging
import time

from xau_company.config import Settings
from xau_company.data import TwelveDataClient
from xau_company.orchestrator import BossAgent
from xau_company.research import StrategyResearchAgent
from xau_company.telegram import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xau-company")


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

            if cycle == 0 or cycle % cfg.research_every_cycles == 0:
                research_df = market.candles(cfg.symbol, cfg.research_interval, cfg.output_size)
                frames[cfg.research_interval] = research_df
                top = lab.run(research_df)
                log.info(
                    "Strategy lab universe=%s evaluated=%s robust_catalog=%s top=%s walk_forward_folds=%s",
                    lab.last_universe_size,
                    lab.last_evaluated,
                    len(lab.catalog),
                    len(top),
                    lab.walk_forward_folds,
                )

            if cycle == 0 or cycle % 5 == 0:
                dxy = market.safe_candles(cfg.dxy_symbol, cfg.macro_interval, cfg.context_output_size)
                yields = market.safe_candles(cfg.yield_symbol, cfg.macro_interval, cfg.context_output_size)

            signal = boss.decide(cfg.symbol, frames, dxy=dxy, yield_df=yields)
            if signal:
                fingerprint = (
                    signal.direction.value,
                    signal.selected_strategy,
                    round(signal.entry, 1),
                    round(signal.stop_loss, 1),
                    round(signal.take_profit, 1),
                )
                if fingerprint != last_fingerprint:
                    log.info(
                        "Signal %s using %s confidence %.1f%%",
                        signal.direction.value,
                        signal.selected_strategy,
                        signal.confidence * 100,
                    )
                    if cfg.paper_mode:
                        log.info("PAPER_MODE: %s", telegram.format_signal(signal).replace("\n", " | "))
                    else:
                        telegram.send(signal)
                    last_fingerprint = fingerprint
            else:
                log.info("No strategy passed research + market-context + risk thresholds")
        except Exception:
            log.exception("Cycle failed")
        cycle += 1
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
