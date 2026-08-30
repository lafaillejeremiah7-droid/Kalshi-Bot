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
    lab = StrategyResearchAgent(cfg.max_candidates, cfg.spread_bps)
    boss = BossAgent(lab, cfg.min_confidence, cfg.min_consensus)
    telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)

    cycle = 0
    last_fingerprint: tuple | None = None
    while True:
        try:
            df = market.candles(cfg.symbol, cfg.interval, cfg.output_size)
            if cycle == 0 or cycle % cfg.research_every_cycles == 0:
                top = lab.run(df)
                log.info("Strategy lab evaluated %s variants; %s survived into top rankings", lab.last_evaluated, len(top))

            signal = boss.decide(cfg.symbol, df)
            if signal:
                fingerprint = (
                    signal.direction.value,
                    round(signal.entry, 1),
                    round(signal.stop_loss, 1),
                    round(signal.take_profit, 1),
                )
                if fingerprint != last_fingerprint:
                    log.info("Signal %s confidence %.1f%%", signal.direction.value, signal.confidence * 100)
                    if cfg.paper_mode:
                        log.info("PAPER_MODE: %s", telegram.format_signal(signal).replace("\n", " | "))
                    else:
                        telegram.send(signal)
                    last_fingerprint = fingerprint
            else:
                log.info("No signal passed boss thresholds")
        except Exception:
            log.exception("Cycle failed")
        cycle += 1
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
