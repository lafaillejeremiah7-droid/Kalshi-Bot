from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field

from kalshi_research.domain.events import (
    EventKind,
    ResearchEvent,
)
from kalshi_research.research.materializer import (
    FeatureMaterializer,
    ModelFeatureRow,
)
from kalshi_research.research.synchronizer import ReceiveTimeSynchronizer
from kalshi_research.storage.sqlite_store import SqliteEventStore


DEFAULT_EMIT_KINDS = frozenset(
    {
        EventKind.ORDERBOOK_SNAPSHOT,
        EventKind.ORDERBOOK_DELTA,
        EventKind.INDEX_TICK,
        EventKind.SPOT_TICK,
    }
)


@dataclass(slots=True)
class FeatureReplayPipeline:
    market_ticker: str
    synchronizer: ReceiveTimeSynchronizer = field(default_factory=ReceiveTimeSynchronizer)
    materializer: FeatureMaterializer = field(default_factory=FeatureMaterializer)
    emit_kinds: frozenset[EventKind] = DEFAULT_EMIT_KINDS
    probability_ready_only: bool = False

    def run(self, events: Iterable[ResearchEvent]) -> Iterator[ModelFeatureRow]:
        for event in events:
            self.synchronizer.ingest(event)
            if not self._should_emit(event):
                continue
            frame = self.synchronizer.frame(self.market_ticker)
            row = self.materializer.materialize(frame)
            if self.probability_ready_only and not row.probability_ready:
                continue
            yield row

    def _should_emit(self, event: ResearchEvent) -> bool:
        if event.kind not in self.emit_kinds:
            return False
        if event.market_ticker is None:
            return True
        return event.market_ticker == self.market_ticker


def feature_rows_from_store(
    store: SqliteEventStore,
    market_ticker: str,
    *,
    probability_ready_only: bool = False,
) -> list[ModelFeatureRow]:
    """Replay all sources by receive time; never ticker-filter the store first."""
    pipeline = FeatureReplayPipeline(
        market_ticker=market_ticker,
        probability_ready_only=probability_ready_only,
    )
    return list(pipeline.run(store.iter_events(order_by="receive")))


def feature_rows_jsonl(rows: Iterable[ModelFeatureRow]) -> str:
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                asdict(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def feature_rows_digest(rows: Iterable[ModelFeatureRow]) -> str:
    payload = feature_rows_jsonl(rows).encode()
    return hashlib.sha256(payload).hexdigest()
