from __future__ import annotations

from decimal import Decimal

import pytest

from kalshi_research.domain.events import Source, SpotTickEvent
from kalshi_research.research.complete import ResearchCompletionError
from kalshi_research.research.completion_entrypoint import run_research_completion_events


def test_completion_entrypoint_rejects_original_receive_time_regression():
    events = (
        SpotTickEvent(
            source=Source.COINBASE,
            event_ts_ns=200,
            recv_ts_ns=200,
            venue="coinbase",
            symbol="BTC-USD",
            bid=Decimal("100"),
            ask=Decimal("101"),
        ),
        SpotTickEvent(
            source=Source.COINBASE,
            event_ts_ns=100,
            recv_ts_ns=100,
            venue="coinbase",
            symbol="BTC-USD",
            bid=Decimal("100"),
            ask=Decimal("101"),
        ),
    )

    with pytest.raises(ResearchCompletionError, match="receive_time_regression"):
        run_research_completion_events(events)
