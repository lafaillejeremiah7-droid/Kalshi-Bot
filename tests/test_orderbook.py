from decimal import Decimal

import pytest

from kalshi_research.feeds.kalshi_ws import BinaryOrderBook, SequenceGap


def test_snapshot_delta_and_complement_ask():
    book = BinaryOrderBook("KXBTC15M-X")
    book.apply_snapshot(10, [["0.40", "5"]], [["0.55", "7"]])
    assert book.best_yes_bid == Decimal("0.40")
    assert book.implied_yes_ask == Decimal("0.45")
    book.apply_delta(11, "yes", "0.41", "3")
    assert book.best_yes_bid == Decimal("0.41")


def test_gap_is_fatal_for_deterministic_replay():
    book = BinaryOrderBook("KXBTC15M-X")
    book.apply_snapshot(10, [], [])
    with pytest.raises(SequenceGap):
        book.apply_delta(12, "yes", "0.40", "1")
