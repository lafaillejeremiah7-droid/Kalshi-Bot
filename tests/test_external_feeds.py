from decimal import Decimal

import pytest

from kalshi_research.feeds.external import (
    ExternalFeedError,
    KrakenBook,
    coinbase_subscription_messages,
    kraken_subscription_message,
    normalize_coinbase_ticker,
    rfc3339_to_ns,
)


def test_rfc3339_preserves_nanoseconds():
    assert rfc3339_to_ns("2023-02-09T20:30:37.167359596Z") == 1_675_974_637_167_359_596
    assert rfc3339_to_ns("2023-02-09T20:30:37.1Z") == 1_675_974_637_100_000_000


def test_coinbase_public_subscription_messages_are_non_trading():
    messages = coinbase_subscription_messages()
    assert messages == (
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"},
        {"type": "subscribe", "channel": "heartbeats"},
    )
    lowered = repr(messages).lower()
    assert "user" not in lowered
    assert "order" not in lowered


def test_normalizes_coinbase_ticker():
    quotes = normalize_coinbase_ticker(
        {
            "channel": "ticker",
            "client_id": "",
            "timestamp": "2023-02-09T20:30:37.167359596Z",
            "sequence_num": 123,
            "events": [
                {
                    "type": "snapshot",
                    "tickers": [
                        {
                            "type": "ticker",
                            "product_id": "BTC-USD",
                            "price": "21932.98",
                            "best_bid": "21931.98",
                            "best_bid_quantity": "8.21",
                            "best_ask": "21933.98",
                            "best_ask_quantity": "3.07",
                        }
                    ],
                }
            ],
        },
        recv_ts_ns=1_675_974_637_200_000_000,
    )
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.venue == "coinbase"
    assert quote.event_ts_ns == 1_675_974_637_167_359_596
    assert quote.bid == Decimal("21931.98")
    assert quote.ask == Decimal("21933.98")
    assert quote.last == Decimal("21932.98")
    assert quote.source_sequence == 123


def test_coinbase_heartbeats_are_control_plane():
    assert (
        normalize_coinbase_ticker(
            {"channel": "heartbeats", "timestamp": "2023-02-09T20:30:37Z", "events": []},
            recv_ts_ns=1,
        )
        == ()
    )


def test_coinbase_crossed_book_fails_closed():
    with pytest.raises(ExternalFeedError):
        normalize_coinbase_ticker(
            {
                "channel": "ticker",
                "timestamp": "2023-02-09T20:30:37Z",
                "sequence_num": 1,
                "events": [
                    {
                        "type": "update",
                        "tickers": [
                            {
                                "product_id": "BTC-USD",
                                "price": "100",
                                "best_bid": "101",
                                "best_bid_quantity": "1",
                                "best_ask": "100",
                                "best_ask_quantity": "1",
                            }
                        ],
                    }
                ],
            },
            recv_ts_ns=1_675_974_637_200_000_000,
        )


def test_kraken_subscription_is_public_book_only():
    assert kraken_subscription_message() == {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": ["BTC/USD"],
            "depth": 10,
            "snapshot": True,
        },
    }


def test_kraken_snapshot_and_update_reconstruct_top_of_book():
    book = KrakenBook()
    snapshot = book.apply(
        {
            "channel": "book",
            "type": "snapshot",
            "data": [
                {
                    "symbol": "BTC/USD",
                    "bids": [
                        {"price": Decimal("30000.0"), "qty": Decimal("2.0")},
                        {"price": Decimal("29999.0"), "qty": Decimal("3.0")},
                    ],
                    "asks": [
                        {"price": Decimal("30001.0"), "qty": Decimal("1.5")},
                        {"price": Decimal("30002.0"), "qty": Decimal("4.0")},
                    ],
                    "checksum": 123456,
                    "timestamp": "2024-01-01T00:00:00.123456789Z",
                }
            ],
        },
        recv_ts_ns=1_704_067_200_200_000_000,
    )
    assert snapshot[0].bid == Decimal("30000.0")
    assert snapshot[0].ask == Decimal("30001.0")
    assert snapshot[0].checksum == 123456

    update = book.apply(
        {
            "channel": "book",
            "type": "update",
            "data": [
                {
                    "symbol": "BTC/USD",
                    "bids": [{"price": Decimal("30000.0"), "qty": Decimal("0")}],
                    "asks": [{"price": Decimal("30001.0"), "qty": Decimal("2.5")}],
                    "checksum": 789012,
                    "timestamp": "2024-01-01T00:00:00.223456789Z",
                }
            ],
        },
        recv_ts_ns=1_704_067_200_300_000_000,
    )
    assert update[0].bid == Decimal("29999.0")
    assert update[0].bid_size == Decimal("3.0")
    assert update[0].ask == Decimal("30001.0")
    assert update[0].ask_size == Decimal("2.5")


def test_kraken_update_before_snapshot_fails_closed():
    book = KrakenBook()
    with pytest.raises(ExternalFeedError):
        book.apply(
            {
                "channel": "book",
                "type": "update",
                "data": [
                    {
                        "symbol": "BTC/USD",
                        "bids": [{"price": 30000.0, "qty": 1.0}],
                        "asks": [],
                        "checksum": 1,
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ],
            },
            recv_ts_ns=1_704_067_200_100_000_000,
        )


def test_kraken_negative_quantity_fails_closed():
    book = KrakenBook()
    with pytest.raises(ExternalFeedError):
        book.apply(
            {
                "channel": "book",
                "type": "snapshot",
                "data": [
                    {
                        "symbol": "BTC/USD",
                        "bids": [{"price": 30000.0, "qty": -1.0}],
                        "asks": [{"price": 30001.0, "qty": 1.0}],
                        "checksum": 1,
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ],
            },
            recv_ts_ns=1_704_067_200_100_000_000,
        )
