from __future__ import annotations

import lzma
import struct
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from xau_company.config import Settings
from xau_company.data import DukascopyClient


def _candle_payload(records: list[tuple[int, int, int, int, int, float]]) -> bytes:
    raw = b"".join(struct.pack(">5If", *record) for record in records)
    return lzma.compress(raw)


def _tick_payload(records: list[tuple[int, int, int, float, float]]) -> bytes:
    raw = b"".join(struct.pack(">IIIff", *record) for record in records)
    return lzma.compress(raw)


def test_settings_validate_without_market_data_credentials() -> None:
    Settings(paper_mode=True).validate()


def test_dukascopy_url_uses_zero_based_month() -> None:
    url = DukascopyClient._day_url(
        "https://datafeed.dukascopy.com/datafeed",
        "XAUUSD",
        date(2026, 8, 31),
    )
    assert "/2026/07/31/BID_candles_min_1.bi5" in url


def test_decode_oclh_candles_and_scale_xau() -> None:
    # Dukascopy O,C,L,H layout; divisor=1000 for XAU/USD.
    payload = _candle_payload(
        [
            (0, 3500000, 3501000, 3499000, 3502000, 10.0),
            (60, 3501000, 3500500, 3498000, 3503000, 12.0),
        ]
    )
    frame = DukascopyClient.decode_candle_payload(payload, date(2026, 8, 31), 1000.0)
    assert list(frame.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert frame.iloc[0]["open"] == 3500.0
    assert frame.iloc[0]["close"] == 3501.0
    assert frame.iloc[0]["low"] == 3499.0
    assert frame.iloc[0]["high"] == 3502.0


def test_decoder_accepts_ohlc_layout_when_invariants_make_it_clear() -> None:
    # Alternative O,H,L,C layout. The decoder chooses the interpretation that
    # satisfies high/low invariants across the payload.
    payload = _candle_payload(
        [
            (0, 3500000, 3503000, 3499000, 3501000, 10.0),
            (60, 3501000, 3504000, 3500000, 3502000, 11.0),
        ]
    )
    frame = DukascopyClient.decode_candle_payload(payload, date(2026, 8, 31), 1000.0)
    assert frame.iloc[-1]["high"] == 3504.0
    assert frame.iloc[-1]["close"] == 3502.0


def test_decode_tick_returns_latest_mid_price() -> None:
    hour = datetime(2026, 8, 31, 18, tzinfo=timezone.utc)
    payload = _tick_payload(
        [
            (1000, 3501100, 3500900, 1.0, 1.0),
            (2500, 3502200, 3501800, 1.0, 1.0),
        ]
    )
    latest = DukascopyClient.decode_tick_payload(payload, hour, 1000.0)
    assert latest is not None
    stamp, mid = latest
    assert stamp == hour + timedelta(milliseconds=2500)
    assert mid == 3502.0


def test_resample_builds_15m_from_one_minute_source() -> None:
    start = pd.Timestamp("2026-08-31T18:00:00Z")
    rows = []
    for i in range(30):
        value = 3500.0 + i
        rows.append(
            {
                "datetime": start + pd.Timedelta(minutes=i),
                "open": value,
                "high": value + 2,
                "low": value - 2,
                "close": value + 1,
                "volume": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    out = DukascopyClient._resample(frame, "15min")
    assert len(out) == 2
    assert out.iloc[0]["open"] == 3500.0
    assert out.iloc[0]["close"] == 3515.0
    assert out.iloc[0]["high"] == 3516.0
    assert out.iloc[0]["low"] == 3498.0
    assert out.iloc[0]["volume"] == 15.0


def test_dxy_is_supported_but_fake_us10y_substitution_is_not() -> None:
    assert DukascopyClient._instrument("DXY")[0] == "DOLLARIDXUSD"
    client = DukascopyClient(retries=1)
    assert client.safe_candles("US10Y", "1h", 10) is None
