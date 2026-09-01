from kalshi_research.sync.asof import ReceiveTimeSeries, TimedValue


def test_asof_never_uses_future_value_and_enforces_freshness():
    series = ReceiveTimeSeries([TimedValue(100, "a"), TimedValue(200, "b")])
    assert series.asof(199).value == "a"
    assert series.asof(200).value == "b"
    assert series.asof(250, max_age_ns=40) is None
