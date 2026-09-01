from kalshi_research.math.execution import conservative_maker_fill_qty, walk_asks


def test_visible_depth_walk_partial_fill():
    fill = walk_asks([(0.40, 2), (0.42, 3)], 6)
    assert fill.filled_qty == 5
    assert fill.unfilled_qty == 1
    assert round(fill.average_price, 3) == 0.412


def test_conservative_queue_model_requires_trade_through_queue():
    assert conservative_maker_fill_qty(10, 9, order_qty=5) == 0
    assert conservative_maker_fill_qty(10, 13, order_qty=5) == 3
