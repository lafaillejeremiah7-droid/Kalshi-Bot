from kalshi_research.math.features import book_imbalance, microprice, normalized_log_distance


def test_book_imbalance_bounds():
    assert book_imbalance(10, 0) == 1
    assert book_imbalance(0, 10) == -1
    assert book_imbalance(0, 0) == 0


def test_microprice_shifts_toward_thin_side():
    mp = microprice(0.40, 100, 0.42, 10)
    assert 0.40 < mp < 0.42
    assert mp > 0.41


def test_normalized_distance_sign():
    assert normalized_log_distance(101, 100, 0.001, 60) > 0
    assert normalized_log_distance(99, 100, 0.001, 60) < 0
