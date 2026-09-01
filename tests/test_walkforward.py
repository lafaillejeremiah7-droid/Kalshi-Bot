from kalshi_research.research.walkforward import expanding_walkforward


def test_walkforward_groups_whole_markets():
    ids = [f"m{i}" for i in range(20)]
    folds = expanding_walkforward(ids, min_train=8, validation_size=3, test_size=3)
    assert folds
    first = folds[0]
    assert len(first.train) == 8
    assert len(first.validation) == 3
    assert len(first.test) == 3
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.validation).isdisjoint(first.test)
