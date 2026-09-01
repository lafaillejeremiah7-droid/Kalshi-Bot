from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def expanding_walkforward(
    ordered_market_ids: Sequence[str],
    min_train: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
) -> list[WalkForwardFold]:
    """Split by whole market IDs so seconds from one contract never cross folds."""
    if min(min_train, validation_size, test_size) <= 0:
        raise ValueError("split sizes must be positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")

    folds: list[WalkForwardFold] = []
    train_end = min_train
    while train_end + validation_size + test_size <= len(ordered_market_ids):
        val_end = train_end + validation_size
        test_end = val_end + test_size
        folds.append(
            WalkForwardFold(
                train=tuple(ordered_market_ids[:train_end]),
                validation=tuple(ordered_market_ids[train_end:val_end]),
                test=tuple(ordered_market_ids[val_end:test_end]),
            )
        )
        train_end += step
    return folds
