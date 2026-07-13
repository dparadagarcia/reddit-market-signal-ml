from __future__ import annotations

from collections.abc import Iterator


def generate_expanding_splits(
    weeks: list,
    train_min_weeks: int,
    val_weeks: int,
    step_weeks: int,
) -> Iterator[tuple[list, list]]:
    """Genera folds temporales train/validation con ventana expansiva."""
    if len(weeks) < train_min_weeks + val_weeks:
        return

    end = train_min_weeks
    while end + val_weeks <= len(weeks):
        train_window = weeks[:end]
        val_window = weeks[end : end + val_weeks]
        yield train_window, val_window
        end += step_weeks
