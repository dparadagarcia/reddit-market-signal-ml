from __future__ import annotations

import pandas as pd
import pytest

from tfe_reddit.data.weekly_dataset import build_weekly_dataset
from tfe_reddit.evaluation.metrics import compute_classification_metrics
from tfe_reddit.modeling.validation import generate_expanding_splits


def test_generate_expanding_splits_preserves_temporal_order() -> None:
    weeks = list(pd.date_range("2024-01-01", periods=10, freq="W-MON", tz="UTC"))

    folds = list(generate_expanding_splits(weeks=weeks, train_min_weeks=4, val_weeks=2, step_weeks=2))

    assert len(folds) == 3
    for train_weeks, val_weeks in folds:
        assert max(train_weeks) < min(val_weeks)
        assert len(val_weeks) == 2

    assert len(folds[0][0]) == 4
    assert len(folds[1][0]) == 6
    assert len(folds[2][0]) == 8


def test_compute_classification_metrics_includes_confusion_matrix_and_auc() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_score=[0.1, 0.6, 0.7, 0.8],
    )

    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)


def test_build_weekly_dataset_creates_forward_label_without_neutral_rows() -> None:
    reddit_raw = pd.DataFrame(
        [
            {
                "asset": "BTC",
                "created_utc": "2024-01-01T12:00:00Z",
                "text": "bitcoin rally looks strong",
                "source_id": "r1",
                "author": "u1",
                "score": 2,
                "num_comments": 1,
                "subreddit": "Bitcoin",
            },
            {
                "asset": "BTC",
                "created_utc": "2024-01-15T12:00:00Z",
                "text": "bitcoin sell pressure",
                "source_id": "r2",
                "author": "u2",
                "score": 1,
                "num_comments": 0,
                "subreddit": "Bitcoin",
            },
        ]
    )
    market_raw = pd.DataFrame(
        [
            {"asset": "BTC", "date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
            {"asset": "BTC", "date": "2024-01-08", "open": 100, "high": 112, "low": 99, "close": 110, "volume": 12},
            {"asset": "BTC", "date": "2024-01-15", "open": 110, "high": 111, "low": 90, "close": 99, "volume": 14},
            {"asset": "BTC", "date": "2024-01-22", "open": 99, "high": 121, "low": 98, "close": 120, "volume": 16},
        ]
    )
    cfg = {
        "features": {
            "keyword_lexicon": ["bitcoin", "rally", "sell"],
            "sentiment_backend": "vader",
        },
        "reddit": {"min_text_length": 1},
        "time": {"week_frequency": "W-SUN"},
        "label": {"horizon_weeks": 1, "neutral_threshold": 0.002, "drop_neutral": True},
    }

    out = build_weekly_dataset(reddit_raw_df=reddit_raw, market_raw_df=market_raw, base_cfg=cfg)

    assert len(out) == 3
    assert out.loc[0, "future_return_h"] == pytest.approx(0.10)
    assert out.loc[0, "target_up"] == 1
    assert out.loc[1, "future_return_h"] == pytest.approx(-0.10)
    assert out.loc[1, "target_up"] == 0
    assert "reddit_post_count" in out.columns
