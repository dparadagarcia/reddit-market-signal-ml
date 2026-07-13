from __future__ import annotations

import pandas as pd
import pytest

from tfe_reddit.features.text_features import build_weekly_text_features


def _sample_reddit_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "BTC",
                "created_utc": "2024-01-03T10:00:00Z",
                "text": "Bitcoin looks strong and market sentiment is positive",
                "source_id": "a1",
                "author": "u1",
                "score": 10,
                "num_comments": 3,
                "subreddit": "Bitcoin",
            },
            {
                "asset": "BTC",
                "created_utc": "2024-01-04T12:00:00Z",
                "text": "BTC could dump after the rally but maybe recovers later",
                "source_id": "a2",
                "author": "u2",
                "score": 4,
                "num_comments": 1,
                "subreddit": "CryptoCurrency",
            },
        ]
    )


def test_build_weekly_text_features_vader_backend() -> None:
    out = build_weekly_text_features(
        reddit_df=_sample_reddit_df(),
        keyword_lexicon=["bitcoin", "btc", "dump"],
        min_text_length=5,
        week_frequency="W-SUN",
        sentiment_backend="vader",
    )

    assert len(out) == 1
    assert "reddit_sentiment_mean" in out.columns
    assert "reddit_finbert_sentiment_mean" not in out.columns
    assert out.loc[0, "reddit_post_count"] == 2


def test_build_weekly_text_features_hybrid_backend(monkeypatch) -> None:
    def fake_score_financial_sentiment(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "finbert_positive": 0.80,
                    "finbert_negative": 0.10,
                    "finbert_neutral": 0.10,
                    "finbert_sentiment": 0.70,
                },
                {
                    "finbert_positive": 0.20,
                    "finbert_negative": 0.60,
                    "finbert_neutral": 0.20,
                    "finbert_sentiment": -0.40,
                },
            ]
        )

    monkeypatch.setattr(
        "tfe_reddit.features.text_features.score_financial_sentiment",
        fake_score_financial_sentiment,
    )

    out = build_weekly_text_features(
        reddit_df=_sample_reddit_df(),
        keyword_lexicon=["bitcoin", "btc", "dump"],
        min_text_length=5,
        week_frequency="W-SUN",
        sentiment_backend="hybrid",
    )

    assert len(out) == 1
    assert "reddit_finbert_sentiment_mean" in out.columns
    assert out.loc[0, "reddit_finbert_sentiment_mean"] == pytest.approx(0.15)
    assert out.loc[0, "reddit_finbert_sent_pos_share"] == 0.5
    assert out.loc[0, "reddit_finbert_sent_neg_share"] == 0.5
