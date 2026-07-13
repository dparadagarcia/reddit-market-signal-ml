from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tfe_reddit.features.finbert import score_financial_sentiment


URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_WORD_RE = re.compile(r"[^\w\s#@.$%+-]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = URL_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _to_week_start(ts: pd.Series, freq: str = "W-SUN") -> pd.Series:
    utc_naive = pd.to_datetime(ts, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return utc_naive.dt.to_period(freq).dt.start_time.dt.tz_localize("UTC")


def build_weekly_text_features(
    reddit_df: pd.DataFrame,
    keyword_lexicon: list[str],
    min_text_length: int,
    week_frequency: str,
    sentiment_backend: str = "vader",
    finbert_model_name: str = "ProsusAI/finbert",
    finbert_batch_size: int = 16,
    finbert_max_length: int = 256,
) -> pd.DataFrame:
    """Genera agregados semanales de actividad, engagement, sentimiento y texto."""
    sentiment_backend = sentiment_backend.lower().strip()
    if sentiment_backend not in {"vader", "finbert", "hybrid"}:
        raise ValueError("sentiment_backend debe ser 'vader', 'finbert' o 'hybrid'")
    use_vader = sentiment_backend in {"vader", "hybrid"}
    use_finbert = sentiment_backend in {"finbert", "hybrid"}

    if reddit_df.empty:
        base_cols = [
            "asset",
            "week_start",
            "weekly_text",
            "reddit_post_count",
            "reddit_unique_authors",
            "reddit_score_sum",
            "reddit_score_mean",
            "reddit_comments_sum",
            "reddit_comments_mean",
            "reddit_text_len_mean",
            "reddit_keyword_hits_sum",
            "reddit_subreddit_nunique",
        ]
        vader_cols = [
            "reddit_sentiment_mean",
            "reddit_sentiment_std",
            "reddit_sent_pos_share",
            "reddit_sent_neg_share",
        ]
        finbert_cols = [
            "reddit_finbert_positive_mean",
            "reddit_finbert_negative_mean",
            "reddit_finbert_neutral_mean",
            "reddit_finbert_sentiment_mean",
            "reddit_finbert_sentiment_std",
            "reddit_finbert_sent_pos_share",
            "reddit_finbert_sent_neg_share",
        ]
        cols = list(base_cols)
        if use_vader:
            cols.extend(vader_cols)
        if use_finbert:
            cols.extend(finbert_cols)
        return pd.DataFrame(columns=cols)

    work = reddit_df.copy()
    work["text"] = work["text"].fillna("").astype(str)
    work["clean_text"] = work["text"].map(clean_text)
    work = work[work["clean_text"].str.len() >= min_text_length].copy()
    if work.empty:
        return build_weekly_text_features(
            pd.DataFrame(),
            keyword_lexicon=keyword_lexicon,
            min_text_length=min_text_length,
            week_frequency=week_frequency,
            sentiment_backend=sentiment_backend,
            finbert_model_name=finbert_model_name,
            finbert_batch_size=finbert_batch_size,
            finbert_max_length=finbert_max_length,
        )

    work["text_len"] = work["clean_text"].str.len()

    if use_vader:
        analyzer = SentimentIntensityAnalyzer()
        work["sentiment_compound"] = work["clean_text"].map(lambda x: analyzer.polarity_scores(x)["compound"])

    if use_finbert:
        try:
            finbert_scores = score_financial_sentiment(
                texts=work["clean_text"].tolist(),
                model_name=finbert_model_name,
                batch_size=finbert_batch_size,
                max_length=finbert_max_length,
            )
        except ImportError:
            if sentiment_backend == "finbert":
                raise
            use_finbert = False
            warnings.warn(
                "FinBERT no está disponible en el entorno actual; el backend 'hybrid' continuará con VADER.",
                stacklevel=2,
            )
        else:
            work = pd.concat([work.reset_index(drop=True), finbert_scores], axis=1)

    lexicon = [k.lower() for k in keyword_lexicon]
    work["keyword_hits"] = work["clean_text"].map(lambda x: sum(x.count(k) for k in lexicon))

    work["week_start"] = _to_week_start(work["created_utc"], freq=week_frequency)

    agg_spec = {
        "weekly_text": ("clean_text", lambda s: " ".join(s.astype(str).tolist())),
        "reddit_post_count": ("source_id", "count"),
        "reddit_unique_authors": ("author", lambda s: s.fillna("[unknown]").nunique()),
        "reddit_score_sum": ("score", "sum"),
        "reddit_score_mean": ("score", "mean"),
        "reddit_comments_sum": ("num_comments", "sum"),
        "reddit_comments_mean": ("num_comments", "mean"),
        "reddit_text_len_mean": ("text_len", "mean"),
        "reddit_keyword_hits_sum": ("keyword_hits", "sum"),
        "reddit_subreddit_nunique": ("subreddit", "nunique"),
    }
    if use_vader:
        agg_spec.update(
            {
                "reddit_sentiment_mean": ("sentiment_compound", "mean"),
                "reddit_sentiment_std": ("sentiment_compound", "std"),
                "reddit_sent_pos_share": ("sentiment_compound", lambda s: float((s > 0.05).mean())),
                "reddit_sent_neg_share": ("sentiment_compound", lambda s: float((s < -0.05).mean())),
            }
        )
    if use_finbert:
        agg_spec.update(
            {
                "reddit_finbert_positive_mean": ("finbert_positive", "mean"),
                "reddit_finbert_negative_mean": ("finbert_negative", "mean"),
                "reddit_finbert_neutral_mean": ("finbert_neutral", "mean"),
                "reddit_finbert_sentiment_mean": ("finbert_sentiment", "mean"),
                "reddit_finbert_sentiment_std": ("finbert_sentiment", "std"),
                "reddit_finbert_sent_pos_share": ("finbert_sentiment", lambda s: float((s > 0.05).mean())),
                "reddit_finbert_sent_neg_share": ("finbert_sentiment", lambda s: float((s < -0.05).mean())),
            }
        )

    grouped = work.groupby(["asset", "week_start"], as_index=False).agg(**agg_spec)

    subreddit_counts = (
        work.groupby(["asset", "week_start", "subreddit"]).size().rename("count").reset_index()
    )
    subreddit_pivot = subreddit_counts.pivot_table(
        index=["asset", "week_start"],
        columns="subreddit",
        values="count",
        fill_value=0,
    )
    subreddit_pivot.columns = [f"reddit_sub_{c}_count" for c in subreddit_pivot.columns]
    subreddit_pivot = subreddit_pivot.reset_index()

    out = grouped.merge(subreddit_pivot, on=["asset", "week_start"], how="left")
    std_cols = [c for c in ["reddit_sentiment_std", "reddit_finbert_sentiment_std"] if c in out.columns]
    for col in std_cols:
        out[col] = out[col].fillna(0.0)

    numeric_cols = [c for c in out.columns if c.startswith("reddit_") and c != "weekly_text"]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.sort_values(["asset", "week_start"]).reset_index(drop=True)
