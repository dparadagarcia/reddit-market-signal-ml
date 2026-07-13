from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tfe_reddit.features.market_features import build_weekly_market_features
from tfe_reddit.features.text_features import build_weekly_text_features


def build_weekly_dataset(
    reddit_raw_df: pd.DataFrame,
    market_raw_df: pd.DataFrame,
    base_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Construye dataset semanal con features Reddit + mercado y etiqueta futura."""
    text_weekly = build_weekly_text_features(
        reddit_df=reddit_raw_df,
        keyword_lexicon=base_cfg["features"].get("keyword_lexicon", []),
        min_text_length=int(base_cfg["reddit"].get("min_text_length", 0)),
        week_frequency=base_cfg["time"].get("week_frequency", "W-SUN"),
        sentiment_backend=base_cfg["features"].get("sentiment_backend", "vader"),
        finbert_model_name=base_cfg["features"].get("finbert_model_name", "ProsusAI/finbert"),
        finbert_batch_size=int(base_cfg["features"].get("finbert_batch_size", 16)),
        finbert_max_length=int(base_cfg["features"].get("finbert_max_length", 256)),
    )
    mkt_weekly = build_weekly_market_features(
        market_df=market_raw_df,
        week_frequency=base_cfg["time"].get("week_frequency", "W-SUN"),
    )

    ds = mkt_weekly.merge(text_weekly, on=["asset", "week_start"], how="left")

    if "weekly_text" not in ds.columns:
        ds["weekly_text"] = ""

    ds["weekly_text"] = ds["weekly_text"].fillna("")

    reddit_num_cols = [c for c in ds.columns if c.startswith("reddit_") and c != "weekly_text"]
    if reddit_num_cols:
        ds[reddit_num_cols] = ds[reddit_num_cols].fillna(0.0)

    ds = ds.sort_values(["asset", "week_start"]).reset_index(drop=True)

    horizon = int(base_cfg["label"].get("horizon_weeks", 1))
    neutral_th = float(base_cfg["label"].get("neutral_threshold", 0.0))
    drop_neutral = bool(base_cfg["label"].get("drop_neutral", True))

    ds["future_return_h"] = ds.groupby("asset")["mkt_ret_1w"].shift(-horizon)
    ds["target_up"] = np.where(
        ds["future_return_h"] > neutral_th,
        1,
        np.where(ds["future_return_h"] < -neutral_th, 0, np.nan),
    )

    # Baseline ingenuo (persistencia del signo de la última semana observada)
    ds["naive_persistence_signal"] = (ds["mkt_ret_1w"] > 0).astype(int)

    ds = ds.dropna(subset=["future_return_h"]).copy()
    if drop_neutral:
        ds = ds.dropna(subset=["target_up"]).copy()

    ds["target_up"] = ds["target_up"].astype(int)
    return ds.sort_values(["week_start", "asset"]).reset_index(drop=True)
