from __future__ import annotations

import numpy as np
import pandas as pd


def _to_week_start(ts: pd.Series, freq: str = "W-SUN") -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.to_period(freq).dt.start_time.dt.tz_localize("UTC")


def build_weekly_market_features(market_df: pd.DataFrame, week_frequency: str = "W-SUN") -> pd.DataFrame:
    """Genera variables semanales de mercado y lags históricos."""
    if market_df.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "week_start",
                "mkt_open",
                "mkt_high",
                "mkt_low",
                "mkt_close",
                "mkt_volume",
                "mkt_volatility_1w",
                "mkt_ret_1w",
                "mkt_ret_lag1",
                "mkt_ret_lag2",
                "mkt_ret_lag3",
                "mkt_ret_roll4",
                "mkt_ret_roll12",
            ]
        )

    work = market_df.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True)
    work = work.sort_values(["asset", "date"]).reset_index(drop=True)

    work["daily_ret"] = work.groupby("asset")["close"].pct_change()
    work["week_start"] = _to_week_start(work["date"], freq=week_frequency)

    weekly = work.groupby(["asset", "week_start"], as_index=False).agg(
        mkt_open=("open", "first"),
        mkt_high=("high", "max"),
        mkt_low=("low", "min"),
        mkt_close=("close", "last"),
        mkt_volume=("volume", "sum"),
        mkt_volatility_1w=("daily_ret", "std"),
    )

    weekly["mkt_ret_1w"] = weekly.groupby("asset")["mkt_close"].pct_change()
    weekly["mkt_ret_lag1"] = weekly.groupby("asset")["mkt_ret_1w"].shift(1)
    weekly["mkt_ret_lag2"] = weekly.groupby("asset")["mkt_ret_1w"].shift(2)
    weekly["mkt_ret_lag3"] = weekly.groupby("asset")["mkt_ret_1w"].shift(3)

    weekly["mkt_ret_roll4"] = (
        weekly.groupby("asset")["mkt_ret_1w"].transform(lambda s: s.shift(1).rolling(4).mean())
    )
    weekly["mkt_ret_roll12"] = (
        weekly.groupby("asset")["mkt_ret_1w"].transform(lambda s: s.shift(1).rolling(12).mean())
    )

    weekly["mkt_volatility_1w"] = weekly["mkt_volatility_1w"].fillna(0.0)
    num_cols = [c for c in weekly.columns if c.startswith("mkt_")]
    weekly[num_cols] = weekly[num_cols].replace([np.inf, -np.inf], np.nan)
    return weekly.sort_values(["asset", "week_start"]).reset_index(drop=True)
