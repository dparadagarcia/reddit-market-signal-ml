from __future__ import annotations

import io
from typing import Any

import pandas as pd
import requests


BINANCE_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "DOGE-USD": "DOGEUSDT",
}


def _to_unix_seconds(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp())


def _download_binance_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    rows: list[list] = []
    start_ms = _to_unix_seconds(start_date) * 1000
    end_ms = _to_unix_seconds(end_date) * 1000

    while start_ms < end_ms:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1d",
                "limit": 1000,
                "startTime": start_ms,
                "endTime": end_ms,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        rows.extend(data)
        last_open_time = int(data[-1][0])
        next_start = last_open_time + 24 * 60 * 60 * 1000
        if next_start <= start_ms:
            break
        start_ms = next_start

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["open_time"], unit="ms", utc=True),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _download_fred_sp500_proxy(start_date: str, end_date: str) -> pd.DataFrame:
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": "SP500"},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns={"observation_date": "date", "SP500": "close"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)].copy()
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0.0
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def download_market_daily(base_cfg: dict[str, Any]) -> pd.DataFrame:
    """Descarga históricos diarios de mercado para los activos configurados."""
    start_date = base_cfg["time"]["start_date"]
    end_date = base_cfg["time"]["end_date"]

    rows: list[pd.DataFrame] = []
    for asset, asset_cfg in base_cfg["assets"].items():
        ticker = asset_cfg["ticker"]
        if ticker in BINANCE_SYMBOL_MAP:
            hist = _download_binance_daily(
                symbol=BINANCE_SYMBOL_MAP[ticker],
                start_date=start_date,
                end_date=end_date,
            )
            source_ticker = ticker
        elif asset == "SPY":
            hist = _download_fred_sp500_proxy(start_date=start_date, end_date=end_date)
            source_ticker = "SP500_FRED_PROXY"
        else:
            hist = pd.DataFrame()

        if hist.empty:
            continue

        hist["asset"] = asset
        hist["ticker"] = source_ticker
        rows.append(hist[["date", "asset", "ticker", "open", "high", "low", "close", "volume"]])

    if not rows:
        return pd.DataFrame(columns=["date", "asset", "ticker", "open", "high", "low", "close", "volume"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out.sort_values(["asset", "date"]).reset_index(drop=True)
