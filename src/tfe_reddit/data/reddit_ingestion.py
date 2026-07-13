from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any
import warnings

import pandas as pd
import praw
import requests
from dotenv import load_dotenv
from tqdm import tqdm


def _parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_reddit_client_from_env() -> praw.Reddit:
    """Construye cliente de Reddit usando variables de entorno."""
    load_dotenv()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not client_id or not client_secret or not user_agent:
        raise ValueError(
            "Faltan credenciales de Reddit en .env (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _month_windows(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime]]:
    start_ts = pd.Timestamp(start_dt).tz_convert("UTC")
    end_ts = pd.Timestamp(end_dt).tz_convert("UTC")
    cursor = start_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    windows: list[tuple[datetime, datetime]] = []
    while cursor < end_ts:
        next_cursor = cursor + pd.offsets.MonthBegin(1)
        windows.append((cursor.to_pydatetime(), min(next_cursor.to_pydatetime(), end_dt)))
        cursor = next_cursor
    return windows


def _fetch_reddit_posts_arctic_shift(base_cfg: dict[str, Any]) -> pd.DataFrame:
    start_dt = _parse_iso_datetime(base_cfg["time"]["start_date"])
    end_dt = _parse_iso_datetime(base_cfg["time"]["end_date"])
    max_posts = int(base_cfg["reddit"].get("max_posts_per_query", 500))

    endpoint = "https://arctic-shift.photon-reddit.com/api/posts/search"
    windows = _month_windows(start_dt, end_dt)
    rows: list[dict[str, Any]] = []

    for asset, asset_cfg in base_cfg["assets"].items():
        subreddits = asset_cfg.get("subreddits", [])
        keywords = asset_cfg.get("keywords", [])

        for subreddit_name in subreddits:
            for keyword in tqdm(keywords, desc=f"{asset}/{subreddit_name}"):
                collected = 0
                for window_start, window_end in windows:
                    if collected >= max_posts:
                        break

                    after_dt = window_start
                    while after_dt < window_end and collected < max_posts:
                        batch_limit = min(100, max_posts - collected)
                        try:
                            resp = requests.get(
                                endpoint,
                                params={
                                    "subreddit": subreddit_name,
                                    "title": keyword,
                                    "after": after_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "before": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "limit": batch_limit,
                                    "sort": "asc",
                                },
                                timeout=30,
                            )
                            resp.raise_for_status()
                        except requests.RequestException:
                            break
                        data = resp.json().get("data", [])
                        if not data:
                            break

                        last_created_utc = None
                        for post in data:
                            created_dt = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
                            text = f"{post.get('title') or ''} {post.get('selftext') or ''}".strip()
                            rows.append(
                                {
                                    "asset": asset,
                                    "subreddit": subreddit_name,
                                    "source_type": "submission",
                                    "source_id": post.get("id"),
                                    "created_utc": created_dt,
                                    "author": post.get("author") or "[deleted]",
                                    "score": int(post.get("score") or 0),
                                    "num_comments": int(post.get("num_comments") or 0),
                                    "text": text,
                                    "keyword": keyword,
                                }
                            )
                            collected += 1
                            last_created_utc = int(post["created_utc"])
                            if collected >= max_posts:
                                break

                        if last_created_utc is None:
                            break
                        next_after = datetime.fromtimestamp(last_created_utc + 1, tz=timezone.utc)
                        if next_after <= after_dt:
                            break
                        after_dt = next_after
                        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(
            columns=[
                "asset",
                "subreddit",
                "source_type",
                "source_id",
                "created_utc",
                "author",
                "score",
                "num_comments",
                "text",
                "keyword",
            ]
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["source_type", "source_id", "asset"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df.sort_values(["asset", "created_utc"]).reset_index(drop=True)


def fetch_reddit_posts(base_cfg: dict[str, Any]) -> pd.DataFrame:
    """Descarga posts de Reddit por activo/subreddit/keyword."""
    source = str(base_cfg.get("reddit", {}).get("source", "arctic_shift")).lower().strip()
    if source == "arctic_shift":
        return _fetch_reddit_posts_arctic_shift(base_cfg)

    reddit = build_reddit_client_from_env()
    start_dt = _parse_iso_datetime(base_cfg["time"]["start_date"])
    end_dt = _parse_iso_datetime(base_cfg["time"]["end_date"])

    max_posts = int(base_cfg["reddit"].get("max_posts_per_query", 500))
    include_comments = bool(base_cfg["reddit"].get("include_comments", False))
    max_comments_per_post = int(base_cfg["reddit"].get("max_comments_per_post", 20))

    rows: list[dict[str, Any]] = []

    for asset, asset_cfg in base_cfg["assets"].items():
        subreddits = asset_cfg.get("subreddits", [])
        keywords = asset_cfg.get("keywords", [])

        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            for keyword in tqdm(keywords, desc=f"{asset}/{subreddit_name}"):
                try:
                    for submission in subreddit.search(keyword, sort="new", time_filter="all", limit=max_posts):
                        created_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                        if created_dt < start_dt or created_dt > end_dt:
                            continue

                        text = f"{submission.title or ''} {submission.selftext or ''}".strip()
                        rows.append(
                            {
                                "asset": asset,
                                "subreddit": subreddit_name,
                                "source_type": "submission",
                                "source_id": submission.id,
                                "created_utc": created_dt,
                                "author": str(submission.author) if submission.author else "[deleted]",
                                "score": int(submission.score or 0),
                                "num_comments": int(submission.num_comments or 0),
                                "text": text,
                                "keyword": keyword,
                            }
                        )

                        if not include_comments:
                            continue

                        submission.comments.replace_more(limit=0)
                        comments = submission.comments.list()[:max_comments_per_post]
                        for comment in comments:
                            c_text = (comment.body or "").strip()
                            if not c_text:
                                continue
                            c_created_dt = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
                            if c_created_dt < start_dt or c_created_dt > end_dt:
                                continue
                            rows.append(
                                {
                                    "asset": asset,
                                    "subreddit": subreddit_name,
                                    "source_type": "comment",
                                    "source_id": comment.id,
                                    "created_utc": c_created_dt,
                                    "author": str(comment.author) if comment.author else "[deleted]",
                                    "score": int(comment.score or 0),
                                    "num_comments": 0,
                                    "text": c_text,
                                    "keyword": keyword,
                                }
                            )
                except (praw.exceptions.PRAWException, requests.RequestException) as exc:
                    warnings.warn(
                        f"Error recuperando Reddit para {asset}/{subreddit_name}/{keyword}: {exc}",
                        stacklevel=2,
                    )
                    continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "asset",
                "subreddit",
                "source_type",
                "source_id",
                "created_utc",
                "author",
                "score",
                "num_comments",
                "text",
                "keyword",
            ]
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["source_type", "source_id", "asset"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df.sort_values(["asset", "created_utc"]).reset_index(drop=True)
