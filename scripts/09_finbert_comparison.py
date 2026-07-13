from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from copy import deepcopy
import os
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.data.weekly_dataset import build_weekly_dataset
from tfe_reddit.modeling.train import train_and_evaluate
from tfe_reddit.utils.io import read_dataframe, save_dataframe


def _finbert_cfg(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["features"]["sentiment_backend"] = "hybrid"
    cfg["features"]["finbert_batch_size"] = 8
    cfg["features"]["finbert_max_length"] = 160
    cfg["features"]["tfidf_max_features"] = min(int(cfg["features"].get("tfidf_max_features", 8000)), 2500)
    cfg["paths"]["weekly_dataset"] = "data/processed/weekly_dataset_hybrid_finbert.parquet"
    cfg["paths"]["models_dir"] = "models/finbert_hybrid"
    cfg["paths"]["reports_dir"] = "reports/finbert_hybrid"
    return cfg


def _comparison_row(name: str, metrics_path: Path) -> dict[str, Any]:
    metrics = pd.read_csv(metrics_path)
    best = metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
    return {
        "experiment": name,
        "best_model": best["model"],
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "f1": float(best["f1"]),
        "roc_auc": float(best["roc_auc"]),
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_cfg = load_yaml_config("configs/base.yaml")
    models_cfg = load_yaml_config("configs/models.yaml").get("models", {})
    cfg = _finbert_cfg(base_cfg)

    reddit_df = read_dataframe(resolve_path(cfg["paths"]["raw_reddit"]))
    market_df = read_dataframe(resolve_path(cfg["paths"]["raw_market"]))
    dataset = build_weekly_dataset(reddit_raw_df=reddit_df, market_raw_df=market_df, base_cfg=cfg)

    dataset_path = resolve_path(cfg["paths"]["weekly_dataset"])
    save_dataframe(dataset, dataset_path)
    print(f"[OK] Dataset híbrido VADER+FinBERT guardado en {dataset_path} | filas={len(dataset)}", flush=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        results = train_and_evaluate(
            dataset=dataset,
            base_cfg=cfg,
            models_cfg=models_cfg,
            models_dir=resolve_path(cfg["paths"]["models_dir"]),
            reports_dir=resolve_path(cfg["paths"]["reports_dir"]),
        )
    for key, df in results.items():
        print(f" - {key}: {len(df)} filas", flush=True)

    base_reports = resolve_path(base_cfg["paths"]["reports_dir"])
    finbert_reports = resolve_path(cfg["paths"]["reports_dir"])
    comparison = pd.DataFrame(
        [
            _comparison_row("VADER", base_reports / "metrics_test.csv"),
            _comparison_row("VADER+FinBERT", finbert_reports / "metrics_test.csv"),
        ]
    )
    out_path = base_reports / "finbert_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"[OK] Comparativa guardada en {out_path}", flush=True)


if __name__ == "__main__":
    main()
