from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import warnings

import numpy as np
import pandas as pd

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.modeling.train import train_and_evaluate
from tfe_reddit.utils.io import read_dataframe


def _relabel_dataset(dataset: pd.DataFrame, neutral_threshold: float) -> pd.DataFrame:
    out = dataset.copy()
    out["target_up"] = np.where(
        out["future_return_h"] > neutral_threshold,
        1,
        np.where(out["future_return_h"] < -neutral_threshold, 0, np.nan),
    )
    out = out.dropna(subset=["target_up"]).copy()
    out["target_up"] = out["target_up"].astype(int)
    return out


def _variant_cfg(base_cfg: dict[str, Any], *, test_weeks: int) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["validation"]["test_weeks"] = test_weeks
    cfg["features"]["tfidf_max_features"] = min(int(cfg["features"].get("tfidf_max_features", 8000)), 2500)
    return cfg


def _robustness_models(models_cfg: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(models_cfg)
    for name in selected:
        selected[name]["enabled"] = name in {"naive_persistence", "market_logreg", "hybrid_logreg"}
    return selected


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")
    models_cfg = _robustness_models(load_yaml_config("configs/models.yaml").get("models", {}))
    dataset = read_dataframe(resolve_path(base_cfg["paths"]["weekly_dataset"]))
    reports_dir = resolve_path(base_cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {"variant": "base", "neutral_threshold": 0.002, "test_weeks": 26},
        {"variant": "sin_zona_neutral", "neutral_threshold": 0.000, "test_weeks": 26},
        {"variant": "zona_neutral_amplia", "neutral_threshold": 0.005, "test_weeks": 26},
        {"variant": "test_20_semanas", "neutral_threshold": 0.002, "test_weeks": 20},
        {"variant": "test_39_semanas", "neutral_threshold": 0.002, "test_weeks": 39},
    ]

    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="tfm_robustness_") as tmp:
        tmp_root = Path(tmp)
        for variant in variants:
            variant_name = variant["variant"]
            relabeled = _relabel_dataset(dataset, neutral_threshold=float(variant["neutral_threshold"]))
            cfg = _variant_cfg(base_cfg, test_weeks=int(variant["test_weeks"]))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
                results = train_and_evaluate(
                    dataset=relabeled,
                    base_cfg=cfg,
                    models_cfg=models_cfg,
                    models_dir=tmp_root / variant_name / "models",
                    reports_dir=tmp_root / variant_name / "reports",
                )

            test_metrics = results["test_metrics"].copy()
            best = test_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
            naive = test_metrics[test_metrics["model"] == "naive_persistence"].iloc[0]

            rows.append(
                {
                    "variant": variant_name,
                    "neutral_threshold": float(variant["neutral_threshold"]),
                    "test_weeks": int(variant["test_weeks"]),
                    "n_rows": int(len(relabeled)),
                    "best_model": best["model"],
                    "best_balanced_accuracy": float(best["balanced_accuracy"]),
                    "best_f1": float(best["f1"]),
                    "best_roc_auc": float(best["roc_auc"]),
                    "naive_balanced_accuracy": float(naive["balanced_accuracy"]),
                    "delta_vs_naive": float(best["balanced_accuracy"] - naive["balanced_accuracy"]),
                }
            )
            print(
                f"[OK] {variant_name}: {best['model']} "
                f"BA={best['balanced_accuracy']:.3f} "
                f"delta={best['balanced_accuracy'] - naive['balanced_accuracy']:.3f}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "robustness_summary.csv", index=False)
    print(f"[OK] Guardado {reports_dir / 'robustness_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
