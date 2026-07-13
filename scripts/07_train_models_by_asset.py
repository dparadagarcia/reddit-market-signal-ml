from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from pathlib import Path

import pandas as pd

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.modeling.train import train_and_evaluate
from tfe_reddit.utils.io import read_dataframe


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")
    models_cfg = load_yaml_config("configs/models.yaml").get("models", {})

    ds_path = resolve_path(base_cfg["paths"]["weekly_dataset"])
    dataset = read_dataframe(ds_path)

    reports_root = resolve_path(base_cfg["paths"]["reports_dir"]) / "per_asset"
    models_root = resolve_path(base_cfg["paths"]["models_dir"]) / "per_asset"
    reports_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    cv_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []

    for asset in sorted(dataset["asset"].dropna().unique().tolist()):
        asset_df = dataset[dataset["asset"] == asset].copy()
        asset_reports_dir = reports_root / asset.lower()
        asset_models_dir = models_root / asset.lower()

        results = train_and_evaluate(
            dataset=asset_df,
            base_cfg=base_cfg,
            models_cfg=models_cfg,
            models_dir=asset_models_dir,
            reports_dir=asset_reports_dir,
        )

        cv_summary = results["cv_summary"].copy()
        cv_summary["asset"] = asset
        cv_rows.append(cv_summary)

        test_metrics = results["test_metrics"].copy()
        test_metrics["asset"] = asset
        test_rows.append(test_metrics)

        best_row = test_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
        summary_rows.append(
            {
                "asset": asset,
                "best_model_test": best_row["model"],
                "best_balanced_accuracy_test": float(best_row["balanced_accuracy"]),
                "best_f1_test": float(best_row["f1"]),
                "best_roc_auc_test": float(best_row["roc_auc"]),
                "n_weeks": int(asset_df["week_start"].nunique()),
                "n_rows": int(len(asset_df)),
            }
        )
        print(
            f"[OK] {asset}: mejor modelo test={best_row['model']} | "
            f"balanced_accuracy={best_row['balanced_accuracy']:.3f}"
        )

    pd.DataFrame(summary_rows).to_csv(reports_root / "summary_best_models.csv", index=False)
    pd.concat(cv_rows, ignore_index=True).to_csv(reports_root / "metrics_cv_summary_all_assets.csv", index=False)
    pd.concat(test_rows, ignore_index=True).to_csv(reports_root / "metrics_test_all_assets.csv", index=False)
    print(f"[OK] Resumen agregado guardado en {reports_root}")


if __name__ == "__main__":
    main()
