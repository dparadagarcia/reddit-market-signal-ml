from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tfe_reddit.evaluation.interpretability import export_model_importance
from tfe_reddit.evaluation.metrics import compute_classification_metrics
from tfe_reddit.modeling.models import build_pipeline
from tfe_reddit.modeling.validation import generate_expanding_splits


def _model_enabled(models_cfg: dict[str, Any], model_name: str) -> bool:
    return bool(models_cfg.get(model_name, {}).get("enabled", False))


def _predict_with_model(model_name: str, model, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    if model_name == "naive_persistence":
        pred = x["naive_persistence_signal"].astype(int).to_numpy()
        score = pred.astype(float)
        return pred, score

    pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        score = 1.0 / (1.0 + np.exp(-decision))
    else:
        score = None
    return pred.astype(int), score


def _get_numeric_feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded = {
        "asset",
        "week_start",
        "weekly_text",
        "future_return_h",
        "target_up",
    }
    out = []
    for col in dataset.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(dataset[col]):
            out.append(col)
    return out


def _compute_test_metrics_by_asset(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.drop_duplicates(subset=["week_start", "asset", "model"]).copy()
    rows: list[dict[str, Any]] = []
    for (model_name, asset), group in pred_df.groupby(["model", "asset"], sort=True):
        metrics = compute_classification_metrics(
            y_true=group["target_up"],
            y_pred=group["y_pred"],
            y_score=group["y_score"] if group["y_score"].notna().any() else None,
        )
        metrics.update(
            {
                "model": model_name,
                "asset": asset,
                "n_obs": int(len(group)),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["model", "asset"]).reset_index(drop=True)


def train_and_evaluate(
    dataset: pd.DataFrame,
    base_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Entrena, valida y evalúa modelos con protocolo temporal."""
    dataset = dataset.sort_values(["week_start", "asset"]).reset_index(drop=True)
    dataset["week_start"] = pd.to_datetime(dataset["week_start"], utc=True)

    numeric_cols = _get_numeric_feature_columns(dataset)
    if not numeric_cols:
        raise ValueError("No hay columnas numéricas disponibles para entrenar modelos")

    val_cfg = base_cfg["validation"]
    test_weeks = int(val_cfg.get("test_weeks", 26))
    all_weeks = sorted(dataset["week_start"].dropna().unique())
    if len(all_weeks) <= test_weeks + 10:
        raise ValueError("No hay suficientes semanas para un split temporal robusto")

    test_start = all_weeks[-test_weeks]
    train_val_df = dataset[dataset["week_start"] < test_start].copy()
    test_df = dataset[dataset["week_start"] >= test_start].copy()

    train_val_weeks = sorted(train_val_df["week_start"].unique())

    candidate_models = ["naive_persistence"] + [
        name for name in models_cfg.keys() if _model_enabled(models_cfg, name) and name != "naive_persistence"
    ]

    cv_rows: list[dict[str, Any]] = []
    fitted_on_trainval: dict[str, Any] = {}

    for model_name in candidate_models:
        folds = list(
            generate_expanding_splits(
                weeks=train_val_weeks,
                train_min_weeks=int(val_cfg.get("train_min_weeks", 78)),
                val_weeks=int(val_cfg.get("val_weeks", 12)),
                step_weeks=int(val_cfg.get("step_weeks", 4)),
            )
        )

        for fold_id, (train_weeks, val_weeks_block) in enumerate(folds, start=1):
            tr = train_val_df[train_val_df["week_start"].isin(train_weeks)].copy()
            va = train_val_df[train_val_df["week_start"].isin(val_weeks_block)].copy()

            x_train = tr[["weekly_text", *numeric_cols]]
            y_train = tr["target_up"]
            x_val = va[["weekly_text", *numeric_cols]]
            y_val = va["target_up"]

            if model_name == "naive_persistence":
                model = None
            else:
                model = build_pipeline(
                    model_name=model_name,
                    model_cfg=models_cfg[model_name],
                    features_cfg=base_cfg["features"],
                    numeric_columns=numeric_cols,
                )
                model.fit(x_train, y_train)

            y_pred, y_score = _predict_with_model(model_name=model_name, model=model, x=x_val)
            fold_metrics = compute_classification_metrics(y_true=y_val, y_pred=y_pred, y_score=y_score)
            fold_metrics.update(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                }
            )
            cv_rows.append(fold_metrics)

        # Entrenamiento final del modelo con train+val para test final.
        x_train_val = train_val_df[["weekly_text", *numeric_cols]]
        y_train_val = train_val_df["target_up"]
        if model_name == "naive_persistence":
            fitted_on_trainval[model_name] = None
        else:
            final_model = build_pipeline(
                model_name=model_name,
                model_cfg=models_cfg[model_name],
                features_cfg=base_cfg["features"],
                numeric_columns=numeric_cols,
            )
            final_model.fit(x_train_val, y_train_val)
            fitted_on_trainval[model_name] = final_model

    cv_df = pd.DataFrame(cv_rows)
    if cv_df.empty:
        raise ValueError("La validación temporal no generó folds; revisa configuración de ventanas")

    rank_df = (
        cv_df.groupby("model", as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "cv_balanced_accuracy_mean"})
        .sort_values("cv_balanced_accuracy_mean", ascending=False)
    )
    best_model_name = rank_df.iloc[0]["model"]

    # Evaluación final en test para todos los modelos.
    test_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    x_test = test_df[["weekly_text", *numeric_cols]]
    y_test = test_df["target_up"]

    for model_name in candidate_models:
        model = fitted_on_trainval.get(model_name)
        y_pred, y_score = _predict_with_model(model_name=model_name, model=model, x=x_test)
        m = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_score=y_score)
        m["model"] = model_name
        test_rows.append(m)

        tmp = test_df[["week_start", "asset", "target_up"]].copy()
        tmp["model"] = model_name
        tmp["y_pred"] = y_pred
        tmp["y_score"] = y_score if y_score is not None else np.nan
        pred_rows.append(tmp)

    test_df_metrics = pd.DataFrame(test_rows).sort_values("balanced_accuracy", ascending=False)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    test_by_asset_df = _compute_test_metrics_by_asset(pred_df)

    # Persistencia de resultados y mejor modelo.
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cv_df.to_csv(reports_dir / "metrics_cv_folds.csv", index=False)
    rank_df.to_csv(reports_dir / "metrics_cv_summary.csv", index=False)
    test_df_metrics.to_csv(reports_dir / "metrics_test.csv", index=False)
    test_by_asset_df.to_csv(reports_dir / "metrics_test_by_asset.csv", index=False)
    pred_df.to_csv(reports_dir / "predictions_test.csv", index=False)

    best_model_obj = fitted_on_trainval.get(best_model_name)
    if best_model_name != "naive_persistence":
        joblib.dump(best_model_obj, models_dir / "best_model.joblib")
        export_model_importance(
            model_name=best_model_name,
            fitted_pipeline=best_model_obj,
            output_path=reports_dir / "feature_importance_best_model.csv",
        )

    metadata = {
        "best_model": best_model_name,
        "numeric_columns": numeric_cols,
        "test_start": str(test_start),
    }
    pd.Series(metadata).to_json(models_dir / "training_metadata.json", indent=2)

    return {
        "cv_folds": cv_df,
        "cv_summary": rank_df,
        "test_metrics": test_df_metrics,
        "test_metrics_by_asset": test_by_asset_df,
        "test_predictions": pred_df,
    }
