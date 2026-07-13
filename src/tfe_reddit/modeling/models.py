from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


def build_preprocessor(
    numeric_columns: list[str],
    features_cfg: dict[str, Any],
    include_text: bool,
    include_numeric: bool,
) -> ColumnTransformer:
    transformers = []

    if include_text:
        ngram = tuple(features_cfg.get("tfidf_ngram_range", [1, 2]))
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=int(features_cfg.get("tfidf_max_features", 8000)),
                    min_df=int(features_cfg.get("tfidf_min_df", 2)),
                    ngram_range=ngram,
                ),
                "weekly_text",
            )
        )

    if include_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_columns))

    if not transformers:
        raise ValueError("Debe activarse al menos un tipo de variable (texto o numérica)")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(model_name: str, model_cfg: dict[str, Any]):
    if model_name == "market_logreg":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )

    if model_name in {"text_logreg", "hybrid_logreg"}:
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            class_weight="balanced",
            solver="saga",
            random_state=42,
        )

    if model_name == "linear_svm":
        return LinearSVC(C=float(model_cfg.get("C", 1.0)), class_weight="balanced")

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            max_depth=int(model_cfg.get("max_depth", 8)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 8)),
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost no está instalado. Instala extras boosting o desactiva el modelo en configs/models.yaml"
            ) from exc

        return XGBClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 400)),
            max_depth=int(model_cfg.get("max_depth", 4)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


def build_pipeline(
    model_name: str,
    model_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
    numeric_columns: list[str],
) -> Pipeline:
    include_text = model_name in {"text_logreg", "hybrid_logreg", "linear_svm"}
    include_numeric = model_name in {"market_logreg", "hybrid_logreg", "random_forest", "xgboost"}

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        features_cfg=features_cfg,
        include_text=include_text,
        include_numeric=include_numeric,
    )
    clf = build_estimator(model_name=model_name, model_cfg=model_cfg)
    return Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
