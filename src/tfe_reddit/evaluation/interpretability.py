from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_model_importance(model_name: str, fitted_pipeline, output_path: Path, top_n: int = 40) -> None:
    """Exporta importancias para modelos lineales/árboles cuando sea posible."""
    if fitted_pipeline is None:
        return

    if not hasattr(fitted_pipeline, "named_steps"):
        return

    if "preprocessor" not in fitted_pipeline.named_steps or "clf" not in fitted_pipeline.named_steps:
        return

    pre = fitted_pipeline.named_steps["preprocessor"]
    clf = fitted_pipeline.named_steps["clf"]

    if not hasattr(pre, "get_feature_names_out"):
        return

    feature_names = pre.get_feature_names_out()
    if hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_.ravel()
    else:
        return

    if len(vals) != len(feature_names):
        return

    imp = pd.DataFrame({"feature": feature_names, "importance": vals})
    imp["abs_importance"] = np.abs(imp["importance"])
    imp = imp.sort_values("abs_importance", ascending=False).head(top_n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(output_path, index=False)
