from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from tfe_reddit.evaluation.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REDDIT_PATH = PROJECT_ROOT / "data" / "raw" / "reddit_posts.parquet"
WEEKLY_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_dataset.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "predictions_test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def _compute_metrics_by_asset(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.drop_duplicates(subset=["week_start", "asset", "model"]).copy()
    rows = []
    for (model_name, asset), group in pred_df.groupby(["model", "asset"], sort=True):
        metrics = compute_classification_metrics(
            y_true=group["target_up"],
            y_pred=group["y_pred"],
            y_score=group["y_score"] if group["y_score"].notna().any() else None,
        )
        metrics["model"] = model_name
        metrics["asset"] = asset
        metrics["n_obs"] = int(len(group))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["model", "asset"]).reset_index(drop=True)


def _compute_coverage_summary(reddit_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    reddit_summary = (
        reddit_df.groupby("asset", as_index=False)
        .agg(
            reddit_posts=("source_id", "count"),
            reddit_start=("created_utc", "min"),
            reddit_end=("created_utc", "max"),
        )
    )
    weekly_summary = (
        weekly_df.groupby("asset", as_index=False)
        .agg(
            weekly_rows=("week_start", "count"),
            weekly_start=("week_start", "min"),
            weekly_end=("week_start", "max"),
            target_up_share=("target_up", "mean"),
        )
    )
    return reddit_summary.merge(weekly_summary, on="asset", how="outer").sort_values("asset")


def _plot_weekly_reddit_volume(reddit_df: pd.DataFrame, output_path: Path) -> None:
    weekly = reddit_df.copy()
    weekly["week_start"] = (
        pd.to_datetime(weekly["created_utc"], utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .dt.start_time
    )
    plot_df = (
        weekly.groupby(["week_start", "asset"])
        .size()
        .rename("posts")
        .reset_index()
        .sort_values(["week_start", "asset"])
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for asset in ["BTC", "DOGE", "SPY"]:
        asset_df = plot_df[plot_df["asset"] == asset]
        if asset_df.empty:
            continue
        ax.plot(asset_df["week_start"], asset_df["posts"], label=asset, linewidth=1.6)

    ax.set_title("Cobertura semanal de publicaciones de Reddit por activo")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Numero de publicaciones")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_inspection_snapshot(metrics_df: pd.DataFrame, pred_df: pd.DataFrame, output_path: Path) -> None:
    best = metrics_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    model_name = str(best["model"])
    asset = "DOGE" if "DOGE" in set(pred_df["asset"]) else str(pred_df["asset"].iloc[0])
    filt = pred_df[(pred_df["model"] == model_name) & (pred_df["asset"] == asset)].copy()
    filt = filt.sort_values("week_start")

    plt.style.use("default")
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 2.2], hspace=0.65, wspace=0.5)
    fig.suptitle("Herramienta de inspección de resultados", fontsize=15, fontweight="bold", y=0.98)

    metrics = [
        ("Balanced accuracy", best["balanced_accuracy"]),
        ("F1", best["f1"]),
        ("Precision", best["precision"]),
        ("Recall", best["recall"]),
    ]
    for idx, (label, value) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        ax.axis("off")
        ax.text(0.5, 0.68, label, ha="center", va="center", fontsize=10, color="#555555")
        ax.text(0.5, 0.25, f"{value:.3f}", ha="center", va="center", fontsize=18, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)

    ax_line = fig.add_subplot(gs[1, :])
    if not filt.empty:
        ax_line.plot(filt["week_start"], filt["target_up"], marker="o", label="real", linewidth=1.5)
        ax_line.plot(filt["week_start"], filt["y_pred"], marker="s", label="predicción", linewidth=1.5)
    ax_line.set_title(f"Evolución temporal en test - {asset} / {model_name}", fontsize=11)
    ax_line.set_ylim(-0.1, 1.1)
    ax_line.set_ylabel("Clase")
    ax_line.grid(alpha=0.25)
    ax_line.legend(frameon=False, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_test_model_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df.sort_values("balanced_accuracy", ascending=True).copy()
    labels = plot_df["model"].str.replace("_", " ", regex=False)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y_pos = range(len(plot_df))
    ax.barh(y_pos, plot_df["balanced_accuracy"], color="#1f77b4", alpha=0.85, label="Balanced accuracy")
    ax.scatter(plot_df["f1"], y_pos, color="#ff7f0e", s=52, label="F1", zorder=3)
    ax.scatter(plot_df["roc_auc"], y_pos, color="#2ca02c", s=52, label="ROC-AUC", zorder=3)
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.25, 0.72)
    ax.set_xlabel("Valor de la métrica")
    ax.set_title("Comparación de modelos en el bloque final de test")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_asset_balanced_accuracy(metrics_by_asset_df: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        metrics_by_asset_df.pivot(index="model", columns="asset", values="balanced_accuracy")
        .loc[["naive_persistence", "market_logreg", "text_logreg", "hybrid_logreg", "linear_svm", "random_forest"]]
    )
    labels = [idx.replace("_", " ") for idx in pivot.index]
    x = range(len(labels))
    width = 0.24
    colors = {"BTC": "#1f77b4", "DOGE": "#ff7f0e", "SPY": "#2ca02c"}

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for offset, asset in zip([-width, 0, width], ["BTC", "DOGE", "SPY"], strict=True):
        ax.bar([pos + offset for pos in x], pivot[asset], width=width, label=asset, color=colors[asset], alpha=0.85)

    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.65)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.25, 0.72)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Rendimiento por activo y familia de modelo")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_robustness_summary(robustness_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = robustness_df.copy()
    plot_df["variant_label"] = (
        plot_df["variant"]
        .str.replace("_", " ", regex=False)
        .str.replace("sin zona neutral", "sin zona neutral", regex=False)
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = range(len(plot_df))
    ax.bar(x, plot_df["best_balanced_accuracy"], color="#1f77b4", alpha=0.82, label="Mejor modelo")
    ax.plot(x, plot_df["naive_balanced_accuracy"], color="#d62728", marker="o", linewidth=2, label="Persistencia")
    for idx, value in enumerate(plot_df["delta_vs_naive"]):
        ax.text(idx, plot_df["best_balanced_accuracy"].iloc[idx] + 0.01, f"+{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["variant_label"], rotation=20, ha="right")
    ax.set_ylim(0.35, 0.68)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Sensibilidad del resultado ante cambios de configuración")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_finbert_comparison(finbert_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["balanced_accuracy", "f1", "roc_auc"]
    labels = ["Balanced accuracy", "F1", "ROC-AUC"]
    x = range(len(metrics))
    width = 0.34

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, (_, row) in zip([-width / 2, width / 2], finbert_df.iterrows(), strict=True):
        ax.bar([pos + offset for pos in x], [row[m] for m in metrics], width=width, label=row["experiment"], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 0.68)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación complementaria de sentimiento: VADER frente a VADER+FinBERT")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _plot_dataset_pipeline(output_path: Path) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.36, 0.16, 0.28, "#dbeafe", "Datos brutos", "Reddit + mercado"),
        (0.28, 0.36, 0.16, 0.28, "#ede9fe", "Limpieza", "fechas, texto y\nvalores nulos"),
        (0.51, 0.36, 0.16, 0.28, "#dcfce7", "Agregación semanal", "alineación por activo\ny week_start"),
        (0.74, 0.36, 0.16, 0.28, "#fef3c7", "Variables", "mercado, actividad,\nsentimiento"),
    ]

    for x, y, w, h, color, title, subtitle in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#475569", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=9, color="#334155")

    final_rect = plt.Rectangle((0.82, 0.08), 0.13, 0.16, facecolor="#fee2e2", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(final_rect)
    ax.text(0.885, 0.18, "Dataset final", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.885, 0.11, "clasificación semanal", ha="center", va="center", fontsize=8.8, color="#334155")

    arrow_y = 0.50
    for x1, x2 in [(0.21, 0.28), (0.44, 0.51), (0.67, 0.74)]:
        ax.annotate("", xy=(x2, arrow_y), xytext=(x1, arrow_y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))
    ax.annotate("", xy=(0.885, 0.24), xytext=(0.82, 0.36), arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

    ax.text(0.5, 0.84, "Flujo de construcción del dataset analítico", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        0.76,
        "El pipeline transforma publicaciones y series de mercado en observaciones semanales comparables para el modelado.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_validation_timeline(weekly_df: pd.DataFrame, output_path: Path) -> None:
    cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    val_cfg = cfg["validation"]

    weeks = sorted(pd.to_datetime(weekly_df["week_start"], utc=True).dropna().unique())
    test_weeks = int(val_cfg["test_weeks"])
    train_min_weeks = int(val_cfg["train_min_weeks"])
    val_weeks = int(val_cfg["val_weeks"])
    step_weeks = int(val_cfg["step_weeks"])

    test_start = weeks[-test_weeks]
    train_val_weeks = [w for w in weeks if w < test_start]

    folds = []
    end = train_min_weeks
    while end + val_weeks <= len(train_val_weeks):
        train_window = train_val_weeks[:end]
        val_window = train_val_weeks[end : end + val_weeks]
        folds.append((train_window, val_window))
        end += step_weeks

    last_train, last_val = folds[-1]
    test_window = [w for w in weeks if w >= test_start]

    def _span(a, b):
        return pd.Timestamp(a).tz_localize(None), pd.Timestamp(b).tz_localize(None)

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 4.8))
    colors = {"train": "#93c5fd", "val": "#86efac", "test": "#fca5a5"}

    # Esquema exterior train+val vs test.
    train_val_start = pd.Timestamp(train_val_weeks[0]).tz_localize(None)
    train_val_end = pd.Timestamp(train_val_weeks[-1]).tz_localize(None)
    test_start_naive = pd.Timestamp(test_window[0]).tz_localize(None)
    test_end_naive = pd.Timestamp(test_window[-1]).tz_localize(None)
    zoom_start = pd.Timestamp(last_train[max(0, len(last_train) - 60)]).tz_localize(None)

    # Esquema del último fold interno (solo tramo final visible).
    axes[0].barh(["Último fold de validación"], [(pd.Timestamp(last_train[-1]).tz_localize(None) - zoom_start).days], left=zoom_start, color=colors["train"], edgecolor="#475569")
    axes[0].barh(["Último fold de validación"], [(_span(last_val[0], last_val[-1])[1] - _span(last_val[0], last_val[-1])[0]).days], left=_span(last_val[0], last_val[-1])[0], color=colors["val"], edgecolor="#475569")
    axes[0].barh(["Último fold de validación"], [(_span(test_window[0], test_window[-1])[1] - _span(test_window[0], test_window[-1])[0]).days], left=_span(test_window[0], test_window[-1])[0], color=colors["test"], edgecolor="#475569", alpha=0.55)
    axes[0].set_title("Detalle del último fold interno y bloque final de test", fontsize=11)

    axes[1].barh(["Partición exterior final"], [(train_val_end - train_val_start).days], left=train_val_start, color="#cbd5e1", edgecolor="#475569")
    axes[1].barh(["Partición exterior final"], [(test_end_naive - test_start_naive).days], left=test_start_naive, color=colors["test"], edgecolor="#475569")
    axes[1].set_title(
        f"Configuración real: train mínimo {train_min_weeks} semanas, validación {val_weeks} semanas, paso {step_weeks}, test final {test_weeks}",
        fontsize=11,
    )

    for ax in axes:
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    zoom_end = test_end_naive
    axes[0].set_xlim(zoom_start, zoom_end)
    axes[1].set_xlim(train_val_start, test_end_naive)

    axes[0].text(
        pd.Timestamp(last_val[0]).tz_localize(None),
        0.22,
        "inicio validación",
        fontsize=8.5,
        color="#166534",
    )
    axes[0].text(
        pd.Timestamp(test_window[0]).tz_localize(None),
        0.22,
        "inicio test",
        fontsize=8.5,
        color="#991b1b",
    )

    handles = [
        plt.Line2D([0], [0], color=colors["train"], lw=8, label="Train"),
        plt.Line2D([0], [0], color=colors["val"], lw=8, label="Validación"),
        plt.Line2D([0], [0], color=colors["test"], lw=8, label="Test"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper left", ncol=3)
    fig.suptitle("Protocolo temporal utilizado en el experimento", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_confusion_matrix(metrics_df: pd.DataFrame, output_path: Path) -> None:
    best = metrics_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    matrix = [
        [int(best["tn"]), int(best["fp"])],
        [int(best["fn"]), int(best["tp"])],
    ]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicción bajista", "Predicción alcista"])
    ax.set_yticklabels(["Clase real bajista", "Clase real alcista"])
    ax.set_title(f"Matriz de confusión en test\nMejor modelo agregado: {best['model']}", fontsize=12)

    total = sum(sum(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            pct = value / total * 100 if total else 0
            ax.text(j, i, f"{value}\n({pct:.1f}%)", ha="center", va="center", color="#0f172a", fontsize=11, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reddit_df = pd.read_parquet(RAW_REDDIT_PATH)
    weekly_df = pd.read_parquet(WEEKLY_DATASET_PATH)
    pred_df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["week_start"])
    metrics_df = pd.read_csv(REPORTS_DIR / "metrics_test.csv")
    robustness_df = pd.read_csv(REPORTS_DIR / "robustness_summary.csv")
    finbert_df = pd.read_csv(REPORTS_DIR / "finbert_comparison.csv")
    pred_df = pred_df.drop_duplicates(subset=["week_start", "asset", "model"]).copy()

    metrics_by_asset_df = _compute_metrics_by_asset(pred_df=pred_df)
    coverage_df = _compute_coverage_summary(reddit_df=reddit_df, weekly_df=weekly_df)

    metrics_by_asset_df.to_csv(REPORTS_DIR / "metrics_test_by_asset.csv", index=False)
    coverage_df.to_csv(REPORTS_DIR / "coverage_summary.csv", index=False)
    pred_df.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)
    _plot_weekly_reddit_volume(reddit_df=reddit_df, output_path=FIGURES_DIR / "cobertura_reddit_semanal.png")
    _plot_inspection_snapshot(
        metrics_df=metrics_df,
        pred_df=pred_df,
        output_path=FIGURES_DIR / "pantallazo_herramienta_inspeccion.png",
    )
    _plot_test_model_metrics(metrics_df=metrics_df, output_path=FIGURES_DIR / "resultados_modelos_test.png")
    _plot_asset_balanced_accuracy(
        metrics_by_asset_df=metrics_by_asset_df,
        output_path=FIGURES_DIR / "resultados_por_activo_balanced_accuracy.png",
    )
    _plot_robustness_summary(
        robustness_df=robustness_df,
        output_path=FIGURES_DIR / "sensibilidad_configuracion.png",
    )
    _plot_finbert_comparison(
        finbert_df=finbert_df,
        output_path=FIGURES_DIR / "comparacion_finbert.png",
    )
    _plot_dataset_pipeline(output_path=FIGURES_DIR / "pipeline_construccion_dataset.png")
    _plot_validation_timeline(weekly_df=weekly_df, output_path=FIGURES_DIR / "protocolo_validacion_temporal.png")
    _plot_confusion_matrix(metrics_df=metrics_df, output_path=FIGURES_DIR / "matriz_confusion_hybrid_logreg.png")

    print(f"[OK] Guardado {REPORTS_DIR / 'metrics_test_by_asset.csv'}")
    print(f"[OK] Guardado {REPORTS_DIR / 'coverage_summary.csv'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'cobertura_reddit_semanal.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'pantallazo_herramienta_inspeccion.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'resultados_modelos_test.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'resultados_por_activo_balanced_accuracy.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'sensibilidad_configuracion.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'comparacion_finbert.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'pipeline_construccion_dataset.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'protocolo_validacion_temporal.png'}")
    print(f"[OK] Guardada {FIGURES_DIR / 'matriz_confusion_hybrid_logreg.png'}")


if __name__ == "__main__":
    main()
