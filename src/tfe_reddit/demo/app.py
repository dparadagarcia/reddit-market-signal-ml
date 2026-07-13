from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = REPORTS_DIR / "metrics_test.csv"
    pred_path = REPORTS_DIR / "predictions_test.csv"
    if not metrics_path.exists() or not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(metrics_path), pd.read_csv(pred_path, parse_dates=["week_start"])


def main() -> None:
    st.set_page_config(page_title="TFM Reddit Market Predictor", layout="wide")
    st.title("Demo — Predicción semanal con Reddit + ML")
    st.caption("Proyecto TFM: BTC, DOGE y SPY | Validación temporal")

    metrics_df, pred_df = load_data()
    if metrics_df.empty or pred_df.empty:
        st.warning(
            "No hay resultados todavía. Ejecuta primero los scripts de dataset y entrenamiento para generar métricas."
        )
        st.stop()

    models = sorted(pred_df["model"].unique().tolist())
    assets = sorted(pred_df["asset"].unique().tolist())

    selected_model = st.sidebar.selectbox("Modelo", options=models)
    selected_asset = st.sidebar.selectbox("Activo", options=assets)

    mrow = metrics_df[metrics_df["model"] == selected_model]
    if not mrow.empty:
        row = mrow.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balanced Accuracy", f"{row['balanced_accuracy']:.3f}")
        c2.metric("F1", f"{row['f1']:.3f}")
        c3.metric("Precision", f"{row['precision']:.3f}")
        c4.metric("Recall", f"{row['recall']:.3f}")

    filt = pred_df[(pred_df["model"] == selected_model) & (pred_df["asset"] == selected_asset)].copy()
    filt = filt.sort_values("week_start")

    st.subheader(f"Evolución temporal en test — {selected_asset}")
    long_df = filt.melt(
        id_vars=["week_start"],
        value_vars=["target_up", "y_pred"],
        var_name="serie",
        value_name="valor",
    )
    fig = px.line(long_df, x="week_start", y="valor", color="serie", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matriz de confusión")
    if not filt.empty:
        cm = confusion_matrix(filt["target_up"], filt["y_pred"], labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

        last = filt.iloc[-1]
        pred_txt = "SUBE" if int(last["y_pred"]) == 1 else "BAJA"
        st.info(
            f"Última semana disponible ({last['week_start'].date()}): predicción del modelo **{selected_model}** = **{pred_txt}**"
        )

    st.subheader("Detalle de predicciones")
    st.dataframe(filt.tail(40), use_container_width=True)


if __name__ == "__main__":
    main()
