# reddit-market-signal-ml

Sistema reproducible de análisis y modelado para estudiar si el texto publicado en Reddit aporta señal útil en la predicción semanal de BTC, DOGE y SPY.

El proyecto separa ingesta, construcción de variables, validación temporal, entrenamiento, evaluación y generación de artefactos para memoria o revisión técnica.

## Estructura

- `src/tfe_reddit/`: código fuente del proyecto.
- `scripts/`: ejecución por fases del pipeline.
- `configs/`: configuración de datos, validación y modelos.
- `tests/`: pruebas unitarias básicas.
- `data/`: estructura esperada para datos `raw`, `interim`, `processed` y `external`.

## Requisitos

- Python 3.10, 3.11 o 3.12
- instalación editable recomendada:

```bash
python -m pip install -e ".[dev]"
```

Extras opcionales:

```bash
python -m pip install -e .[boosting,interpretability]
python -m pip install -e .[nlp]
```

## Variables de entorno

Para la ingesta directa desde Reddit mediante PRAW:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Puedes usar un fichero `.env` en la raíz del proyecto. Se incluye un ejemplo en `.env.example`.

## Ejecución

Los scripts pueden ejecutarse directamente desde la raíz del proyecto.

```bash
python scripts/01_fetch_reddit.py
python scripts/02_fetch_market.py
python scripts/03_build_weekly_dataset.py
python scripts/04_train_models.py
python scripts/05_run_demo.py
python scripts/06_generate_report_assets.py
python scripts/07_train_models_by_asset.py
python scripts/08_robustness_analysis.py
python scripts/09_finbert_comparison.py
```

## Flujo recomendado

1. Descargar Reddit y mercado.
2. Construir el dataset semanal.
3. Entrenar modelos con validación temporal.
4. Generar métricas, figuras y análisis complementarios.

## Criterios metodológicos

- validación temporal estricta;
- comparación con referencias sencillas;
- separación entre entrenamiento, validación y test final;
- configuración versionada mediante YAML;
- resultados exportables a CSV y figuras reproducibles.

## Estado del repositorio

Este repositorio contiene el software. La memoria académica y los materiales LaTeX del TFM se mantienen aparte.
