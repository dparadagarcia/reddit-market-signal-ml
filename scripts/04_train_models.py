from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.modeling.train import train_and_evaluate
from tfe_reddit.utils.io import read_dataframe


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")
    models_cfg = load_yaml_config("configs/models.yaml").get("models", {})

    ds_path = resolve_path(base_cfg["paths"]["weekly_dataset"])
    models_dir = resolve_path(base_cfg["paths"]["models_dir"])
    reports_dir = resolve_path(base_cfg["paths"]["reports_dir"])

    dataset = read_dataframe(ds_path)
    results = train_and_evaluate(
        dataset=dataset,
        base_cfg=base_cfg,
        models_cfg=models_cfg,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    print("[OK] Entrenamiento y evaluación finalizados")
    for key, df in results.items():
        print(f" - {key}: {len(df)} filas")


if __name__ == "__main__":
    main()
