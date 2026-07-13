from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    """Resuelve rutas relativas respecto a la raíz del proyecto."""
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML de configuración."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"La configuración en {path} no es un diccionario YAML válido")
    return cfg


def get_asset_names(base_cfg: dict[str, Any]) -> list[str]:
    return list(base_cfg.get("assets", {}).keys())
