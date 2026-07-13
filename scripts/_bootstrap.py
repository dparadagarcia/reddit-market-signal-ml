from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root() -> None:
    """Permite ejecutar scripts directamente desde la raíz del proyecto."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
