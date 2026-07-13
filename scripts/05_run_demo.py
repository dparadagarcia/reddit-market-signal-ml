from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    app_path = root / "src" / "tfe_reddit" / "demo" / "app.py"
    cmd = ["streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
