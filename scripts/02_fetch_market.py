from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.data.market_data import download_market_daily
from tfe_reddit.utils.io import save_dataframe


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")
    out_path = resolve_path(base_cfg["paths"]["raw_market"])

    market_df = download_market_daily(base_cfg)
    save_dataframe(market_df, out_path)
    print(f"[OK] Mercado guardado en: {out_path} | filas={len(market_df)}")


if __name__ == "__main__":
    main()
