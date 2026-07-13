from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.data.weekly_dataset import build_weekly_dataset
from tfe_reddit.utils.io import read_dataframe, save_dataframe


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")

    reddit_path = resolve_path(base_cfg["paths"]["raw_reddit"])
    market_path = resolve_path(base_cfg["paths"]["raw_market"])
    out_path = resolve_path(base_cfg["paths"]["weekly_dataset"])

    reddit_df = read_dataframe(reddit_path)
    market_df = read_dataframe(market_path)

    weekly_df = build_weekly_dataset(reddit_raw_df=reddit_df, market_raw_df=market_df, base_cfg=base_cfg)
    save_dataframe(weekly_df, out_path)
    print(f"[OK] Dataset semanal guardado en: {out_path} | filas={len(weekly_df)}")


if __name__ == "__main__":
    main()
