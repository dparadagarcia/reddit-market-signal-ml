from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from tfe_reddit.config import load_yaml_config, resolve_path
from tfe_reddit.data.reddit_ingestion import fetch_reddit_posts
from tfe_reddit.utils.io import save_dataframe


def main() -> None:
    base_cfg = load_yaml_config("configs/base.yaml")
    out_path = resolve_path(base_cfg["paths"]["raw_reddit"])

    reddit_df = fetch_reddit_posts(base_cfg)
    save_dataframe(reddit_df, out_path)
    print(f"[OK] Reddit guardado en: {out_path} | filas={len(reddit_df)}")


if __name__ == "__main__":
    main()
