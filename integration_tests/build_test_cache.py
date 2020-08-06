"""Builds default test cache."""
import pathlib

import integration_tests.heyhi_utils


BUILD_DB_CONF = pathlib.Path(__file__).parent / "conf" / "build_db_cache.prototxt"
GAMES_ROOT = pathlib.Path(__file__).parent / "data" / "selfplay_games"
GAMES_CACHE_ROOT = pathlib.Path(__file__).parent / "data" / "selfplay.cache"


def main():
    overrides = [f"glob={GAMES_ROOT.absolute()}/*.json", f"out_path={GAMES_CACHE_ROOT.absolute()}"]
    print(GAMES_ROOT)
    integration_tests.heyhi_utils.run_config(cfg=BUILD_DB_CONF, overrides=overrides)


if __name__ == "__main__":
    main()
