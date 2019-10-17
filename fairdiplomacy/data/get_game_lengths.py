import argparse
import json
import glob
import joblib
import os
from typing import List

import diplomacy


def get_game_length(game_json) -> int:
    with open(game_json, "r") as f:
        j = json.load(f)

    return len([p for p in j["phases"] if p["results"]])


def get_all_game_lengths(paths, n_jobs=-1) -> List[int]:
    return joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_game_length)(p) for p in paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("games_dir", help="Dir containing game.json files")
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.games_dir, "game*.json"))
    lengths = get_all_game_lengths(paths)

    for path, length in zip(paths, lengths):
        print(path, length)
