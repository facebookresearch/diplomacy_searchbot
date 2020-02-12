#!/usr/bin/env python
import argparse
import json
import re
import glob
import os

from fairdiplomacy.models.consts import POWERS


def get_power_one(game_json_path):
    """Returns:
    - power: one of {"AUSTRIA", "FRANCE", ... }
    """
    name = re.findall("game.*\.json", game_json_path)[0]
    for power in POWERS:
        if power[:3] in name:
            return power

    raise ValueError(f"Couldn't parse power name from {name}")


def get_result(game_json_path):
    """Read a game.json and return the winner

    Returns:
    - winner: 'one' or 'six' or 'draw'
    - power_one: 'AUSTRIA' or 'FRANCE' or ...
    """
    power_one = get_power_one(game_json_path)

    with open(game_json_path) as f:
        j = json.load(f)

    counts = {k: len(v) for k, v in j["phases"][-1]["state"]["centers"].items()}

    if counts[power_one] == 0:
        return "six", power_one

    winner_count, winner = max([(c, p) for p, c in counts.items()])
    if winner_count < 18:
        return "draw", power_one

    if winner == power_one:
        return "one", power_one
    else:
        return "six", power_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing game.json files")
    args = parser.parse_args()

    results_dir = ""
    for d in [args.results_dir, "/fairdiplomacy", "/compare_agents"]:
        results_dir += d
        paths = glob.glob(os.path.join(results_dir, "game*.json"))
        if len(paths) > 0:
            break

    results = [get_result(path) for path in paths]

    # print each result
    from pprint import pprint

    pprint(results)
    print()

    # print args
    try:
        for x in ["A", "B"]:
            with open(os.path.join(results_dir, f"AGENT_{x}.arg")) as f:
                print("ARG", x, f.read().strip())
        print()
    except FileNotFoundError:
        pass

    # print win percentages
    from collections import Counter

    counts = Counter([w for w, _ in results])
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}, {v/len(results)}")
