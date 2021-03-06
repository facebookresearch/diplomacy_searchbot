#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import json
import re
import glob
import os
from collections import Counter

import tabulate

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import average_game_scores, compute_game_scores, GameScores


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
    - power_won: 'AUSTRIA' or 'FRANCE' or ...
    """
    power_one = get_power_one(game_json_path)

    try:
        with open(game_json_path) as f:
            j = json.load(f)
    except Exception as e:
        print(e)
        return None

    rl_rewards = compute_game_scores(POWERS.index(power_one), j)

    counts = {k: len(v) for k, v in j["phases"][-1]["state"]["centers"].items()}
    for p in POWERS:
        if p not in counts:
            counts[p] = 0
    powers_won = {p for p, v in counts.items() if v == max(counts.values())}
    power_won = power_one if power_one in powers_won else powers_won.pop()

    if counts[power_one] == 0:
        return "six", power_one, power_won, rl_rewards

    winner_count, winner = max([(c, p) for p, c in counts.items()])
    if winner_count < 18:
        return "draw", power_one, power_won, rl_rewards

    if winner == power_one:
        return "one", power_one, power_won, rl_rewards
    else:
        return "six", power_one, power_won, rl_rewards


def print_rl_stats(results, print_stderr=False):
    stats_per_power = collections.defaultdict(list)
    for _, real_power, _, rl_stats in results:
        stats_per_power[real_power].append(rl_stats)
        stats_per_power["_TOTAL"].append(rl_stats)
    stats_per_power = {
        power: average_game_scores(stats) for power, stats in stats_per_power.items()
    }

    cols = list(GameScores._fields)

    table = [["-"] + cols]
    for power, (avgs, stderrs) in sorted(stats_per_power.items()):
        stats = stderrs if print_stderr else avgs
        table.append([power[:3]] + [getattr(stats, col) for col in cols])
    print(tabulate.tabulate(table, headers="firstrow"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", help="Directories containing game.json files")
    parser.add_argument("--stderr", action="store_true", default=False)
    args = parser.parse_args()

    paths = [
        x
        for results_dir in args.results_dirs
        for x in glob.glob(os.path.join(results_dir, "game*.json"))
    ]

    results = [get_result(path) for path in paths]
    results = [r for r in results if r is not None]
    print_rl_stats(results, print_stderr=args.stderr)
