#!/usr/bin/env python
import argparse
import json
import re
import glob
import os
from collections import Counter

import tabulate

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.selfplay.exploit import compute_game_scores


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

    with open(game_json_path) as f:
        j = json.load(f)

    rl_rewards = compute_game_scores(POWERS.index(power_one), j)

    counts = {k: len(v) for k, v in j["phases"][-1]["state"]["centers"].items()}
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


def print_power_h2h(results, args):
    table = (
        [["player_one"] + [p[:3] for p in POWERS] + ["games_played"]]
        + [
            [p[:3]]
            + [sum(1 for _, p_, w_, _ in results if p_ == p and w_ == w) for w in POWERS]
            + [sum(1 for _, p_, _, _ in results if p_ == p)]
            for p in POWERS
        ]
        + [""]
        + [
            ["total won"]
            + [sum(1 for _, _, w_, _ in results if w_ == w) for w in POWERS]
            + [len(results)]
        ]
    )
    if args.csv:
        print("POWER H2H")
        print("\n".join([",".join(map(str, cols)) for cols in table]))
    else:
        print("====  POWER H2H  ====")
        print(tabulate.tabulate(table, headers="firstrow"))


def print_win_rates(results, args):
    if args.csv:
        print("WIN RATES")
    else:
        print("====  WIN RATES  ====")

    counts = Counter([w for w, _, _, _ in results])
    for k in ["draw", "one", "six"]:
        v = counts[k]
        if args.csv:
            print(k, v, v / len(results), sep=",")
        else:
            print(f"{k}: {v}, {v/len(results)}")


def print_rl_stats(results, args):
    if args.csv:
        # TODO(akhti): do csv
        return
    print("==== AVERAGE REWARDS ====")
    stats_per_power = {}
    for _, real_power, _, rl_stats in results:
        for power in (real_power, "_TOTAL"):
            if power not in stats_per_power:
                stats_per_power[power] = rl_stats.copy()
                stats_per_power[power]["num_games"] = 1
            else:
                for k, v in rl_stats.items():
                    stats_per_power[power][k] += v
                stats_per_power[power]["num_games"] += 1

    for power in stats_per_power:
        for k in list(stats_per_power[power]):
            if k != "num_games":
                stats_per_power[power][k] /= stats_per_power[power]["num_games"]

    cols = list(stats_per_power[POWERS[0]])

    table = [["-"] + cols]
    for power, stats in sorted(stats_per_power.items()):
        table.append([power[:3]] + [stats[col] for col in cols])
    print(tabulate.tabulate(table, headers="firstrow"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing game.json files")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    results_dir = ""
    for d in [args.results_dir, "/fairdiplomacy", "/compare_agents"]:
        results_dir += d
        paths = glob.glob(os.path.join(results_dir, "game*.json"))
        if len(paths) > 0:
            break

    results = [get_result(path) for path in paths]

    print_power_h2h(results, args)
    print("\n\n")
    print_win_rates(results, args)
    print("\n\n")
    print_rl_stats(results, args)
    print("\n\n")

    if args.csv:
        with open(glob.glob(f"{results_dir}/*/config.prototxt")[0]) as f:
            print("CONFIG")
            print(f.read())
