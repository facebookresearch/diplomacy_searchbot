#!/usr/bin/env python
import argparse
import collections
import json
import re
import glob
import os
import warnings
from collections import Counter, defaultdict

import tabulate

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import average_game_scores, compute_game_scores, GameScores


def avg(L):
    return sum(L) / len(L)


def print_situation_stats(paths0, paths1):
    data = defaultdict(dict)
    cfr_iters = set()
    results = defaultdict(dict)
    for repeat, paths in enumerate((paths0, paths1)):
        for path in paths:
            match = re.search(r"game_(\d+)\.log$", path)
            assert match
            game_idx = int(match[1])
            for line in open(path):
                fields = line.split()
                if "<> [" in line:
                    power = fields[9]
                    cfr_iter = int(fields[5])
                    avg_utility = float(fields[13].split("=")[1])
                    row_key = (game_idx, repeat, power, cfr_iter)
                    cfr_iters.add(cfr_iter)
                if "|>" in line:
                    orders = " ".join(fields[8:])
                    data[row_key][orders] = {
                        "prob": float(fields[4]),
                        "bp_prob": float(fields[5]),
                        "avg_u": float(fields[6]),
                        "regret": float(fields[6]) - avg_utility,
                        "state_u": avg_utility,
                    }
                if "Result" in line:
                    result = fields[4]
                    test = " ".join(fields[6:])
                    results[test][(game_idx, repeat)] = 1 if result == "PASSED" else 0

    # print(data)
    M = max(cfr_iters)
    N = len(paths) // 2
    for power in POWERS:
        # print(f"{'='*20} {power:8} {'='*20}")
        # 1. calculate how different the plausible orders are
        num_orders, num_common_orders, common_prob, common_bp = [], [], [], []
        for game_i in range(N):
            for game_j in range(N):
                data_i = data[(game_i, 0, power, M)]
                data_j = data[(game_j, 0, power, M)]
                common_orders = set(data_i.keys()) & set(data_j.keys())
                # print(len(data_i), len(common_orders))
                if len(data_i) != len(data_j):
                    assert len(data_i) == 0 or len(data_j) == 0
                    continue

                assert len(data_i) == len(data_j)
                num_orders.append(len(data_i))
                num_common_orders.append(len(common_orders))
                common_prob.append(sum(data_i[o]["prob"] for o in common_orders))
                common_bp.append(sum(data_i[o]["bp_prob"] for o in common_orders))
        print(
            f"{power:8s} : common_orders= {avg(num_common_orders):6.3f} / {avg(num_orders):6.3f} common_bp= {avg(common_bp):6.3f} common_prob= {avg(common_prob):6.3f}"
        )
    for power in POWERS:
        # 2. calculate how consistent the values are between games, at different iters
        for cfr_iter in sorted(list(cfr_iters)):
            prob_mse, u_mse, r_mse, avg_cr, order_prob_mse, state_u = [], [], [], [], [], []
            for game in range(N):
                data_i = data[(game, 0, power, cfr_iter)]
                data_j = data[(game, 1, power, cfr_iter)]
                if len(data_i) != len(data_j):
                    assert len(data_i) == 0 or len(data_j) == 0
                    continue
                this_prob_mse, this_u_mse, this_r_mse, this_cr, this_order_prob_mse = 0, 0, 0, 0, 0
                order_probs_i, order_probs_j = defaultdict(float), defaultdict(float)
                num_orders = 1
                for action in data_i.keys():
                    action_i = data_i[action]
                    if action in data_j:
                        action_j = data_j[action]
                    else:
                        warnings.warn(f"Did not find {action} in game {game}")
                        action_j = action_i
                    orders = eval(action)
                    num_orders = len(orders)
                    prob_i, u_i, r_i = action_i["prob"], action_i["avg_u"], action_i["regret"]
                    prob_j, u_j, r_j = action_j["prob"], action_j["avg_u"], action_j["regret"]
                    for order in orders:
                        order_probs_i[order] += prob_i
                        order_probs_j[order] += prob_j

                    this_prob_mse += (prob_i - prob_j) ** 2
                    this_u_mse += (prob_i + prob_j) / 2 * (u_i - u_j) ** 2
                    this_r_mse += (prob_i + prob_j) / 2 * (r_i - r_j) ** 2
                    this_cr += ((prob_j - prob_i) * u_j + (prob_i - prob_j) * u_i) / 2

                for order in order_probs_i:
                    this_order_prob_mse += (
                        order_probs_i[order] - order_probs_j[order]
                    ) ** 2 / num_orders

                # print(
                #     f"{power:8s} {game} {cfr_iter:4d}: prob_rmse= {this_prob_mse**0.5:6.3f}  u_rmse= {this_u_mse**0.5:6.3f}  r_rmse=  {this_r_mse**0.5:6.3f}"
                # )

                prob_mse.append(this_prob_mse)
                u_mse.append(this_u_mse)
                r_mse.append(this_r_mse)
                avg_cr.append(this_cr)
                order_prob_mse.append(this_order_prob_mse)
                state_u.append(list(data_i.values())[0]["state_u"] if len(data_i) else 0)

            print(
                f"{power:8s}  {cfr_iter:4d}: prob_rmse= {avg(prob_mse)**0.5:6.3f}  op_rmse= {avg(order_prob_mse)**0.5:6.3f}  u_rmse= {avg(u_mse)**0.5:6.3f}  r_rmse=  {avg(r_mse)**0.5:7.4f}  avg_cr=  {avg(avg_cr):7.4f}"
            )

    for test, r in results.items():
        # print(list(r.keys()))
        print(
            f"{test:40s} avg_pass= {avg(r.values())}  ( {sum(r.values())} / {len(r)} )"  # "  intra_consistency={avg([(r[(game_idx, 1)] == r[(game_idx, 2)]) for game_idx in range(N)])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir0", help="Directory containing game.json files")
    parser.add_argument("results_dir1", help="Directory containing game.json files")

    args = parser.parse_args()

    assert os.path.exists(args.results_dir0)
    assert os.path.exists(args.results_dir1)

    paths0 = glob.glob(os.path.join(args.results_dir0, "game*.log"))
    paths1 = glob.glob(os.path.join(args.results_dir1, "game*.log"))

    print_situation_stats(paths0, paths1)
