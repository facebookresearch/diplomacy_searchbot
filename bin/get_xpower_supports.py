#!/usr/bin/env python
import argparse
import collections
import json
import re
import glob
import os
from collections import Counter
import pydipcc
import fairdiplomacy.game
import pydipcc
import tabulate
import pprint
import joblib
import torch

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import average_game_scores, compute_game_scores, GameScores


def make_safe(fn):
    def foo(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(e)
            return None

    return foo


def print_rl_stats(results, args):
    stats_per_power = collections.defaultdict(list)
    for _, real_power, _, rl_stats in results:
        stats_per_power[real_power].append(rl_stats)
        stats_per_power["_TOTAL"].append(rl_stats)
    stats_per_power = {
        power: average_game_scores(stats) for power, stats in stats_per_power.items()
    }

    cols = list(GameScores._fields)

    table = [["-"] + cols]
    for power, stats in sorted(stats_per_power.items()):
        table.append([power[:3]] + [getattr(stats, col) for col in cols])
    print(tabulate.tabulate(table, headers="firstrow"))


def compute_xpower_supports(path, max_year=None):
    with open(path) as f:
        try:
            j = json.load(f)
        except json.JSONDecodeError as e:
            raise type(e)(f"Error loading {path}: \n {e.message}")

    game = fairdiplomacy.game.Game.from_saved_game_format(j)
    # game = pydipcc.Game.from_json(game.to_dict())

    num_supports, num_xpower, num_eff = 0, 0, 0
    for phase in game.get_phase_history():
        state = phase.state
        if state["name"][1:-1] == max_year:
            break
        loc_power = {
            unit.split()[1]: power for power, units in state["units"].items() for unit in units
        }
        for power, power_orders in phase.orders.items():
            for order in power_orders:
                order_tokens = order.split()
                is_support = (
                    len(order_tokens) >= 5
                    and order_tokens[2] == "S"
                    and order_tokens[3] in ("A", "F")
                )
                if not is_support:
                    continue
                num_supports += 1
                src = order_tokens[4]
                if src not in loc_power:
                    raise RuntimeError(f"{order}: {src} not in {loc_power}")
                if loc_power[src] == power:
                    continue
                num_xpower += 1
                # print(order)

                cf_states = []
                for do_support in (False, True):
                    g_cf = fairdiplomacy.game.Game.clone_from(game, up_to_phase=state["name"])
                    # g_cf.clear_orders()
                    # print(g_cf.orders, power, order)
                    assert g_cf.get_state()["name"] == state["name"]
                    if not do_support:
                        hold_order = " ".join(order_tokens[:2] + ["H"])
                        g_cf.set_orders(power, [hold_order])
                    # for pp, ppo in phase.orders.items():
                    #     if not do_support and pp == power:
                    #         assert order in ppo
                    #         ppo = [o for o in ppo if o != order]
                    #     print(pp, ppo)
                    #     g_cf.set_orders(pp, ppo)
                    # print(order, do_support, hold_order, g_cf.get_orders()[power])
                    g_cf.process()
                    assert g_cf.get_state()["name"] != state["name"]
                    s = g_cf.get_state()
                    cf_states.append((s["name"], s["units"], s["retreats"]))

                # pprint.pprint(cf_states)
                # for s in cf_states:
                #     del s["zobrist_hash"]
                #     del s["timestamp"]

                if cf_states[0] != cf_states[1]:
                    num_eff += 1

    return {"name": os.path.basename(path), "s": num_supports, "x": num_xpower, "e": num_eff}


def compute_xpower_statistics(paths, max_year=None, num_jobs=40):

    stats = joblib.Parallel(num_jobs)(
        joblib.delayed(compute_xpower_supports)(path, max_year=max_year) for path in paths
    )

    print(
        tabulate.tabulate(
            [(s["name"], s["s"], s["x"], s["e"]) for s in stats[:10]],
            headers=("name", "supports", "xpower", "effective"),
        )
    )
    print("...\n")

    x_support_ratio = sum([s["x"] / s["s"] for s in stats if s["s"] > 0]) / len(
        [s for s in stats if s["s"] > 0]
    )
    eff_x_support_ratio = sum([s["e"] / s["x"] for s in stats if s["x"] > 0]) / len(
        [s for s in stats if s["x"] > 0]
    )

    print(
        f"{len(paths)} games; x_support= {x_support_ratio:.4f}  eff_x_support= {eff_x_support_ratio:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_dir", help="Directory containing game.json files")
    parser.add_argument(
        "--max-games", type=int, default=1000000, help="Max # of games to evaluate"
    )
    parser.add_argument(
        "--max-year", help="Stop computing at this year (to avoid endless supports for draw",
    )
    parser.add_argument(
        "--metadata-path", help="Path to metadata file for games, allowing for filtering"
    )
    parser.add_argument(
        "--metadata-filter", help="Lambda function to filter games based on metadata"
    )

    args = parser.parse_args()

    if args.metadata_path:
        with open(args.metadata_path) as mf:
            metadata = json.load(mf)
            if args.metadata_filter is not None:
                filter_lambda = eval(args.metadata_filter)
                game_ids = [k for k, g in metadata.items() if filter_lambda(g)]
                print(f"Selected {len(game_ids)} / {len(metadata)} games from metadata file.")
            else:
                game_ids = metadata.keys()

            metadata_paths = [f"{args.game_dir}/game_{game_id}.json" for game_id in game_ids]
            paths = [p for p in metadata_paths if os.path.exists(p)]
            print(f"{len(paths)} / {len(metadata_paths)} from metadata exist.")
    else:
        # just use all the paths
        paths = glob.glob(f"{args.game_dir}/game*.json")
        assert len(paths) > 0

    # reduce the number of games if necessary
    if len(paths) > args.max_games:
        print(f"Sampling {args.max_games} from dataset of size {len(paths)}")
        paths = [paths[i] for i in torch.randperm(len(paths))[: args.max_games]]

    compute_xpower_statistics(paths, max_year=args.max_year)
