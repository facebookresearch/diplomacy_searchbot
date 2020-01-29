import argparse
import json
import re
import glob
import os


def parse_name(game_json_path):
    """Returns:
    - game_type: one of {'1v6', '6v1'}
    - power: one of {"AUSTRIA", "FRANCE", ... }
    """
    game_type, power = re.findall("game\.(.v.)\.([A-Z]+)\.", game_json_path)[0]
    return game_type, power


def get_result(game_json_path):
    """Read a game.json and return the winner

    Returns:
    - winner: 'A' or 'B' or 'DRAW'
    - game_type: '1v6' or '6v1'
    - power_one: 'AUSTRIA' or 'FRANCE' or ...
    """
    game_type, power_one = parse_name(game_json_path)

    with open(game_json_path) as f:
        j = json.load(f)

    counts = [(len(v), k) for k, v in j["phases"][-1]["state"]["centers"].items()]
    count, winner = max(counts)

    if count < 18:
        return "DRAW", game_type, power_one

    if winner == power_one:
        return "A" if game_type == "1v6" else "B", game_type, power_one
    else:
        return "A" if game_type == "6v1" else "B", game_type, power_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing game.json files")
    args = parser.parse_args()

    results_dir = ""
    for d in [args.results_dir, "/fairdiplomacy", "/compare_agents"]:
        results_dir += d
        paths = glob.glob(os.path.join(results_dir, "game.*.json"))
        if len(paths ) > 0:
            break

    results = [get_result(path) for path in paths]

    # print each result
    from pprint import pprint
    pprint(results)
    print()

    # print args
    for x in ['A', 'B']:
        with open(os.path.join(results_dir, f'AGENT_{x}.arg')) as f:
            print('ARG', x, f.read().strip())
    print()

    # print win percentages
    from collections import Counter
    counts = Counter([w for w, _, _ in results])
    for k, v in sorted(counts.items()):
        print(f'{k}: {v}, {v/len(results)}')

