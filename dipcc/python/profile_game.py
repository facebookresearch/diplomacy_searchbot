import os
import argparse
import json
import time
from pprint import pprint

import pydipcc
import fairdiplomacy.game

parser = argparse.ArgumentParser()
parser.add_argument("game_json")
args = parser.parse_args()

with open(args.game_json) as f:
    json_s = f.read()
j = json.loads(json_s)


cc_game = pydipcc.Game()
py_game = fairdiplomacy.game.Game()


times = {"cc": 0, "py": 0}
n = 0

for phase_data in j["phases"]:
    phase = phase_data["name"]
    print(phase)

    assert cc_game.phase == py_game.phase, f"{cc_game.phase} != {py_game.phase}"
    assert cc_game.get_state()["name"] == phase, cc_game.get_state()["name"]

    if "orders" not in phase_data or phase == "COMPLETED":
        break

    all_orders = phase_data["orders"]
    pprint(phase)
    pprint(all_orders)

    for game, t in [(cc_game, "cc"), (py_game, "py")]:
        for power, orders in all_orders.items():
            if orders:
                game.set_orders(power, orders)

        t_start = time.time()
        game.process()
        times[t] += time.time() - t_start

    n += 1

print("\n## Process")
print(f"cc: {times['cc']/n*1e3:.3f}ms/phase")
print(f"py: {times['py']/n*1e3:.3f}ms/phase")
print(f"py / cc = {times['py']/times['cc']:.0f}")


# profile from_json

n = 50
times = {"cc": 0, "py": 0}
for _ in range(n):
    t = time.time()
    pydipcc.Game.from_json(json_s)
    times["cc"] += time.time() - t

    t = time.time()
    fairdiplomacy.game.Game.from_saved_game_format(j)
    times["py"] += time.time() - t

print("\n## from_json")
print(f"cc: {times['cc']/n*1e3:.0f}ms/from_json")
print(f"py: {times['py']/n*1e3:.0f}ms/from_json")
print(f"py / cc = {times['py']/times['cc']:.0f}")
