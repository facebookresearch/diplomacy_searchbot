import os
import argparse
import json

from fairdiplomacy import pydipcc

parser = argparse.ArgumentParser()
parser.add_argument("crash_dump")
parser.add_argument("orders_json", nargs="?")
args = parser.parse_args()

with open(args.crash_dump, "r") as f:
    j = json.load(f)
game = pydipcc.Game.from_json(json.dumps(j))

if args.orders_json:
    print("Using orders from file:", args.orders_json)
    with open(args.orders_json, "r") as f:
        all_orders = json.load(f)
elif "staged_orders" in j:
    print("Using staged_orders")
    all_orders = j["staged_orders"]
else:
    # use last orders
    print("Using last phase's orders")
    last_phase = game.get_phase_history()[-1]
    all_orders = last_phase.orders
    game.rollback_to_phase(last_phase.name)

for power, orders in all_orders.items():
    game.set_orders(power, orders)

game.process()

print("SUCCESS!")
print(game.phase)
