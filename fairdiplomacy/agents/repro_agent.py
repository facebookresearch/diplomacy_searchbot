# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from fairdiplomacy.agents.base_agent import BaseAgent


class ReproAgent(BaseAgent):
    def __init__(self, game_path):
        with open(game_path, "r") as f:
            self.json = json.load(f)

    def get_orders(self, game, power):
        phase = game.get_state()["name"]
        all_possible_orders_set = {
            x for lst in game.get_all_possible_orders().values() for x in lst
        }
        orders = self.json["order_history"][phase].get(power, [])
        assert set(orders).issubset(all_possible_orders_set), set(orders) - all_possible_orders_set
        return list(orders)
