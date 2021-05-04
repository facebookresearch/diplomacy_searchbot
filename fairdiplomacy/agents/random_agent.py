# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def get_orders(self, game, power):
        possible_orders = {
            loc: orders
            for loc, orders in game.get_all_possible_orders().items()
            if loc in game.get_orderable_locations(power)
        }
        return [random.choice(orders) for orders in possible_orders.values()]
