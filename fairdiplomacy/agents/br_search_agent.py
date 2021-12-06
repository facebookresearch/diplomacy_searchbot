# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from math import ceil
from collections import Counter
from typing import List, Tuple, Dict

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import BaseSearchAgent, num_orderable_units
from fairdiplomacy.agents.model_rollouts import ModelRollouts
from fairdiplomacy.agents.model_wrapper import ModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler


class BRSearchAgent(BaseSearchAgent):
    """One-ply search with model rollouts

    ## Policy
    1. Consider a set of orders that are suggested by the policy network.
    2. For each set of orders, perform a number of rollouts using the
    policy network for each power.
    3. Score each order set by the average supply center count at the end
    of the rollout.
    4. Choose the order set with the highest score.
    """

    def __init__(self, cfg: agents_cfgs.BRSearchAgent):
        super().__init__(cfg)
        self.model = ModelWrapper(
            cfg.model_path, cfg.device, cfg.value_model_path, cfg.max_batch_size
        )
        self.order_sampler = PlausibleOrderSampler(cfg.plausible_orders_cfg, model=self.model)
        self.model_rollouts = ModelRollouts(self.model, cfg.rollouts_cfg)

    def get_orders(self, game, power) -> List[str]:
        if type(game) != pydipcc.Game:
            game = pydipcc.Game.from_json(json.dumps(game.to_saved_game_format()))

        plausible_orders = list(self.order_sampler.sample_orders(game).get(power, {}).keys())
        logging.info("Plausible orders: {}".format(plausible_orders))

        if len(plausible_orders) == 0:
            return []
        if len(plausible_orders) == 1:
            return list(plausible_orders.pop())

        results = self.model_rollouts.do_rollouts(
            game, [{power: orders} for orders in plausible_orders]
        )

        return self.best_order_from_results(results, power)

    @classmethod
    def best_order_from_results(cls, results: List[Tuple[Dict, Dict]], power) -> List[str]:
        """Given a set of rollout results, choose the move to play

        Arguments:
        - results: List[Tuple[set_orders_dict, all_scores]], where
            -> set_orders_dict: Dict[power, orders] on first turn
            -> all_scores: Dict[power, supply count], e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        - power: the power making the orders, e.g. "ITALY"

        Returns:
        - the orders with the highest average score for power
        """
        order_scores = Counter()
        order_counts = Counter()

        for set_orders_dict, all_scores in results:
            orders = set_orders_dict[power]
            order_scores[orders] += all_scores[power]
            order_counts[orders] += 1

        order_avg_score = {
            orders: order_scores[orders] / order_counts[orders] for orders in order_scores
        }
        logging.info("order_avg_score: {}".format(order_avg_score))
        return list(max(order_avg_score.items(), key=lambda kv: kv[1])[0])
