import json
import logging
from math import ceil
from collections import Counter
from typing import List, Tuple, Dict

from fairdiplomacy.agents.base_search_agent import num_orderable_units
from fairdiplomacy.agents.threaded_search_agent import ThreadedSearchAgent
from fairdiplomacy import pydipcc


class BRSearchAgent(ThreadedSearchAgent):
    """One-ply search with dipnet-policy rollouts

    ## Policy
    1. Consider a set of orders that are suggested by the dipnet policy network.
    2. For each set of orders, perform a number of rollouts using the dipnet
    policy network for each power.
    3. Score each order set by the average supply center count at the end
    of the rollout.
    4. Choose the order set with the highest score.
    """

    def __init__(
        self,
        *,
        rollouts_per_plausible_order,
        n_plausible_orders,
        max_actions_units_ratio=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollouts_per_plausible_order = rollouts_per_plausible_order
        self.n_plausible_orders = n_plausible_orders
        self.max_actions_units_ratio = (
            max_actions_units_ratio if max_actions_units_ratio > 0 else 1e6
        )

    def get_orders(self, game, power) -> List[str]:
        if type(game) != pydipcc.Game:
            game = pydipcc.Game.from_json(json.dumps(game.to_saved_game_format()))

        n_units = num_orderable_units(game.get_state(), power)
        plausible_orders = list(
            self.get_plausible_orders(
                game,
                limit=min(self.n_plausible_orders, ceil(n_units * self.max_actions_units_ratio)),
            )[power].keys()
        )
        logging.info("Plausible orders: {}".format(plausible_orders))

        if len(plausible_orders) == 0:
            return []
        if len(plausible_orders) == 1:
            return list(plausible_orders.pop())

        n_chunks, chunk_size = self.get_chunk_size(
            len(plausible_orders), self.rollouts_per_plausible_order
        )
        results = []
        for _ in range(n_chunks):
            r = self.do_rollouts(
                game, [{power: orders} for orders in plausible_orders], chunk_size
            )
            results.extend(r)

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

    @classmethod
    def get_chunk_size(cls, n_actions, n_rollouts):
        MAX = 1024  # make config param? ideally we just handle this in do_rollouts, really...
        n_chunks, chunk_size = 1, n_rollouts
        while chunk_size * n_actions > MAX and chunk_size % 2 == 0:
            n_chunks *= 2
            chunk_size //= 2
        return n_chunks, chunk_size
