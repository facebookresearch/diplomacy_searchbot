import logging
from collections import Counter
from typing import List, Tuple, Dict

from fairdiplomacy.agents.base_search_agent import BaseSearchAgent


class BRSearchAgent(BaseSearchAgent):
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
        model_path="/checkpoint/jsgray/diplomacy/dipnet.pth",
        n_rollout_procs=70,
        n_server_procs=1,
        n_gpu=1,
        max_batch_size=1000,
        rollouts_per_plausible_order=10,
        max_rollout_length=20,
    ):
        super().__init__(
            model_path=model_path,
            n_rollout_procs=n_rollout_procs,
            n_server_procs=n_server_procs,
            n_gpu=n_gpu,
            max_batch_size=max_batch_size,
        )

        self.rollouts_per_plausible_order = rollouts_per_plausible_order
        self.max_rollout_length = max_rollout_length

    def get_orders(self, game, power) -> List[str]:
        plausible_orders = self.get_plausible_orders(game, power)
        logging.info("Plausible orders: {}".format(plausible_orders))

        if len(plausible_orders) == 1:
            return list(plausible_orders.pop())

        results = self.distribute_rollouts(
            game,
            [{power: orders} for orders in plausible_orders],
            self.rollouts_per_plausible_order,
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
