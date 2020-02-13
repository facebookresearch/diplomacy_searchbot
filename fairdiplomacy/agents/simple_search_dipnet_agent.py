import logging
from collections import Counter
from functools import partial
from typing import List

import torch
from diplomacy.utils.export import to_saved_game_format

from fairdiplomacy.agents.base_search_agent import BaseSearchAgent


class SimpleSearchDipnetAgent(BaseSearchAgent):
    """One-ply search with dipnet-policy rollouts

    ## Policy
    1. Consider a set of orders that are suggested by the dipnet policy network.
    2. For each set of orders, perform a number of rollouts using the dipnet
    policy network for each power.
    3. Score each order set by the average supply center count at the end
    of the rollout.
    4. Choose the order set with the highest score.

    ## Implementation Details
    - __init__ forks some number of server processes running a ModelServer instance
      listening on different ports, and a ProcesssPoolExecutor for rollouts
    - get_orders first gets plausible orders to search through, then launches
      rollouts for each plausible order via the proc pool
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

        game_json = to_saved_game_format(game)

        # divide up the rollouts among the processes
        procs_per_order = max(1, self.n_rollout_procs // len(plausible_orders))
        logging.info(
            f"num_plausible_orders={len(plausible_orders)} , procs_per_order={procs_per_order}"
        )
        batch_sizes = [
            len(x)
            for x in torch.arange(self.rollouts_per_plausible_order).chunk(procs_per_order)
            if len(x) > 0
        ]
        logging.info(f"procs_per_order={procs_per_order} , batch_sizes={batch_sizes}")
        results = self.proc_pool.map(
            call,
            [
                partial(
                    self.do_rollout,
                    game_json=game_json,
                    set_orders_dict={power: orders},
                    hostport=self.hostports[i % self.n_server_procs],
                    max_rollout_length=self.max_rollout_length,
                    batch_size=batch_size,
                )
                for orders in plausible_orders
                for i, batch_size in enumerate(batch_sizes)
            ],
        )
        results = [
            (order_dict[power], scores) for result in results for (order_dict, scores) in result
        ]

        return self.best_order_from_results(results, power)

    @classmethod
    def best_order_from_results(cls, results, power) -> List[str]:
        """Given a set of rollout results, choose the move to play

        Arguments:
        - results: Tuple[orders, all_scores], where
            -> orders: a complete set of orders, e.g. ("A ROM H", "F NAP - ION", "A VEN H")
            -> all_scores: Dict[power, supply count], e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        - power: the power making the orders, e.g. "ITALY"

        Returns:
        - the "orders" with the highest average score
        """
        order_scores = Counter()
        order_counts = Counter()

        for orders, all_scores in results:
            order_scores[orders] += all_scores[power]
            order_counts[orders] += 1

        order_avg_score = {
            orders: order_scores[orders] / order_counts[orders] for orders in order_scores
        }
        logging.info("order_avg_score: {}".format(order_avg_score))
        return list(max(order_avg_score.items(), key=lambda kv: kv[1])[0])


def call(f):
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()
