import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

from fairdiplomacy.agents.base_search_agent import BaseSearchAgent
from fairdiplomacy.models.consts import POWERS


Action = Tuple[str]  # a set of orders
Power = str


class CFR1PAgent(BaseSearchAgent):
    """One-ply cfr with dipnet-policy rollouts"""

    def __init__(
        self,
        *,
        model_path="/checkpoint/jsgray/diplomacy/dipnet.pth",
        n_rollout_procs=70,
        n_server_procs=1,
        n_gpu=1,
        max_batch_size=1000,
        n_rollouts=100,
        max_rollout_length=3,
        use_predicted_final_scores=True,
        n_plausible_orders=8,
    ):
        super().__init__(
            model_path=model_path,
            n_rollout_procs=n_rollout_procs,
            n_server_procs=n_server_procs,
            n_gpu=n_gpu,
            max_batch_size=max_batch_size,
            use_predicted_final_scores=use_predicted_final_scores,
        )

        self.n_rollouts = n_rollouts
        self.max_rollout_length = max_rollout_length

    def get_orders(self, game, power) -> List[str]:

        # CFR data structures
        self.sigma: Dict[Tuple[Power, Action], float] = {}
        self.cum_sigma: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.cum_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)

        power_plausible_orders = {
            p: sorted(self.get_plausible_orders(game, p, limit=self.n_plausible_orders))
            for p in POWERS
        }
        logging.info(f"power_plausible_orders: {power_plausible_orders}")

        if len(power_plausible_orders[power]) == 1:
            return list(list(power_plausible_orders[power]).pop())

        for _ in range(self.n_rollouts):
            # get policy probs for all powers
            power_action_ps: Dict[Power, List[float]] = {
                pwr: self.strategy(pwr, actions)
                for (pwr, actions) in power_plausible_orders.items()
            }

            # sample policy for all powers
            idxs = {
                pwr: np.random.choice(range(len(action_ps)), p=action_ps)
                for pwr, action_ps in power_action_ps.items()
            }
            power_sampled_orders: Dict[Power, Tuple[Action, float]] = {
                pwr: (power_plausible_orders[pwr][idxs[pwr]], action_ps[idxs[pwr]])
                for pwr, action_ps in power_action_ps.items()
            }
            logging.info(f"power_sampled_orders: {power_sampled_orders}")

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = [
                {**{p: a for p, (a, _) in power_sampled_orders.items()}, pwr: action}
                for pwr, actions in power_plausible_orders.items()
                for action in actions
            ]
            all_rollout_results = self.distribute_rollouts(game, set_orders_dicts, N=1)

            for pwr, actions in power_plausible_orders.items():
                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )

                action_utilities: List[float] = [r[1][pwr] for r in results]
                state_utility = np.dot(power_action_ps[pwr], action_utilities)
                action_regrets = [(u - state_utility) for u in action_utilities]

                if pwr == power:
                    old_avg_strategy = self.avg_strategy(power, actions)

                # update cfr data structures
                for action, regret, s in zip(actions, action_regrets, power_action_ps[pwr]):
                    self.cum_regrets[(pwr, action)] += regret
                    self.cum_sigma[(pwr, action)] += s

                if pwr == power:
                    new_avg_strategy = self.avg_strategy(power, actions)
                    logging.debug(
                        "old_avg_strat= {} new_avg_strat= {} mse= {}".format(
                            old_avg_strategy,
                            new_avg_strategy,
                            sum((a - b) ** 2 for a, b in zip(old_avg_strategy, new_avg_strategy)),
                        )
                    )

                pos_regrets = [max(0, self.cum_regrets[(pwr, a)]) for a in actions]
                sum_pos_regrets = sum(pos_regrets)
                for action, pos_regret in zip(actions, pos_regrets):
                    if sum_pos_regrets == 0:
                        self.sigma[(pwr, action)] = 1.0 / len(actions)
                    else:
                        self.sigma[(pwr, action)] = pos_regret / sum_pos_regrets

        # return best order: sample from average policy
        ps = self.avg_strategy(power, power_plausible_orders[power])
        idx = np.random.choice(range(len(ps)), p=ps)
        return list(power_plausible_orders[power][idx])

    def strategy(self, power, actions) -> List[float]:
        try:
            return [self.sigma[(power, a)] for a in actions]
        except KeyError:
            return [1.0 / len(actions) for _ in actions]

    def avg_strategy(self, power, actions) -> List[float]:
        sigmas = [self.cum_sigma[(power, a)] for a in actions]
        sum_sigmas = sum(sigmas)
        if sum_sigmas == 0:
            return [1 / len(actions) for _ in actions]
        else:
            return [s / sum_sigmas for s in sigmas]


if __name__ == "__main__":
    import diplomacy

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    print(CFR1PAgent().get_orders(diplomacy.Game(), "ITALY"))
