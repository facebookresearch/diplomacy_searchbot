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
        rollout_temperature=0.05,
    ):
        super().__init__(
            model_path=model_path,
            n_rollout_procs=n_rollout_procs,
            n_server_procs=n_server_procs,
            n_gpu=n_gpu,
            max_batch_size=max_batch_size,
            use_predicted_final_scores=use_predicted_final_scores,
            rollout_temperature=rollout_temperature,
        )

        self.n_rollouts = n_rollouts
        self.max_rollout_length = max_rollout_length
        self.n_plausible_orders = n_plausible_orders

    def get_orders(self, game, power) -> List[str]:

        # CFR data structures
        self.sigma: Dict[Tuple[Power, Action], float] = {}
        self.cum_sigma: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.cum_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.last_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)

        # TODO: parallelize these calls
        power_plausible_orders = self.get_plausible_orders(game, limit=self.n_plausible_orders)
        power_plausible_orders = {p: sorted(v) for p, v in power_plausible_orders.items()}
        logging.info(f"power_plausible_orders: {power_plausible_orders}")

        if len(power_plausible_orders[power]) == 1:
            return list(list(power_plausible_orders[power]).pop())

        iter = 0.000001
        for _ in range(self.n_rollouts):
            iter += 1.0
            discount_factor = (iter - 1.0) / iter

            for pwr, actions in power_plausible_orders.items():
                if len(actions) == 0:
                    continue
                for action in actions:
                    self.cum_regrets[(pwr, action)] *= discount_factor
                    self.cum_sigma[(pwr, action)] *= discount_factor

            # get policy probs for all powers
            power_action_ps: Dict[Power, List[float]] = {
                pwr: self.strategy(pwr, actions)
                for (pwr, actions) in power_plausible_orders.items()
            }

            # sample policy for all powers
            idxs = {
                pwr: np.random.choice(range(len(action_ps)), p=action_ps)
                for pwr, action_ps in power_action_ps.items()
                if len(action_ps) > 0
            }
            power_sampled_orders: Dict[Power, Tuple[Action, float]] = {
                pwr: (
                    (power_plausible_orders[pwr][idxs[pwr]], action_ps[idxs[pwr]])
                    if pwr in idxs
                    else ((), 1.0)
                )
                for pwr, action_ps in power_action_ps.items()
            }
            # logging.info(f"power_sampled_orders: {power_sampled_orders}")

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = [
                {**{p: a for p, (a, _) in power_sampled_orders.items()}, pwr: action}
                for pwr, actions in power_plausible_orders.items()
                for action in actions
            ]
            all_rollout_results = self.distribute_rollouts(game, set_orders_dicts, N=1)

            for pwr, actions in power_plausible_orders.items():
                if len(actions) == 0:
                    continue

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
                    self.last_regrets[(pwr, action)] = regret
                    self.cum_sigma[(pwr, action)] += s

                # pos_regrets = [max(0, self.cum_regrets[(pwr, a)]) for a in actions] # Normal CFR
                pos_regrets = [max(0, self.cum_regrets[(pwr, a)] + self.last_regrets[(pwr, a)]) for a in actions] # Optimistic CFR
                sum_pos_regrets = sum(pos_regrets)
                for action, pos_regret in zip(actions, pos_regrets):
                    if sum_pos_regrets == 0:
                        self.sigma[(pwr, action)] = 1.0 / len(actions)
                    else:
                        self.sigma[(pwr, action)] = pos_regret / sum_pos_regrets

                if pwr == power:
                    new_avg_strategy = self.avg_strategy(power, actions)
                    logging.debug(
                        "old_avg_strat= {} new_avg_strat= {} mse= {}".format(
                            old_avg_strategy,
                            new_avg_strategy,
                            sum((a - b) ** 2 for a, b in zip(old_avg_strategy, new_avg_strategy)),
                        )
                    )



            # if (iter > 25 and iter < 25.5) or (iter > 50 and iter < 50.5) or (iter > 100 and iter < 100.5) or (iter > 200 and iter < 200.5) or (iter > 400 and iter < 400.5):
            #     # Compute NashConv. Specifically, for each power, compute EV of each action assuming opponent ave policies
            #     # get policy probs for all powers
            #     power_action_ps: Dict[Power, List[float]] = {
            #         pwr: self.avg_strategy(pwr, actions)
            #         for (pwr, actions) in power_plausible_orders.items()
            #     }

            #     logging.info(
            #         "EV computation on iter {} power_sampled_orders: {}".format(
            #             iter,
            #             power_sampled_orders,
            #         )
            #     )
            #     logging.info("Policies: {}".format(power_action_ps))

            #     total_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
            #     temp_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
            #     total_state_utility: Dict[Power, float] = defaultdict(float)
            #     max_state_utility: Dict[Power, float] = defaultdict(float)
            #     for pwr, actions in power_plausible_orders.items():
            #         total_action_utilities[(pwr,action)] = 0
            #         total_state_utility[pwr] = 0
            #         max_state_utility[pwr] = 0
            #     # total_state_utility = [0 for u in idxs]
            #     nash_conv = 0
            #     for _ in range(100):
            #         # sample policy for all powers
            #         idxs = {
            #             pwr: np.random.choice(range(len(action_ps)), p=action_ps)
            #             for pwr, action_ps in power_action_ps.items()
            #             if len(action_ps) > 0
            #         }
            #         power_sampled_orders: Dict[Power, Tuple[Action, float]] = {
            #             pwr: (
            #                 (power_plausible_orders[pwr][idxs[pwr]], action_ps[idxs[pwr]])
            #                 if pwr in idxs
            #                 else ((), 1.0)
            #             )
            #             for pwr, action_ps in power_action_ps.items()
            #         }

            #         # for each power: compare all actions against sampled opponent action
            #         set_orders_dicts = [
            #             {**{p: a for p, (a, _) in power_sampled_orders.items()}, pwr: action}
            #             for pwr, actions in power_plausible_orders.items()
            #             for action in actions
            #         ]
            #         all_rollout_results = self.distribute_rollouts(game, set_orders_dicts, N=1)

            #         for pwr, actions in power_plausible_orders.items():
            #             if len(actions) == 0:
            #                 continue

            #             # pop this power's results
            #             results, all_rollout_results = (
            #                 all_rollout_results[: len(actions)],
            #                 all_rollout_results[len(actions) :],
            #             )

            #             for r in results:
            #                 action = r[0][pwr]
            #                 val = r[1][pwr]
            #                 temp_action_utilities[(pwr,action)] = val
            #                 total_action_utilities[(pwr,action)] += val
            #             # logging.info("results for power={}".format(pwr))
            #             # for i in range(len(power_plausible_orders[pwr])):
            #             #     action = power_plausible_orders[pwr][i]
            #             #     util = action_utilities[i]
            #             #     logging.info("{} {} = {}".format(pwr,action,util))

            #             # for action in power_plausible_orders[pwr]:
            #             #     logging.info("{} {} = {}".format(pwr,action,action_utilities))
            #             # logging.info("action utilities={}".format(action_utilities))
            #             #logging.info("Results={}".format(results))
            #             #state_utility = np.dot(power_action_ps[pwr], action_utilities)
            #             # action_regrets = [(u - state_utility) for u in action_utilities]
            #             # logging.info("Action utilities={}".format(temp_action_utilities))
            #             # for action in actions:
            #             #     total_action_utilities[(pwr,action)] += temp_action_utilities[(pwr,action)]
            #             # logging.info("Total action utilities={}".format(total_action_utilities))
            #                 # total_state_utility[pwr] += state_utility
            #     # total_state_utility[:] = [x / 100 for x in total_state_utility]
            #     for pwr, actions in power_plausible_orders.items():
            #         #ps = self.avg_strategy(pwr, power_plausible_orders[pwr])
            #         for i in range(len(actions)):
            #             action = actions[i]
            #             total_action_utilities[(pwr,action)] /= 100.0
            #             if total_action_utilities[(pwr,action)] > max_state_utility[pwr]:
            #                 max_state_utility[pwr] = total_action_utilities[(pwr,action)]
            #             total_state_utility[pwr] += total_action_utilities[(pwr,action)] * power_action_ps[pwr][i]

            #     for pwr, actions in power_plausible_orders.items():
            #         logging.info(
            #             "results for power={} value={} diff={}".format(
            #                 pwr,
            #                 total_state_utility[pwr],
            #                 (max_state_utility[pwr] - total_state_utility[pwr])
            #             )
            #         )
            #         nash_conv += max_state_utility[pwr] - total_state_utility[pwr]
            #         for i in range(len(actions)):
            #             action = actions[i]
            #             logging.info(
            #                 "{} {} = {} (prob {})".format(
            #                     pwr,
            #                     action,
            #                     total_action_utilities[(pwr,action)],
            #                     power_action_ps[pwr][i],
            #                 )
            #             )
            #     logging.info(f"Nash Convergence on iter {iter} = {nash_conv}")
            #     # logging.info(
            #     #     "total_state_utility= {} total_action_utilities= {}".format(
            #     #         total_state_utility,
            #     #         total_action_utilities,
            #     #     )
            #     # )


                        
        logging.info(
            "cum_strats= {}".format(
                self.cum_sigma,
            )
        )
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
