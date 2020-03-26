import logging
import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Dict

from fairdiplomacy.agents.base_search_agent import BaseSearchAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.timing_ctx import TimingCtx


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
        n_rollouts=100,
        max_rollout_length=10,
        use_predicted_final_scores=True,
        n_plausible_orders=8,
        rollout_temperature=0.05,
        use_optimistic_cfr=True,
        enable_compute_nash_conv=False,
        plausible_orders_req_size=1000,
        postman_sync_batches=False,
        max_batch_size=700,
        cache_rollout_results=False,
    ):
        super().__init__(
            model_path=model_path,
            n_rollout_procs=n_rollout_procs,
            n_server_procs=n_server_procs,
            n_gpu=n_gpu,
            max_batch_size=(
                n_plausible_orders * len(POWERS) if postman_sync_batches else max_batch_size
            ),
            use_predicted_final_scores=use_predicted_final_scores,
            rollout_temperature=rollout_temperature,
            postman_wait_till_full=postman_sync_batches,
        )

        self.n_rollouts = n_rollouts
        self.max_rollout_length = max_rollout_length
        self.n_plausible_orders = n_plausible_orders
        self.use_optimistic_cfr = use_optimistic_cfr
        self.enable_compute_nash_conv = enable_compute_nash_conv
        self.plausible_orders_req_size = plausible_orders_req_size
        self.postman_sync_batches = postman_sync_batches
        self.cache_rollout_results = cache_rollout_results

    def get_orders(self, game, power) -> List[str]:
        timings = TimingCtx()

        # CFR data structures
        self.sigma: Dict[Tuple[Power, Action], float] = {}
        self.cum_sigma: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.cum_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.last_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)

        if self.cache_rollout_results:
            rollout_results_cache = RolloutResultsCache()

        if self.postman_sync_batches:
            self.client.set_batch_size(torch.LongTensor([self.plausible_orders_req_size]))

        power_plausible_orders = self.get_plausible_orders(
            game,
            limit=self.n_plausible_orders,
            n=self.plausible_orders_req_size,
            batch_size=self.plausible_orders_req_size,
        )
        power_plausible_orders = {p: sorted(v) for p, v in power_plausible_orders.items()}
        logging.info(f"power_plausible_orders: {power_plausible_orders}")

        if self.postman_sync_batches:
            self.client.set_batch_size(
                torch.LongTensor([sum(map(len, power_plausible_orders.values()))])
            )

        if len(power_plausible_orders[power]) == 1:
            return list(list(power_plausible_orders[power]).pop())

        for cfr_iter in range(self.n_rollouts):
            discount_factor = (cfr_iter + 0.000001) / (cfr_iter + 1)

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
            logging.info(f"power_sampled_orders: {power_sampled_orders}")

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = [
                {**{p: a for p, (a, _) in power_sampled_orders.items()}, pwr: action}
                for pwr, actions in power_plausible_orders.items()
                for action in actions
            ]

            # run rollouts or get from cache
            def on_miss():
                with timings("distribute_rollouts"):
                    return self.distribute_rollouts(game, set_orders_dicts, N=1)

            all_rollout_results = (
                rollout_results_cache.get(set_orders_dicts, on_miss)
                if self.cache_rollout_results
                else on_miss()
            )

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

                # update cfr data structures
                for action, regret, s in zip(actions, action_regrets, power_action_ps[pwr]):
                    self.cum_regrets[(pwr, action)] += regret
                    self.last_regrets[(pwr, action)] = regret
                    self.cum_sigma[(pwr, action)] += s

                if self.use_optimistic_cfr:
                    pos_regrets = [
                        max(0, self.cum_regrets[(pwr, a)] + self.last_regrets[(pwr, a)])
                        for a in actions
                    ]
                else:
                    pos_regrets = [max(0, self.cum_regrets[(pwr, a)]) for a in actions]

                sum_pos_regrets = sum(pos_regrets)
                for action, pos_regret in zip(actions, pos_regrets):
                    if sum_pos_regrets == 0:
                        self.sigma[(pwr, action)] = 1.0 / len(actions)
                    else:
                        self.sigma[(pwr, action)] = pos_regret / sum_pos_regrets

            if self.enable_compute_nash_conv and cfr_iter in [25, 50, 100, 200, 400]:
                logging.info(f"Computing nash conv for iter {cfr_iter}")
                self.compute_nash_conv(cfr_iter, game, power_plausible_orders)

            logging.info(
                f"Timing[cfr_iter]: {str(timings)}, len(set_orders_dicts)={len(set_orders_dicts)}"
            )
            timings.clear()

            if self.cache_rollout_results and (rollout_i + 1) % 10 == 0:
                logging.info(f"{rollout_results_cache}")

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

    def compute_nash_conv(self, cfr_iter, game, power_plausible_orders):
        """For each power, compute EV of each action assuming opponent ave policies"""

        # get policy probs for all powers
        power_action_ps: Dict[Power, List[float]] = {
            pwr: self.avg_strategy(pwr, actions)
            for (pwr, actions) in power_plausible_orders.items()
        }
        logging.info("Policies: {}".format(power_action_ps))

        total_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        temp_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        total_state_utility: Dict[Power, float] = defaultdict(float)
        max_state_utility: Dict[Power, float] = defaultdict(float)
        for pwr, actions in power_plausible_orders.items():
            total_state_utility[pwr] = 0
            max_state_utility[pwr] = 0
        # total_state_utility = [0 for u in idxs]
        nash_conv = 0
        for _ in range(100):
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

                for r in results:
                    action = r[0][pwr]
                    val = r[1][pwr]
                    temp_action_utilities[(pwr, action)] = val
                    total_action_utilities[(pwr, action)] += val
                # logging.info("results for power={}".format(pwr))
                # for i in range(len(power_plausible_orders[pwr])):
                #     action = power_plausible_orders[pwr][i]
                #     util = action_utilities[i]
                #     logging.info("{} {} = {}".format(pwr,action,util))

                # for action in power_plausible_orders[pwr]:
                #     logging.info("{} {} = {}".format(pwr,action,action_utilities))
                # logging.info("action utilities={}".format(action_utilities))
                # logging.info("Results={}".format(results))
                # state_utility = np.dot(power_action_ps[pwr], action_utilities)
                # action_regrets = [(u - state_utility) for u in action_utilities]
                # logging.info("Action utilities={}".format(temp_action_utilities))
                # for action in actions:
                #     total_action_utilities[(pwr,action)] += temp_action_utilities[(pwr,action)]
                # logging.info("Total action utilities={}".format(total_action_utilities))
                # total_state_utility[pwr] += state_utility
        # total_state_utility[:] = [x / 100 for x in total_state_utility]
        for pwr, actions in power_plausible_orders.items():
            # ps = self.avg_strategy(pwr, power_plausible_orders[pwr])
            for i in range(len(actions)):
                action = actions[i]
                total_action_utilities[(pwr, action)] /= 100.0
                if total_action_utilities[(pwr, action)] > max_state_utility[pwr]:
                    max_state_utility[pwr] = total_action_utilities[(pwr, action)]
                total_state_utility[pwr] += (
                    total_action_utilities[(pwr, action)] * power_action_ps[pwr][i]
                )

        for pwr, actions in power_plausible_orders.items():
            logging.info(
                "results for power={} value={} diff={}".format(
                    pwr,
                    total_state_utility[pwr],
                    (max_state_utility[pwr] - total_state_utility[pwr]),
                )
            )
            nash_conv += max_state_utility[pwr] - total_state_utility[pwr]
            for i in range(len(actions)):
                action = actions[i]
                logging.info(
                    "{} {} = {} (prob {})".format(
                        pwr, action, total_action_utilities[(pwr, action)], power_action_ps[pwr][i]
                    )
                )

        logging.info(f"Nash conv for iter {cfr_iter} = {nash_conv}")


class RolloutResultsCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, set_orders_dicts, onmiss_fn):
        key = frozenset(frozenset(d.items()) for d in set_orders_dicts)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            r = onmiss_fn()
            self.cache[key] = r
            return r

    def __repr__(self):
        return "RolloutResultsCache[{} / {} = {:.3f}]".format(
            self.hits, self.hits + self.misses, self.hits / (self.hits + self.misses)
        )


if __name__ == "__main__":
    import diplomacy

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    agent = CFR1PAgent(n_rollouts=500, postman_sync_batches=True)
    print(agent.get_orders(diplomacy.Game(), "ITALY"))
