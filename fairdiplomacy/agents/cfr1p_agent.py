import logging
from math import ceil
import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Dict

from fairdiplomacy.agents.base_search_agent import BaseSearchAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.selfplay.metrics import MultiStopWatchTimer
from fairdiplomacy.utils.sampling import sample_p_dict


Action = Tuple[str]  # a set of orders
Power = str


class CFR1PAgent(BaseSearchAgent):
    """One-ply cfr with dipnet-policy rollouts"""

    def __init__(
        self,
        *,
        n_rollouts,
        cache_rollout_results=False,
        enable_compute_nash_conv=False,
        n_plausible_orders,
        postman_sync_batches=False,
        use_optimistic_cfr=True,
        use_final_iter=True,
        use_pruning=False,
        max_batch_size=700,
        average_n_rollouts=1,
        n_rollout_procs,
        max_actions_units_ratio=None,
        plausible_orders_req_size=None,
        bp_iters=0,
        bp_prob=0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            n_rollout_procs=(
                n_plausible_orders * len(POWERS) if postman_sync_batches else n_rollout_procs
            ),
            max_batch_size=(
                n_plausible_orders * len(POWERS) if postman_sync_batches else max_batch_size
            ),
            postman_wait_till_full=postman_sync_batches,
        )

        if postman_sync_batches:
            assert n_rollout_procs >= n_plausible_orders * len(POWERS)

        self.n_rollouts = n_rollouts
        self.cache_rollout_results = cache_rollout_results
        self.enable_compute_nash_conv = enable_compute_nash_conv
        self.n_plausible_orders = n_plausible_orders
        self.postman_sync_batches = postman_sync_batches
        self.use_optimistic_cfr = use_optimistic_cfr
        self.use_final_iter = use_final_iter
        self.use_pruning = use_pruning
        self.plausible_orders_req_size = plausible_orders_req_size or max_batch_size
        self.average_n_rollouts = average_n_rollouts
        self.max_actions_units_ratio = (
            max_actions_units_ratio
            if max_actions_units_ratio is not None and max_actions_units_ratio > 0
            else 1e6
        )
        self.bp_iters = bp_iters
        self.bp_prob = bp_prob

        logging.info(f"Initialized CFR1P Agent: {self.__dict__}")

    def get_orders(self, game, power) -> List[str]:
        prob_distributions = self.get_all_power_prob_distributions(
            game, early_exit_for_power=power
        )
        logging.info(f"Final strategy: {prob_distributions[power]}")
        if len(prob_distributions[power]) == 0:
            return []
        return list(sample_p_dict(prob_distributions[power]))

    def get_all_power_prob_distributions(
        self, game, early_exit_for_power=None
    ) -> Dict[str, Dict[Tuple[str], float]]:
        """Return dict {power: {action: prob}}"""

        # CFR data structures
        self.sigma: Dict[Tuple[Power, Action], float] = {}
        self.cum_sigma: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.cum_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.cum_utility: Dict[Action, float] = defaultdict(float)
        self.last_regrets: Dict[Tuple[Power, Action], float] = defaultdict(float)
        self.bp_sigma: Dict[Tuple[Power, Action], float] = defaultdict(float)

        game_state = game.get_state()
        phase = game_state["name"]

        if self.cache_rollout_results:
            rollout_results_cache = RolloutResultsCache()

        if self.postman_sync_batches:
            self.client.set_batch_size(torch.LongTensor([self.plausible_orders_req_size]))

        # Determine the set of plausible actions to consider for each power
        power_n_units = [num_orderable_units(game_state, p) for p in POWERS]
        power_plausible_orders = self.get_plausible_orders(
            game,
            limit=[
                min(self.n_plausible_orders, ceil(u * self.max_actions_units_ratio))
                for u in power_n_units
            ],
            n=self.plausible_orders_req_size,
            batch_size=self.plausible_orders_req_size,
        )
        for p, orders_to_logprob in power_plausible_orders.items():
            for o, prob in orders_to_logprob.items():
                self.bp_sigma[(p, o)] = float(np.exp(prob))
        power_plausible_orders = {
            p: sorted(list(v.keys())) for p, v in power_plausible_orders.items()
        }

        logging.debug(f"{phase} power_plausible_orders: {power_plausible_orders}")

        if self.postman_sync_batches:
            self.client.set_batch_size(
                torch.LongTensor([sum(map(len, power_plausible_orders.values()))])
            )

        if early_exit_for_power and len(power_plausible_orders[early_exit_for_power]) == 0:
            return {early_exit_for_power: {tuple(): 1.0}}
        if early_exit_for_power and len(power_plausible_orders[early_exit_for_power]) == 1:
            return {
                early_exit_for_power: {
                    tuple(list(power_plausible_orders[early_exit_for_power]).pop()): 1.0
                }
            }

        timing_logger = logging.getLogger("timing")
        timings = MultiStopWatchTimer()
        iter_weight = 0.0
        for cfr_iter in range(self.n_rollouts):
            timings.start("start")

            if self.use_pruning and cfr_iter == 1 + int(self.n_rollouts / 4):
                for pwr, actions in power_plausible_orders.items():
                    paired_list = []
                    for action in actions:
                        ave_regret = self.cum_regrets[(pwr, action)] / iter_weight
                        new_pair = (action, ave_regret)
                        paired_list.append(new_pair)
                    paired_list.sort(key=lambda tup: tup[1])
                    for (action, ave_regret) in paired_list:
                        ave_strat = self.cum_sigma[(pwr, action)] / iter_weight
                        if (
                            ave_regret < -0.06
                            and ave_strat < 0.002
                            and self.sigma[(pwr, action)] == 0
                        ):
                            self.cum_sigma[(pwr, action)] = 0
                            logging.info(
                                "pruning on iter {} action {} with ave regret {} and ave strat {}".format(
                                    cfr_iter, action, ave_regret, ave_strat
                                )
                            )
                            actions.remove(action)

            if self.use_pruning and cfr_iter == 1 + int(self.n_rollouts / 2):
                for pwr, actions in power_plausible_orders.items():
                    paired_list = []
                    for action in actions:
                        ave_regret = self.cum_regrets[(pwr, action)] / iter_weight
                        new_pair = (action, ave_regret)
                        paired_list.append(new_pair)
                    paired_list.sort(key=lambda tup: tup[1])
                    for (action, ave_regret) in paired_list:
                        ave_strat = self.cum_sigma[(pwr, action)] / iter_weight
                        if (
                            ave_regret < -0.03
                            and ave_strat < 0.001
                            and self.sigma[(pwr, action)] == 0
                        ):
                            self.cum_sigma[(pwr, action)] = 0
                            logging.info(
                                "pruning on iter {} action {} with ave regret {} and ave strat {}".format(
                                    cfr_iter, action, ave_regret, ave_strat
                                )
                            )
                            actions.remove(action)

            discount_factor = (cfr_iter + 0.000001) / (cfr_iter + 1)
            iter_weight *= discount_factor
            iter_weight += 1.0

            for pwr, actions in power_plausible_orders.items():
                if len(actions) == 0:
                    continue
                self.cum_utility[pwr] *= discount_factor
                for action in actions:
                    self.cum_regrets[(pwr, action)] *= discount_factor
                    self.cum_sigma[(pwr, action)] *= discount_factor

            timings.start("query_policy")
            # get policy probs for all powers
            power_action_cfr = {
                pwr: self.strategy(pwr, actions)
                for (pwr, actions) in power_plausible_orders.items()
            }
            power_action_ps: Dict[Power, List[float]] = {
                pwr: (
                    self.bp_strategy(pwr, actions)
                    if cfr_iter < self.bp_iters or np.random.rand() < self.bp_prob
                    else self.strategy(pwr, actions)
                )
                for (pwr, actions) in power_plausible_orders.items()
            }
            timings.start("apply_orders")
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
            logging.debug(f"{phase}.{cfr_iter} power_sampled_orders={power_sampled_orders}")

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = [
                {**{p: a for p, (a, _) in power_sampled_orders.items()}, pwr: action}
                for pwr, actions in power_plausible_orders.items()
                for action in actions
            ]

            # run rollouts or get from cache
            def on_miss():
                return self.distribute_rollouts(
                    game, set_orders_dicts, average_n_rollouts=self.average_n_rollouts
                )

            timings.start("distribute_rollouts")
            all_rollout_results = (
                rollout_results_cache.get(set_orders_dicts, on_miss)
                if self.cache_rollout_results
                else on_miss()
            )
            timings.start("update")
            if cfr_iter & (cfr_iter + 1) == 0:  # 2^n-1
                logging.info(f"[{cfr_iter+1}/{self.n_rollouts}] Power sampled orders:")
                for power, (orders, _) in power_sampled_orders.items():
                    logging.info(f"    {power:10s}  {orders}")

            for pwr, actions in power_plausible_orders.items():
                if len(actions) == 0:
                    continue

                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )

                # calculate regrets
                action_utilities: List[float] = [r[1][pwr] for r in results]
                state_utility = np.dot(power_action_ps[pwr], action_utilities)
                action_regrets = [(u - state_utility) for u in action_utilities]

                # log some action values
                # if cfr_iter & (cfr_iter + 1) == 0:  # 2^n-1
                if cfr_iter == self.n_rollouts - 1:
                    logging.info(
                        f"[{cfr_iter+1}/{self.n_rollouts}] {pwr} avg_utility={self.cum_utility[pwr] / iter_weight:.5f} cur_utility={state_utility:.5f}"
                    )
                    logging.info(
                        f"    {'probs':8s}  {'bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders"
                    )
                    action_probs: List[float] = self.avg_strategy(pwr, power_plausible_orders[pwr])
                    bp_probs: List[float] = self.bp_strategy(pwr, power_plausible_orders[pwr])
                    avg_utilities = [
                        (self.cum_regrets[(pwr, a)] + self.cum_utility[pwr]) / iter_weight
                        for a in actions
                    ]
                    sorted_metrics = sorted(
                        zip(actions, action_probs, bp_probs, avg_utilities, action_utilities),
                        key=lambda ac: -ac[1],
                    )
                    for orders, p, bp_p, avg_u, cur_u in sorted_metrics:
                        logging.info(
                            f"    {p:8.5f}  {bp_p:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}"
                        )

                # elif pwr == early_exit_for_power:
                #     u = action_utilities[idxs[pwr]]
                #     logging.info(
                #         f"Sampled action utility={u} exp_utility={state_utility:.5f} regret={u - state_utility:.5f}"
                #     )

                # print(f"iter {cfr_iter} {pwr}")
                # def PP(label, L):
                #     print(label, [f"{x:.3f}" for x in L])
                # PP("    utilities", action_utilities)
                # PP("    regrets  ", action_regrets)
                # PP("    cur_strat", power_action_ps[pwr])

                # update cfr data structures
                self.cum_utility[pwr] += state_utility
                for action, regret, s in zip(actions, action_regrets, power_action_cfr[pwr]):
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

            if self.enable_compute_nash_conv and cfr_iter in (
                24,
                49,
                99,
                199,
                399,
                self.n_rollouts - 1,
            ):
                logging.info(f"Computing nash conv for iter {cfr_iter}")
                self.compute_nash_conv(cfr_iter, game, power_plausible_orders)

            if self.cache_rollout_results and (cfr_iter + 1) % 10 == 0:
                logging.info(f"{rollout_results_cache}")
        timing_logger.debug(
            f"Timing[cfr_iter {cfr_iter+1}/{self.n_rollouts}]: {str(timings)}, len(set_orders_dicts)={len(set_orders_dicts)}"
        )

        # return prob. distributions for each power
        ret = {}
        for p in POWERS:
            final_ps = self.strategy(p, power_plausible_orders[p])
            avg_ps = self.avg_strategy(p, power_plausible_orders[p])
            ps = final_ps if self.use_final_iter else avg_ps
            ret[p] = dict(sorted(zip(power_plausible_orders[p], ps), key=lambda ac_p: -ac_p[1]))

            if early_exit_for_power == p:
                avg_ps_dict = dict(
                    sorted(zip(power_plausible_orders[p], avg_ps), key=lambda ac_p: -ac_p[1])
                )
                logging.info(f"Final avg strategy: {avg_ps_dict}")

        return ret

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

    def bp_strategy(self, power, actions) -> List[float]:
        sigmas = [self.bp_sigma[(power, a)] for a in actions]
        sum_sigmas = sum(sigmas)
        assert len(actions) == 0 or sum_sigmas > 0, f"{actions} {self.bp_sigma}"
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
        br_iters = 100
        for _ in range(br_iters):
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
            all_rollout_results = self.distribute_rollouts(
                game, set_orders_dicts, average_n_rollouts=self.average_n_rollouts
            )

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
                total_action_utilities[(pwr, action)] /= br_iters
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


def num_orderable_units(game_state, power):
    if game_state["name"][-1] == "A":
        return abs(game_state["builds"].get(power, {"count": 0})["count"])
    if game_state["name"][-1] == "R":
        return len(game_state["retreats"].get(power, []))
    else:
        return len(game_state["units"].get(power, []))


if __name__ == "__main__":
    from fairdiplomacy.game import Game

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    np.random.seed(0)
    torch.manual_seed(0)

    agent = CFR1PAgent(
        n_rollouts=10,
        max_rollout_length=5,
        model_path="/checkpoint/alerer/fairdiplomacy/sl_fbdata_all/checkpoint.pth.best",
        postman_sync_batches=False,
        rollout_temperature=0.5,
        n_rollout_procs=24 * 7,
        rollout_top_p=0.9,
        mix_square_ratio_scoring=0.1,
        n_plausible_orders=24,
        average_n_rollouts=3,
    )
    print(agent.get_orders(Game(), "AUSTRIA"))
