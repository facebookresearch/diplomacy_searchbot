# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import collections
import copy
import itertools
import json
import logging
import random
import warnings
import tabulate
import time

import numpy as np
import torch

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    make_set_orders_dicts,
    sample_orders_from_policy,
)

# fairdiplomacy.action_generation and fairdiplomacy.action_exploration
# both circularly refer fairdiplomacy.agents, so we import those modules whole
# instead of from "fairdiplomacy.action_generation import blah".
# That way, we break the circular initialization issue by not requiring the symbols
# *within* those modules to exist at import time, since they very well might not exist
# if we were only halfway through importing those when we began importing this file.
import fairdiplomacy.action_generation
import fairdiplomacy.action_exploration
from fairdiplomacy.agents.model_rollouts import ModelRollouts
from fairdiplomacy.agents.model_wrapper import ModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler, renormalize_policy
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.typedefs import (
    Action,
    JointAction,
    PlausibleOrders,
    Power,
    PowerPolicies,
)


ActionDict = Dict[Tuple[Power, Action], float]


class CFRData:
    def __init__(self, bp_policy: PowerPolicies, use_optimistic_cfr: bool):

        self.use_optimistic_cfr = use_optimistic_cfr
        self.sigma: ActionDict = {}
        self.cum_sigma: ActionDict = defaultdict(float)
        self.cum_regrets: ActionDict = defaultdict(float)
        self.cum_utility: Dict[Power, float] = defaultdict(float)
        self.bp_sigma: Optional[ActionDict] = defaultdict(float)
        self.cum_weight = 0

        self.power_plausible_orders: PlausibleOrders = {p: sorted(v) for p, v in bp_policy.items()}

        if all(x <= 0 for y in bp_policy.values() for x in y.values()):
            # this is a dummy policy, only the order keys should be used
            self.bp_sigma = None
        else:
            for p, orders_to_prob in bp_policy.items():
                if len(orders_to_prob) > 0 and abs(sum(orders_to_prob.values()) - 1) > 1e-3:
                    raise RuntimeError(f"Invalid policy for {p}: {orders_to_prob}")

                for o, prob in orders_to_prob.items():
                    self.bp_sigma[(p, o)] = float(prob)

    def strategy(self, pwr) -> List[float]:
        actions = self.power_plausible_orders[pwr]
        try:
            return [self.sigma[(pwr, a)] for a in actions]
        except KeyError:
            return [1.0 / len(actions) for _ in actions]

    def avg_strategy(self, pwr) -> List[float]:
        actions = self.power_plausible_orders[pwr]
        sigmas = [self.cum_sigma[(pwr, a)] for a in actions]
        sum_sigmas = sum(sigmas)
        if sum_sigmas == 0:
            return [1 / len(actions) for _ in actions]
        else:
            return [s / sum_sigmas for s in sigmas]

    def avg_utility(self, pwr):
        return self.cum_utility[pwr] / self.cum_weight

    def avg_action_utility(self, pwr, a):
        return (self.cum_regrets[(pwr, a)] + self.cum_utility[pwr]) / self.cum_weight

    def bp_strategy(self, pwr, temperature=1.0) -> List[float]:
        if self.bp_sigma is None:
            warnings.warn("Tried to access bp_strategy when a dummy bp policy was provided.")
            return [-1.0] * len(self.power_plausible_orders[pwr])
        actions = self.power_plausible_orders[pwr]
        sigmas = [self.bp_sigma[(pwr, a)] ** (1.0 / temperature) for a in actions]
        sum_sigmas = sum(sigmas)
        assert len(actions) == 0 or sum_sigmas > 0, f"{actions} {self.bp_sigma}"
        return [s / sum_sigmas for s in sigmas]

    def discount_linear_cfr(self, cfr_iter):
        discount_factor = (cfr_iter + 0.000001) / (cfr_iter + 1)

        for pwr, actions in self.power_plausible_orders.items():
            if len(actions) == 0:
                continue
            self.cum_utility[pwr] *= discount_factor
            for action in actions:
                self.cum_regrets[(pwr, action)] *= discount_factor
                self.cum_sigma[(pwr, action)] *= discount_factor

        # note: this isn't correct until update is called!
        self.cum_weight = self.cum_weight * discount_factor + 1.0

    def update(self, pwr, actions, state_utility, action_regrets, sigmas):
        for action, regret, sigma in zip(actions, action_regrets, sigmas):
            self.cum_regrets[(pwr, action)] += regret
            self.cum_sigma[(pwr, action)] += sigma
        self.cum_utility[pwr] += state_utility

        if self.use_optimistic_cfr:
            pos_regrets = [
                max(0, self.cum_regrets[(pwr, a)] + regret)
                for a, regret in zip(actions, action_regrets)
            ]
        else:
            pos_regrets = [max(0, self.cum_regrets[(pwr, a)]) for a in actions]

        sum_pos_regrets = sum(pos_regrets)
        if sum_pos_regrets == 0:
            max_action = max(actions, key=lambda action: self.cum_regrets[(pwr, action)])
            for action in actions:
                self.sigma[(pwr, action)] = float(action == max_action)
        else:
            for action, pos_regret in zip(actions, pos_regrets):
                self.sigma[(pwr, action)] = pos_regret / sum_pos_regrets

    def sorted_policy(self, pwr, probs):
        return dict(
            sorted(zip(self.power_plausible_orders[pwr], probs), key=lambda ac_p: -ac_p[1])
        )


class WeightedAverager:
    def __init__(self):
        self._cum = 0
        self._weight = 0
        self._count = 0

    def accum(self, val, weight):
        self._cum += val * weight
        self._weight += weight
        self._count += 1

    def get_avg(self):
        return self._cum / (self._weight + 1e-8)

    def get_weight(self):
        return self._weight

    def get_count(self):
        return self._count


class SearchBotAgent(BaseSearchAgent):
    """One-ply cfr with policy rollouts"""

    def __init__(self, cfg: agents_cfgs.SearchBotAgent, *, skip_model_cache=False):
        super().__init__(cfg)
        self.model = ModelWrapper(
            cfg.model_path,
            cfg.device,
            cfg.value_model_path,
            cfg.max_batch_size,
            half_precision=cfg.half_precision,
            skip_model_cache=skip_model_cache,
        )
        self.model_rollouts = ModelRollouts(self.model, cfg.rollouts_cfg)
        assert cfg.n_rollouts >= 0, "Set searchbot.n_rollouts"

        self.n_rollouts = cfg.n_rollouts
        self.cache_rollout_results = cfg.cache_rollout_results
        self.precompute_cache = cfg.precompute_cache
        self.enable_compute_nash_conv = cfg.enable_compute_nash_conv
        self.n_plausible_orders = cfg.plausible_orders_cfg.n_plausible_orders
        self.use_optimistic_cfr = cfg.use_optimistic_cfr
        self.use_final_iter = cfg.use_final_iter
        self.use_pruning = cfg.use_pruning
        self.bp_iters = cfg.bp_iters
        self.bp_prob = cfg.bp_prob
        self.loser_bp_iter = cfg.loser_bp_iter
        self.loser_bp_value = cfg.loser_bp_value
        self.share_strategy = cfg.share_strategy
        self.reset_seed_on_rollout = cfg.reset_seed_on_rollout
        self.max_seconds = cfg.max_seconds

        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg, model=self.model
        )
        self.order_aug_cfg = cfg.order_aug

        logging.info(f"Initialized SearchBotAgent: {self.__dict__}")

    def get_orders(self, game, power) -> Action:
        prob_distributions = self.get_all_power_prob_distributions(
            game, early_exit_for_power=power
        )
        logging.info(f"Final strategy: {prob_distributions[power]}")
        logging.info(
            "JSON_PASRING played_strategy %s %s",
            power,
            json.dumps([(list(k), v) for k, v in prob_distributions[power].items()]),
        )
        if len(prob_distributions[power]) == 0:
            return ()
        return sample_p_dict(prob_distributions[power])

    def get_orders_many_powers(
        self, game, powers, timings=None, single_cfr=False, bp_policy=None
    ) -> JointAction:

        if timings is None:
            timings = TimingCtx()
        if single_cfr is None:
            single_cfr = self.share_strategy
        with timings("get_orders_many_powers"):
            # Noop to differentiate from single power call.
            pass
        prob_distributions: PowerPolicies = {}
        if single_cfr:
            inner_timings = TimingCtx()
            prob_distributions = self.get_all_power_prob_distributions(
                game, timings=inner_timings, bp_policy=bp_policy
            )
            timings += inner_timings
        else:
            for power in powers:
                inner_timings = TimingCtx()
                prob_distributions[power] = self.get_all_power_prob_distributions(
                    game,
                    early_exit_for_power=power,
                    timings=inner_timings,
                    bp_policy=bp_policy,
                )[power]
                timings += inner_timings
        all_orders: JointAction = {}
        for power in powers:
            logging.info(f"Final strategy ({power}): {prob_distributions[power]}")
            if len(prob_distributions[power]) == 0:
                all_orders[power] = ()
            else:
                all_orders[power] = sample_p_dict(prob_distributions[power])
        timings.pprint(logging.getLogger("timings").info)
        return all_orders

    def get_plausible_orders_policy(self, game):
        # Determine the set of plausible actions to consider for each power
        policy = self.order_sampler.sample_orders(game)
        policy = augment_plausible_orders(
            game,
            policy,
            self,
            self.order_aug_cfg,
            limits=self.order_sampler.get_plausible_order_limits(game),
        )

        return policy

    def get_all_power_prob_distributions(
        self,
        game: pydipcc.Game,
        *,
        bp_policy: PowerPolicies = None,
        early_exit_for_power: Optional[Power] = None,
        timings: TimingCtx = None,
        extra_plausible_orders=None,
    ) -> PowerPolicies:
        """
        Computes an equilibrium policy for all powers.

        Arguments:
            - game: Game object encoding current game state.
            - bp_policy: If set, overrides the plausible order set and blueprint policy for initialization.
                         Values should be probabilities, but can be set to -1 to simply specify plausible orders;
                         in that case, this function will raise an error if any feature uses the BP distribution (e.g. bp_iters > 0)
            - early_exit_for_power: If set, then if this power has <= 1 plausible order, will exit early without computing a full equilibrium.
            - timings: A TimingCtx object to measure timings
            - extra_plausible_orders: Extra plausible orders to add to the model-computed set.

        Returns:
            - Equilibrium policy dict {power: {action: prob}}
        """
        if timings is None:
            timings = TimingCtx()
        timings.start("one-time")

        deadline: Optional[float] = (
            time.monotonic() + self.max_seconds if self.max_seconds > 0 else None
        )

        # If there are no locations to order, bail
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            return {early_exit_for_power: {tuple(): 1.0}}

        if type(game) != pydipcc.Game:
            game = pydipcc.Game.from_json(json.dumps(game.to_saved_game_format()))

        # If this power has nothing to do, need to search
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            return {early_exit_for_power: {tuple(): 1.0}}

        logging.info(f"BEGINNING CFR get_all_power_prob_distributions")

        rollout_results_cache = RolloutResultsCache()

        if bp_policy is None:
            bp_policy = self.get_plausible_orders_policy(game)

        if extra_plausible_orders:
            for p, orders in extra_plausible_orders.items():
                bp_policy[p].update({order: 0 for order in orders})
                logging.info(f"Adding extra plausible orders {p}: {orders}")

        cfr_data = CFRData(bp_policy, self.use_optimistic_cfr)
        del bp_policy

        # If there are <=1 plausible orders, no need to search
        if (
            early_exit_for_power
            and len(cfr_data.power_plausible_orders[early_exit_for_power]) == 0
        ):
            return {early_exit_for_power: {tuple(): 1.0}}
        if (
            early_exit_for_power
            and len(cfr_data.power_plausible_orders[early_exit_for_power]) == 1
        ):
            return {
                early_exit_for_power: {
                    list(cfr_data.power_plausible_orders[early_exit_for_power]).pop(): 1.0
                }
            }

        if self.enable_compute_nash_conv:
            logging.info("Computing nash conv for blueprint")
            for temperature in (1.0, 0.5, 0.1, 0.01):
                self.compute_nash_conv(
                    cfr_data,
                    f"blueprint T={temperature}",
                    game,
                    lambda power: cfr_data.bp_strategy(power, temperature=temperature),
                )

        def on_miss(set_orders_dicts):
            nonlocal timings
            inner_timings = TimingCtx()
            ret = self.model_rollouts.do_rollouts(
                game, set_orders_dicts, timings=inner_timings, log_timings=False
            )
            timings += inner_timings
            return ret

        # run rollouts or get from cache
        if self.cache_rollout_results and self.precompute_cache:
            num_active_powers = sum(
                len(actions) > 1 for actions in cfr_data.power_plausible_orders.values()
            )
            if num_active_powers > 2:
                logging.warning(
                    "Disabling precomputation of the CFR cache as have %d > 2 active powers",
                    num_active_powers,
                )
            else:
                verbose_log_iter = False  # Hack: this is used in on_miss
                joint_orders = sample_all_joint_orders(cfr_data.power_plausible_orders)
                rollout_results_cache.get(joint_orders, on_miss)

        sampled_action_history = []
        rollout_result_history = []
        power_is_loser = {}  # make typechecker happy
        for cfr_iter in range(self.n_rollouts):
            if cfr_iter > 0 and deadline is not None and time.monotonic() >= deadline:
                logging.info(f"Early exit from CFR after {cfr_iter} iterations by timeout")
                break
            timings.start("start")
            # do verbose logging on 2^x iters
            verbose_log_iter = (
                (cfr_iter & (cfr_iter + 1) == 0 and cfr_iter > self.n_rollouts / 8)
                or cfr_iter == self.n_rollouts - 1
                or (cfr_iter + 1) == self.bp_iters
            )

            self.maybe_do_pruning(cfr_iter=cfr_iter, cfr_data=cfr_data)

            cfr_data.discount_linear_cfr(cfr_iter)

            timings.start("query_policy")
            # get policy probs for all powers
            power_is_loser = {
                pwr: self.is_loser(cfr_data, pwr, cfr_iter, actions)
                for (pwr, actions) in cfr_data.power_plausible_orders.items()
            }
            power_action_ps: Dict[Power, List[float]] = {
                pwr: (
                    cfr_data.bp_strategy(pwr)
                    if (
                        cfr_iter < self.bp_iters
                        or np.random.rand() < self.bp_prob
                        or power_is_loser[pwr]
                    )
                    else cfr_data.strategy(pwr)
                )
                for (pwr, actions) in cfr_data.power_plausible_orders.items()
            }

            timings.start("apply_orders")
            # sample policy for all powers
            idxs, power_sampled_orders = sample_orders_from_policy(
                cfr_data.power_plausible_orders, power_action_ps
            )
            sampled_action_history.append(power_sampled_orders)
            set_orders_dicts = make_set_orders_dicts(
                cfr_data.power_plausible_orders, power_sampled_orders
            )

            timings.stop()
            all_rollout_results = (
                rollout_results_cache.get(set_orders_dicts, on_miss)
                if self.cache_rollout_results
                else on_miss(set_orders_dicts)
            )
            timings.start("cfr")

            for pwr, actions in cfr_data.power_plausible_orders.items():
                if len(actions) == 0:
                    continue

                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )
                rollout_result_history.append((cfr_iter, pwr, results))
                # logging.info(f"Results {pwr} = {results}")
                # calculate regrets
                action_utilities: List[float] = [r[1][pwr] for r in results]
                state_utility: float = np.dot(power_action_ps[pwr], action_utilities)
                action_regrets = [(u - state_utility) for u in action_utilities]

                # log some action values
                if verbose_log_iter:
                    self.log_cfr_iter_state(
                        game=game,
                        pwr=pwr,
                        actions=actions,
                        cfr_data=cfr_data,
                        cfr_iter=cfr_iter,
                        power_is_loser=power_is_loser,
                        state_utility=state_utility,
                        action_utilities=action_utilities,
                        power_sampled_orders=power_sampled_orders,
                    )

                # update cfr data structures
                # FIXME: shouldn't this happen before `log_cfr_iter_state`? (not a big deal)
                cfr_data.update(
                    pwr, actions, state_utility, action_regrets, cfr_data.strategy(pwr)
                )

            if self.enable_compute_nash_conv and verbose_log_iter:
                logging.info(f"Computing nash conv for iter {cfr_iter}")
                self.compute_nash_conv(
                    cfr_data, f"cfr iter {cfr_iter}", game, cfr_data.avg_strategy
                )

            if self.cache_rollout_results and (cfr_iter + 1) % 10 == 0:
                logging.info(f"{rollout_results_cache}")

        timings.start("to_dict")

        # return prob. distributions for each power
        ret = {}
        for p in POWERS:
            final_ps = cfr_data.strategy(p)
            avg_ps = cfr_data.avg_strategy(p)
            bp_ps = cfr_data.bp_strategy(p)
            ps = bp_ps if power_is_loser[p] else (final_ps if self.use_final_iter else avg_ps)
            ret[p] = cfr_data.sorted_policy(p, ps)

        logging.info(
            "Values: %s", {p: f"{x:.3f}" for p, x in zip(POWERS, self.model.get_values(game))}
        )

        timings.stop()
            
        timings.pprint(logging.getLogger("timings").info)
        return ret

    def maybe_do_pruning(self, *, cfr_iter, **kwargs):
        if not self.use_pruning:
            return

        if cfr_iter == 1 + int(self.n_rollouts / 4):
            self.prune_actions(
                cfr_iter=cfr_iter, ave_regret_thresh=-0.06, ave_strat_thresh=0.002, **kwargs
            )

        if cfr_iter == 1 + int(self.n_rollouts / 2):
            self.prune_actions(
                cfr_iter=cfr_iter, ave_regret_thresh=-0.03, ave_strat_thresh=0.001, **kwargs
            )

    @classmethod
    def prune_actions(cls, *, cfr_iter, cfr_data, ave_regret_thresh, ave_strat_thresh):
        for pwr, actions in cfr_data.power_plausible_orders.items():
            paired_list = []
            for action in actions:
                ave_regret = cfr_data.cum_regrets[(pwr, action)] / cfr_data.cum_weight
                new_pair = (action, ave_regret)
                paired_list.append(new_pair)
            paired_list.sort(key=lambda tup: tup[1])
            for (action, ave_regret) in paired_list:
                ave_strat = cfr_data.cum_sigma[(pwr, action)] / cfr_data.cum_weight
                if (
                    ave_regret < ave_regret_thresh
                    and ave_strat < ave_strat_thresh
                    and cfr_data.sigma[(pwr, action)] == 0
                ):
                    cfr_data.cum_sigma[(pwr, action)] = 0
                    logging.info(
                        "pruning on iter {} action {} with ave regret {} and ave strat {}".format(
                            cfr_iter, action, ave_regret, ave_strat
                        )
                    )
                    actions.remove(action)

    def log_cfr_iter_state(
        self,
        *,
        game,
        pwr,
        actions,
        cfr_data,
        cfr_iter,
        power_is_loser,
        state_utility,
        action_utilities,
        power_sampled_orders,
    ):
        logging.info(
            f"<> [ {cfr_iter+1} / {self.n_rollouts} ] {pwr} {game.phase} avg_utility={cfr_data.avg_utility(pwr):.5f} cur_utility={state_utility:.5f} "
            f"is_loser= {int(power_is_loser[pwr])}"
        )
        logging.info(f">> {pwr} cur action at {cfr_iter+1}: {power_sampled_orders[pwr]}")
        logging.info(f"     {'probs':8s}  {'bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders")
        action_probs: List[float] = cfr_data.avg_strategy(pwr)
        bp_probs: List[float] = cfr_data.bp_strategy(pwr)
        avg_utilities = [cfr_data.avg_action_utility(pwr, a) for a in actions]
        sorted_metrics = sorted(
            zip(actions, action_probs, bp_probs, avg_utilities, action_utilities),
            key=lambda ac: -ac[1],
        )
        for orders, p, bp_p, avg_u, cur_u in sorted_metrics:
            logging.info(f"|>  {p:8.5f}  {bp_p:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}")

    def compute_nash_conv(self, cfr_data, label, game, strat_f):
        """For each power, compute EV of each action assuming opponent ave policies"""

        # get policy probs for all powers
        power_action_ps: Dict[Power, List[float]] = {
            pwr: strat_f(pwr) for (pwr, actions) in cfr_data.power_plausible_orders.items()
        }
        logging.info("Policies: {}".format(power_action_ps))

        total_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        temp_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        total_state_utility: Dict[Power, float] = defaultdict(float)
        max_state_utility: Dict[Power, float] = defaultdict(float)
        for pwr, actions in cfr_data.power_plausible_orders.items():
            total_state_utility[pwr] = 0
            max_state_utility[pwr] = 0
        # total_state_utility = [0 for u in idxs]
        nash_conv = 0
        br_iters = 100
        for _ in range(br_iters):
            # sample policy for all powers
            idxs, power_sampled_orders = sample_orders_from_policy(
                cfr_data.power_plausible_orders, power_action_ps
            )

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = make_set_orders_dicts(
                cfr_data.power_plausible_orders, power_sampled_orders
            )
            all_rollout_results = self.model_rollouts.do_rollouts(game, set_orders_dicts)

            for pwr, actions in cfr_data.power_plausible_orders.items():
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
                # for i in range(len(cfr_data.power_plausible_orders[pwr])):
                #     action = cfr_data.power_plausible_orders[pwr][i]
                #     util = action_utilities[i]
                #     logging.info("{} {} = {}".format(pwr,action,util))

                # for action in cfr_data.power_plausible_orders[pwr]:
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
        for pwr, actions in cfr_data.power_plausible_orders.items():
            # ps = self.avg_strategy(pwr, cfr_data.power_plausible_orders[pwr])
            for i in range(len(actions)):
                action = actions[i]
                total_action_utilities[(pwr, action)] /= br_iters
                if total_action_utilities[(pwr, action)] > max_state_utility[pwr]:
                    max_state_utility[pwr] = total_action_utilities[(pwr, action)]
                total_state_utility[pwr] += (
                    total_action_utilities[(pwr, action)] * power_action_ps[pwr][i]
                )

        for pwr, actions in cfr_data.power_plausible_orders.items():
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

        logging.info(f"Nash conv for {label} = {nash_conv}")

    def eval_policy_values(
        self, game: pydipcc.Game, policy: PowerPolicies, n_rollouts: int = 1000,
    ) -> Dict[Power, float]:
        """Compute the EV of a {pwr: policy} dict at a state by running `n_rollouts rollouts.

        Returns:
            - {power: avg_sos}
        """

        power_actions = {pwr: list(p.keys()) for pwr, p in policy.items()}
        power_action_probs = {pwr: list(p.values()) for pwr, p in policy.items()}

        set_orders_dicts = [
            sample_orders_from_policy(power_actions, power_action_probs)[1]
            for _i in range(n_rollouts)
        ]

        rollout_results = self.model_rollouts.do_rollouts(game, set_orders_dicts)

        def mean(L: List[float]):
            return sum(L) / len(L)

        utilities = {
            pwr: mean([values[pwr] for order, values in rollout_results]) for pwr in POWERS
        }
        return utilities

    def is_loser(self, cfr_data, pwr, cfr_iter, plausible_orders):
        if cfr_iter >= self.loser_bp_iter and self.loser_bp_value > 0:
            for action in plausible_orders:
                if cfr_data.avg_action_utility(pwr, action) > self.loser_bp_value:
                    return False
            return True
        return False


class RolloutResultsCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.calls = 0

    def get(
        self, set_orders_dicts: List[Dict[Power, Action]], onmiss_fn
    ) -> List[Tuple[Dict[Power, Action], Dict[Power, float]]]:
        joint_actions = tuple(frozenset(d.items()) for d in set_orders_dicts)
        n_unique = len(frozenset(joint_actions))
        self.calls += n_unique

        unknown_order_dicts = [
            set_orders_dicts[i]
            for i, joint_action in enumerate(joint_actions)
            if joint_action not in self.cache
        ]
        # Minor optimization. Orders may have duplicates.
        unknown_order_dicts = list({frozenset(x.items()): x for x in unknown_order_dicts}.values())
        self.hits += n_unique - len(unknown_order_dicts)
        for r in onmiss_fn(unknown_order_dicts):
            set_order_dict, _ = r
            joint_action = frozenset(set_order_dict.items())
            self.cache[joint_action] = r
        results = [self.cache[joint_action] for joint_action in joint_actions]
        return results

    def __repr__(self):
        return "RolloutResultsCache[hits/calls = {} / {} = {:.3f}]".format(
            self.hits, self.calls, self.hits / self.calls
        )


def augment_plausible_orders(
    game: pydipcc.Game,
    power_plausible_orders: PowerPolicies,
    agent: SearchBotAgent,
    cfg: agents_cfgs.SearchBotAgent.PlausibleOrderAugmentation,
    *,
    limits: Optional[List[int]],
) -> PowerPolicies:
    policy_model = agent.model.model
    augmentation_type = cfg.WhichOneof("augmentation_type")
    if augmentation_type is None:
        return power_plausible_orders
    if not game.current_short_phase.endswith("M"):
        # FIXME(akhti): maybe add a flag for this.
        return power_plausible_orders

    cfg = getattr(cfg, augmentation_type)

    if augmentation_type == "do":
        policy, _ = fairdiplomacy.action_exploration.double_oracle_fva(
            game, agent, double_oracle_cfg=cfg
        )
        return policy

    assert augmentation_type == "random"

    # Creating a copy.
    power_plausible_orders = dict(power_plausible_orders)

    alive_powers = [p for p, score in zip(POWERS, game.get_square_scores()) if score > 1e-3]

    for power in alive_powers:
        actions = fairdiplomacy.action_generation.generate_order_by_column_from_model(
            policy_model, game, power
        )
        logging.info(
            "Found %s actions for %s. Not in plausible: %s",
            len(actions),
            power,
            len(frozenset(actions).difference(power_plausible_orders[power])),
        )
        max_actions = limits[POWERS.index(power)]
        # Creating space for new orders.
        orig_size = len(power_plausible_orders[power])
        power_plausible_orders[power] = dict(
            collections.Counter(power_plausible_orders[power]).most_common(
                max(cfg.min_actions_to_keep, max_actions - cfg.max_actions_to_drop)
            )
        )
        random.shuffle(actions)
        logging.info("Addding extra plausible orders for %s", power)
        if orig_size != len(power_plausible_orders[power]):
            logging.info(
                " (deleted %d least probable actions)",
                orig_size - len(power_plausible_orders[power]),
            )
        for action in actions:
            if len(power_plausible_orders[power]) >= max_actions:
                break
            if action not in power_plausible_orders[power]:
                power_plausible_orders[power][action] = 0
                logging.info("       %s", action)

    renormalize_policy(power_plausible_orders)

    return power_plausible_orders


def sample_all_joint_orders(power_actions: Dict[Power, List[Action]]) -> List[Dict[Power, Action]]:
    power_actions = dict(power_actions)
    for pwr in list(power_actions):
        if not power_actions[pwr]:
            power_actions[pwr] = [tuple()]

    all_orders = []
    powers, action_sets = zip(*power_actions.items())
    for joint_action in itertools.product(*action_sets):
        all_orders.append(dict(zip(powers, joint_action)))
    return all_orders

