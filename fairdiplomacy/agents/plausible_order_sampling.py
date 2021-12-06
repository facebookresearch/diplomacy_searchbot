# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import Counter, defaultdict
from math import ceil
import torch
from typing import List, Optional

from fairdiplomacy import pydipcc
from conf import agents_cfgs
from fairdiplomacy.agents.base_search_agent import num_orderable_units  # FIXME
from fairdiplomacy.agents.model_wrapper import ModelWrapper
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Action, Power, PowerPolicies
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder

def renormalize_policy(policy: PowerPolicies) -> None:
    for orders_to_probs in policy.values():
        if len(orders_to_probs) > 0:
            total_prob = sum(orders_to_probs.values())
            assert total_prob > 0
            for orders in orders_to_probs:
                orders_to_probs[orders] /= total_prob


class PlausibleOrderSampler:
    def __init__(
        self,
        cfg: agents_cfgs.PlausibleOrderSampling,
        model: Optional[ModelWrapper] = None,
    ):
        self.cfg = cfg
        self.n_plausible_orders = cfg.n_plausible_orders
        assert self.n_plausible_orders > 0
        self.max_actions_units_ratio = cfg.max_actions_units_ratio
        self.exclude_n_holds = cfg.exclude_n_holds
        self.req_size = cfg.req_size
        self.batch_size = cfg.batch_size or self.req_size
        self.model = model

    def get_plausible_order_limits(self, game: pydipcc.Game) -> List[int]:
        """Returns the max # plausible actions that should be sampled for each power
        in the state specified by `game`, based on the specified config and the number

        of units for that power.
        Returns:
            - A list of 7 ints corresponding to the max number of plausible actions that
            should be sampled for each power in POWERS.
        """
        limits = [self.n_plausible_orders] * len(POWERS)
        if self.max_actions_units_ratio > 0:
            game_state = game.get_state()
            power_n_units = [num_orderable_units(game_state, p) for p in POWERS]
            limits = [
                min(limit, ceil(u * self.max_actions_units_ratio))
                for limit, u in zip(limits, power_n_units)
            ]
        return limits

    def _log_orders(self, game: pydipcc.Game, policies: PowerPolicies) -> None:
        logging.info("Plausible orders:")
        limit = self.get_plausible_order_limits(game)
        for power, orders_to_probs in policies.items():
            logging.info(
                f"    {power} ( found {len(orders_to_probs)} / {limit[POWERS.index(power)]} )"
            )
            logging.info("        prob,order")
            for orders, probs in orders_to_probs.items():
                logging.info(f"        {probs:10.5f}  {orders}")

    def sample_orders(
        self, game: pydipcc.Game
    ) -> PowerPolicies:
        """
        Sample a set of orders for each power. Return the distribution over these orders (policies).

        Returns: A dictionary of Action -> Prob(Action) for each power.
        """
        logging.info("Starting sample_orders...")
        if self.model:
            ret = self._sample_orders(game)
        else:
            raise RuntimeError()

        # take the limit most common orders per power
        limits = self.get_plausible_order_limits(game)
        ret_limit = {
            power: {
                orders: probs
                for orders, probs in sorted(orders_to_probs.items(), key=lambda x: -x[1])[:limit]
            }
            for limit, (power, orders_to_probs) in zip(limits, ret.items())
        }

        # renormalize after cutting off
        renormalize_policy(ret_limit)

        self._log_orders(game, ret_limit)
        return ret_limit

    def _sample_orders(self, game, *, temperature=1.0, top_p=1.0,) -> PowerPolicies:
        n_samples = self.req_size
        batch_size = self.batch_size
        assert n_samples % batch_size == 0, f"{n_samples}, {batch_size}"

        counters = {p: Counter() for p in POWERS}

        # Encode the game once
        batch_inputs = FeatureEncoder().encode_inputs([game])

        orders_to_logprobs = {}
        for _ in range(n_samples // batch_size):
            # Use batch_repeat_interleave so that the model behaves as if we'd duplicated
            # the input batch_size many times - taking that many policy samples.
            batch_orders, batch_order_logprobs, _ = self.model.do_model_request(
                batch_inputs, temperature, top_p, batch_repeat_interleave=batch_size
            )
            batch_orders = list(zip(*batch_orders))  # power -> list[orders]
            batch_order_logprobs = batch_order_logprobs.t()  # [7 x B]
            for p, power in enumerate(POWERS):
                counters[power].update(batch_orders[p])

            # slow and steady
            for power_orders, power_scores in zip(batch_orders, batch_order_logprobs):
                for order, score in zip(power_orders, power_scores):
                    if order not in orders_to_logprobs:
                        orders_to_logprobs[order] = score
                    else:
                        assert (
                            abs(orders_to_logprobs[order] - score)
                            < 0.2  # very loose tolerance, for fp16
                        ), f"{order} : {orders_to_logprobs[order]} != {score}"

        # filter out badly-coordinated actions
        assert self.exclude_n_holds <= 0, "FIXME"

        def sort_key(order_count_pair):
            order, _ = order_count_pair
            return (-int(are_supports_coordinated(order)), -orders_to_logprobs[order])

        most_common = {
            power: sorted(counter.most_common(), key=sort_key)
            for power, counter in counters.items()
        }

        logging.info(
            "get_plausible_orders(n={}, t={}) found {} unique sets, n_0={}".format(
                n_samples,
                temperature,
                list(map(len, counters.values())),
                [safe_idx(most_common[p], 0, default=(None, None))[1] for p in POWERS],
            )
        )

        orders_to_probs = {}
        for pwr, orders_and_counts in most_common.items():
            logprobs = torch.tensor(
                [orders_to_logprobs[orders] for orders, _ in orders_and_counts]
            )
            probs = logprobs.softmax(dim=0)
            orders_to_probs[pwr] = {
                orders: prob for (orders, _), prob in zip(orders_and_counts, probs)
            }

        return orders_to_probs



def is_n_holds(orders: Action, max_holds) -> bool:
    return len(orders) >= max_holds and all([o.endswith(" H") for o in orders])


def filter_keys(d, fn, log_warn=False):
    """Return a copy of a dict-like input containing the subset of keys where fn(k) is truthy"""
    r = type(d)()
    for k, v in d.items():
        if fn(k):
            r[k] = v
        elif log_warn:
            logging.warning(f"filtered bad key: {k}")
    return r


def are_supports_coordinated(orders: Action) -> bool:
    """Return False if any supports or convoys are not properly coordinated

    e.g. if "F BLA S A SEV - RUM", return False if "A SEV" is not ordered "A SEV - RUM"
             0  1  2 3  4  5  6
    """
    required = {}
    ordered = {}

    for order in orders:
        split = order.split()
        ordered[split[1]] = split  # save by location
        if split[2] in ("S", "C"):
            if split[4] in required and required[split[4]] != split[3:]:
                # an order is already required of this unit, but it contradicts this one
                return False
            else:
                required[split[4]] = split[3:]

    for req_loc, req_order in required.items():
        if req_loc not in ordered:
            # supporting a foreign unit is always allowed, since we can't
            # control the coordination
            continue

        actual_order = ordered[req_loc]

        if len(req_order) == 2 and actual_order[2] == "-":
            # we supported a hold, but it tried to move
            return False
        elif (
            len(req_order) > 2
            and req_order[2] == "-"
            and (actual_order[2] != "-" or actual_order[3][:3] != req_order[3][:3])
        ):
            # we supported a move, but the order given was (1) not a move, or
            # (2) a move to the wrong destination
            return False

    # checks passed, return True
    return True


def safe_idx(seq, idx, default=None):
    try:
        return seq[idx]
    except IndexError:
        return default
