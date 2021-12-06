# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import List, Dict, Tuple, Union

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state

from fairdiplomacy.typedefs import Power, JointAction, PlausibleOrders


class BaseSearchAgent(BaseAgent):
    def __init__(self, cfg):
        assert cfg.max_batch_size > 0
        self.max_batch_size = cfg.max_batch_size

    def do_model_request(self, *args, **kwargs):
        raise RuntimeError(
            "This was a temporary stub during agent refactor; should not be called."
        )


def average_score_dicts(score_dicts: List[Dict]) -> Dict:
    return {p: sum(d.get(p, 0) for d in score_dicts) / len(score_dicts) for p in POWERS}


def n_move_phases_later(from_phase, n):
    if n == 0:
        return from_phase
    year_idx = int(from_phase[1:-1]) - 1901
    season = from_phase[0]
    from_move_phase_idx = 2 * year_idx + (1 if season in "FW" else 0)
    to_move_phase_idx = from_move_phase_idx + n
    to_move_phase_year = to_move_phase_idx // 2 + 1901
    to_move_phase_season = "S" if to_move_phase_idx % 2 == 0 else "F"
    return f"{to_move_phase_season}{to_move_phase_year}M"


def get_square_scores_from_game(game):
    return [
        compute_game_scores_from_state(power_idx, game.get_state()).square_score
        for power_idx in range(len(POWERS))
    ]


def num_orderable_units(game_state, power):
    if game_state["name"][-1] == "A":
        return abs(game_state["builds"].get(power, {"count": 0})["count"])
    if game_state["name"][-1] == "R":
        return len(game_state["retreats"].get(power, []))
    else:
        return len(game_state["units"].get(power, []))


def sample_orders_from_policy(
    power_actions: PlausibleOrders,
    power_action_probs: Union[
        Dict[Power, List[float]], Dict[Power, torch.Tensor]
    ],  # make typechecker happy (ask Adam)
) -> Tuple[Dict[Power, int], JointAction]:
    """
    Sample orders for each power from an action distribution (i.e. policy).

    Arguments:
        - power_actions: a list of plausible orders for each power
        - power_action_probs: probabilities for each of the power_actions

    Returns:
        - A dictionary of order indices for each power sampled out of the action distribution
        - A dictionary of orders for each power sampled out of the action distribution
    """
    sampled_idxs = {}
    power_sampled_orders = {}
    for power, action_probs in power_action_probs.items():
        if len(action_probs) <= 0:
            power_sampled_orders[power] = ()
        else:
            # Manually iterating is faster than np.random.choice, it turns out - np.random.choice
            # is surprisingly slow (possibly due to conversion to numpy array).
            # It also saves us from having to manually normalize the array if the policy
            # probabilities don't add up exactly to 1.0 due to float wonkiness.
            lenprobs = len(action_probs)
            sumprobs = sum(action_probs)
            r = sumprobs * np.random.random()
            idx = 0
            while idx < lenprobs - 1:
                r -= action_probs[idx]
                if r < 0:
                    break
                idx += 1
            sampled_idxs[power] = idx
            power_sampled_orders[power] = power_actions[power][idx]

    return sampled_idxs, power_sampled_orders


def make_set_orders_dicts(
    power_actions: PlausibleOrders,
    power_sampled_orders: JointAction,
    traverser_powers: List[Power] = None,
) -> List[JointAction]:
    """
    Construct a list of set_orders dicts for CFR traversal, that can be
    used as an input to BaseSearchAgent.do_rollout.

    Arguments:
        - power_actions: a list of plausible orders for each power
        - power_sampled_orders: orders for each power
        - traverser_powers: a list of powers for whom each plausible order should be
          sampled in the output dict

    Returns:
        - A list of Power -> Action dicts, where each one has one of the plausible orders
        for one of the traverser_powers, and the sampled orders for all other powers.
        Outputs are ordered by traverser_power, then by index in power_plausible_order[pwr].
    """

    if traverser_powers is None:
        traverser_powers = POWERS

    # for each power: compare all actions against sampled opponent action
    return [
        {**{p: a for p, a in power_sampled_orders.items()}, pwr: action}
        for pwr in traverser_powers
        for action in power_actions[pwr]
    ]
