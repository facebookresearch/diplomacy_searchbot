# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools to find extra plausible actions given an equilibrium.

See https://docs.google.com/document/d/1TG4cqK92q9KOnhXUKH-LdPQUUb9nvzOTYy5D1g4fj54/edit
"""
from typing import Dict, Callable, Tuple, List, Optional
import copy
import datetime
import functools
import logging
import itertools
import random

import numpy as np
import torch

from conf.agents_cfgs import DoubleOracleExploration
from fairdiplomacy import pydipcc
from fairdiplomacy.action_generation import generate_double_oracle_actions
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.models.diplomacy_model import load_model
from fairdiplomacy.typedefs import Action, Power, PowerPolicies
from fairdiplomacy.utils import timing_ctx
from fairdiplomacy.utils.batching import batched_forward
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
import nest

BATCH_SIZE = 1024

PerPowerList = List


def sample_joint_actions(
    policies: Dict[Power, List[Tuple[Action, float]]], max_samples: int
) -> List[PerPowerList[Action]]:
    """Sample joint actions given a policy."""
    samples_per_power: Dict[Power, Optional[List[Action]]] = {}
    for power, policy in policies.items():
        if not policy:
            samples_per_power[power] = None
        else:
            actions, probs = zip(*policy.items())
            action_ids = torch.multinomial(
                torch.as_tensor(probs), max_samples, replacement=True
            ).tolist()
            samples_per_power[power] = [actions[i] for i in action_ids]
    assert any(samples_per_power.values()), "oops have no moves!"
    combined = []
    for i in range(max_samples):
        combined.append(
            [
                samples_per_power[power][i] if samples_per_power.get(power) else []
                for power in POWERS
            ]
        )
    return combined


@torch.no_grad()
def get_values_from_model(model, games, selected_power, *, num_threads=10) -> torch.Tensor:
    encoder = FeatureEncoder(num_threads)
    power_id = POWERS.index(selected_power)

    def go(game_indices):
        # As we may have a lot of games, we have to encode them as we go.
        game_indices = game_indices.cpu()
        obs = encoder.encode_inputs_state_only(games[game_indices[0] : game_indices[-1] + 1])
        float_dtype = next(iter(model.parameters())).dtype

        def move_tensor(tensor):
            dtype = float_dtype if tensor.dtype is torch.float else tensor.dtype
            return tensor.to(device=device, dtype=dtype)

        obs = nest.map(move_tensor, obs)
        for fake_keys in "temperature x_loc_idxs x_possible_actions".split():
            obs[fake_keys] = None
        return model(**obs, need_policy=False)[3][:, power_id]

    device = next(iter(model.parameters())).device
    values = batched_forward(go, torch.arange(len(games)), batch_size=BATCH_SIZE, device=device)
    for i, game in enumerate(games):
        if game.is_game_done:
            values[i] = game.get_square_scores()[power_id]
    return values


class ScoreActionsCache(dict):
    """Mapping (power, op_action) -> tensor of values for actions."""

    pass


def score_actions(
    selected_power: Power,
    actions: List[Action],
    game: pydipcc.Game,
    critic: Callable,
    equilibrium: Dict[Power, List[Tuple[Action, float]]],
    max_br_orders=10,
    max_exact_actions=None,
    timings=None,
    use_board_state_hashing=False,
    cache: Optional[ScoreActionsCache] = None,
) -> List[float]:
    """Computes EV of actions given policies of oponents.

    If max_br_orders is positive, then will sample this number of joint
    orders and evaluate against them.

    Otherwise, will expect that there is only one opponent and will compute
    exact value against their policy. If max_exact_actions is set, only top
    max_exact_actions will taken from the opponent's policy.

    If use_board_state_hashing, then game states with the same units on the
    board will be considered identical and queried once.

    Returns EV for each action in actions.
    """
    if timings is None:
        timings = timing_ctx.TimingCtx()
    if cache is None:
        cache = ScoreActionsCache()
    timings.start("score_actions.sample_op")
    op_weighted_actions: List[Tuple[PerPowerList[Action], float]]
    op_powers = [
        power
        for power, policy in equilibrium.items()
        if power != selected_power and len(policy) and list(policy.keys()) != [()]
    ]
    if not op_powers:
        # Only we have a move (probably retreat or something).
        joint_action = [tuple()] * len(POWERS)
        op_weighted_actions = [(joint_action, 1.0)]
    elif max_br_orders > 0:
        op_sampled_actions = sample_joint_actions(equilibrium, max_br_orders)
        op_weighted_actions = [(joint_action, 1.0) for joint_action in op_sampled_actions]
    else:
        assert (
            len(op_powers) == 1
        ), f"Exact EV only supported for 2 players. Got {selected_power} and {op_powers}"
        op_weighted_actions = []
        (op_power,) = op_powers
        for action, prob in equilibrium[op_power].items():
            if prob > 1e-3:
                joint_action = [tuple()] * len(POWERS)
                joint_action[POWERS.index(op_power)] = action
                op_weighted_actions.append((joint_action, prob))
                if max_exact_actions and len(op_weighted_actions) >= max_exact_actions:
                    break

    logging.info("Sampled oponent actions: %s", op_weighted_actions)
    timings.stop()
    # print(selected_power, *op_weighted_actions, sep="\n")

    scores = []
    for power_actions, _ in op_weighted_actions:
        # Will be redefined in the loop. Setting to None to make cache work.
        power_actions[POWERS.index(selected_power)] = None
        cache_key = (tuple(power_actions), selected_power)
        if cache_key not in cache:
            cartesian_product_games = []
            with timings("score_actions.step"):
                if use_board_state_hashing:
                    indices = []
                    hashes = []
                    a_scores = torch.zeros(len(actions))
                    for i, action in enumerate(actions):
                        step_game = pydipcc.Game(game)
                        for i, power in enumerate(POWERS):
                            if power == selected_power:
                                step_game.set_orders(power, action)
                            else:
                                step_game.set_orders(power, power_actions[i])
                        step_game.process()
                        step_game_hash = step_game.compute_board_hash()
                        if step_game_hash in cache:
                            a_scores[i] = cache[step_game_hash]
                        else:
                            indices.append(i)
                            hashes.append(step_game_hash)
                            cartesian_product_games.append(step_game)
                else:
                    for action in actions:
                        step_game = pydipcc.Game(game)
                        for i, power in enumerate(POWERS):
                            if power == selected_power:
                                step_game.set_orders(power, action)
                            else:
                                step_game.set_orders(power, power_actions[i])
                        step_game.process()
                        cartesian_product_games.append(step_game)
            with timings("score_actions.model"):
                if use_board_state_hashing:
                    if cartesian_product_games:
                        selected_scores = critic(cartesian_product_games, selected_power)
                        a_scores[indices] = selected_scores
                        cache.update(dict(zip(hashes, selected_scores)))
                    cache[cache_key] = a_scores
                else:
                    cache[cache_key] = critic(cartesian_product_games, selected_power)
        a_scores = cache[cache_key]
        scores.append(a_scores)
    _, weights = zip(*op_weighted_actions)
    weights = torch.as_tensor(weights).unsqueeze(0)
    # Shape action X op_action.
    scores = torch.stack(scores, -1)
    scores = (scores * weights).sum(-1) / weights.sum()
    return scores.tolist()


def compactify(action):
    def cc(order):
        order = order.split(" ", 1)[1]
        order = order.replace(" A ", " ")
        order = order.replace(" F ", " ")
        return order

    return type(action)(map(cc, action))


def build_matrix_fva(game, agent, policies):
    """Compute matrix of EV values for all actions in the policy vs equilibrium."""
    critic = lambda *args, **kwargs: get_values_from_model(
        agent.model.value_model, *args, **kwargs
    )
    actions_aus = list(policies["AUSTRIA"])
    actions_fra = list(policies["FRANCE"])

    cartesian_product_games = []
    for aa in actions_aus:
        for af in actions_fra:
            step_game = pydipcc.Game(game)
            step_game.set_orders("AUSTRIA", aa)
            step_game.set_orders("FRANCE", af)
            step_game.process()
            cartesian_product_games.append(step_game)

    scores_aus = critic(cartesian_product_games, "AUSTRIA").view(
        len(actions_aus), len(actions_fra)
    )
    scores_fra = critic(cartesian_product_games, "FRANCE").view(len(actions_aus), len(actions_fra))
    return scores_aus, scores_fra, actions_aus, actions_fra


def build_matrix_fva_rollouts(game, model_rollouts, policies):
    """Compute matrix of EV values for all actions in the policy vs equilibrium."""
    actions_aus = list(policies["AUSTRIA"])
    actions_fra = list(policies["FRANCE"])

    set_orders_dicts = []
    for aa in actions_aus:
        for af in actions_fra:
            orders = {p: [] for p in POWERS}
            orders["AUSTRIA"] = aa
            orders["FRANCE"] = af
            set_orders_dicts.append(orders)

    _, scores_dicts = zip(*model_rollouts.do_rollouts(game, set_orders_dicts))

    scores_aus = torch.FloatTensor([x["AUSTRIA"] for x in scores_dicts]).view(
        len(actions_aus), len(actions_fra)
    )
    scores_fra = torch.FloatTensor([x["FRANCE"] for x in scores_dicts]).view(
        len(actions_aus), len(actions_fra)
    )
    return scores_aus, scores_fra, actions_aus, actions_fra


def compute_evs_fva(
    game, agent, policies, min_action_prob=0.0
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Compute matrix of EV values for all actions in the policy vs equilibrium.

    Note, this function retuns FloatTensor even if the inputs are half.

    Returns gave EV for AUS and tensors of EVs for AUS and FRA.
    """
    for power, score in zip(POWERS, game.get_square_scores()):
        assert (
            power == "FRANCE" or power == "AUSTRIA" or score < 1e-5
        ), f"Works only for FvA, bur {power} is alive!"
    critic = lambda *args, **kwargs: get_values_from_model(
        agent.model.value_model, *args, **kwargs
    )

    if not policies["AUSTRIA"] or not policies["FRANCE"]:
        d_file = "compute_evs_fva.debug.pt" % datetime.datetime.now()
        torch.save(d_file, dict(game=game.to_json(), policies=policies))
        logging.error("Empty policies. See %s", d_file)
        assert False

    a_actions, a_probs = zip(*policies["AUSTRIA"].items())
    f_actions, f_probs = zip(*policies["FRANCE"].items())

    a_probs = torch.FloatTensor(a_probs)
    f_probs = torch.FloatTensor(f_probs)

    if min_action_prob > 0:
        for probs in (a_probs, f_probs):
            probs[probs < min(probs.max(), min_action_prob)] = 0.0
            probs /= 1.0

    cartesian_sub_product_games = []
    indices = []
    idx = 0
    for i in range(len(a_actions)):
        for j in range(len(f_actions)):
            idx += 1
            if a_probs[i] == 0.0 and f_probs[j] == 0.0:
                continue
            step_game = pydipcc.Game(game)
            step_game.set_orders("AUSTRIA", a_actions[i])
            step_game.set_orders("FRANCE", f_actions[j])
            step_game.process()
            cartesian_sub_product_games.append(step_game)
            indices.append(idx - 1)

    scores_aus = torch.zeros((len(a_actions), len(f_actions)))
    scores_aus.view(-1)[torch.as_tensor(indices)] = critic(
        cartesian_sub_product_games, "AUSTRIA"
    ).float()

    ev_aus = torch.mv(scores_aus, f_probs)
    ev_fra = torch.mv(1.0 - scores_aus.T, a_probs)

    game_ev_aus = torch.dot(ev_aus, a_probs).item()

    return game_ev_aus, ev_aus, ev_fra


def compute_evs_fva_with_rollouts(
    game, policies, model_rollouts, min_action_prob=0.0
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Compute matrix of EV values for all actions in the policy vs equilibrium.

    Returns gave EV for AUS and tensors of EVs for AUS and FRA.
    """
    for power, score in zip(POWERS, game.get_square_scores()):
        assert (
            power == "FRANCE" or power == "AUSTRIA" or score < 1e-5
        ), f"Works only for FvA, bur {power} is alive!"

    if not policies["AUSTRIA"] or not policies["FRANCE"]:
        d_file = "compute_evs_fva.debug.pt" % datetime.datetime.now()
        torch.save(d_file, dict(game=game.to_json(), policies=policies))
        logging.error("Empty policies. See %s", d_file)
        assert False

    a_actions, a_probs = zip(*policies["AUSTRIA"].items())
    f_actions, f_probs = zip(*policies["FRANCE"].items())

    a_probs = torch.FloatTensor(a_probs)
    f_probs = torch.FloatTensor(f_probs)

    if min_action_prob > 0:
        for probs in (a_probs, f_probs):
            probs[probs < min(probs.max(), min_action_prob)] = 0.0
            probs /= 1.0

    set_orders_dicts = []
    indices = []
    idx = 0
    for i in range(len(a_actions)):
        for j in range(len(f_actions)):
            idx += 1
            if a_probs[i] == 0.0 and f_probs[j] == 0.0:
                continue
            orders = {p: [] for p in POWERS}
            orders["AUSTRIA"] = a_actions[i]
            orders["FRANCE"] = f_actions[j]
            set_orders_dicts.append(orders)
            indices.append(idx - 1)

    _, scores_dicts = zip(*model_rollouts.do_rollouts(game, set_orders_dicts))

    scores_aus = torch.zeros((len(a_actions), len(f_actions)))
    scores_aus.view(-1)[torch.as_tensor(indices)] = torch.as_tensor(
        [x["AUSTRIA"] for x in scores_dicts]
    )

    ev_aus = torch.mv(scores_aus, f_probs)
    ev_fra = torch.mv(1.0 - scores_aus.T, a_probs)

    game_ev_aus = torch.dot(ev_aus, a_probs).item()

    return game_ev_aus, ev_aus, ev_fra


def double_oracle_fva(
    game: pydipcc.Game,
    agent,
    *,
    double_oracle_cfg: DoubleOracleExploration,
    need_actions_only=True,
    play_agent=None,
    initial_plausible_orders_policy: Optional[PowerPolicies] = None,
    timings=None,
) -> Tuple[PowerPolicies, Dict]:
    """Run a double oracle on top of a seach agent.

    Currently implemented only for FvA games.

    Return object has type PowerPolicies, but the values are guaranteed to
    have meaning only if need_actions_only=False. If need_actions_only=True,
    the caller should compute a policy on these plausible actions manually.

    If need_actions_only is set and play_agent is provided, this agent to be
    used to compute the policy. Otherwise, default agent is used.
    """
    max_iters = double_oracle_cfg.max_iters or int(1e100)
    if timings is None:
        timings = timing_ctx.TimingCtx()
    timings.start("do.main")
    allowed_powers = "AUSTRIA FRANCE".split()
    if double_oracle_cfg.shuffle_powers:
        random.shuffle(allowed_powers)
        allowed_powers = allowed_powers[:max_iters]
    elif len(allowed_powers) > max_iters:
        logging.warning(
            "DoubleOracle: shuffling is disabled, but doing only %s iterations given %s powers",
            max_iters,
            len(allowed_powers),
        )

    if initial_plausible_orders_policy is None:
        with timings("do.plausible_orders"):
            initial_plausible_orders_policy = agent.order_sampler.sample_orders(game)
    # Last policies refers to a policy computed by `agent` on `plausible_orders`.
    inner_timings = timing_ctx.TimingCtx()
    last_policies = agent.get_all_power_prob_distributions(
        game, bp_policy=initial_plausible_orders_policy, timings=inner_timings
    )
    timings += inner_timings

    timings.start("do.generate_actions")
    all_actions = generate_double_oracle_actions(
        double_oracle_cfg.generation,
        game,
        agent,
        allowed_powers,
        initial_plausible_orders_policy,
        last_policies,
    )
    timings.stop()
    logging.info(
        "DoubleOracle: generated action sets: %s", {k: len(v) for k, v in all_actions.items()}
    )

    critic = lambda *args, **kwargs: get_values_from_model(
        agent.model.value_model, *args, **kwargs
    )

    if double_oracle_cfg.min_diff_percentage > 0:
        assert (
            1.0 <= double_oracle_cfg.min_diff_percentage <= 100.0
        ), "Must be a percentage, not share"

    steps_since_update = 0
    num_iters = 0
    num_changes = 0
    plausible_orders_policy = copy.deepcopy(initial_plausible_orders_policy)
    score_action_cache = ScoreActionsCache()

    for power in itertools.cycle(allowed_powers):
        if steps_since_update >= len(allowed_powers) or num_iters >= max_iters:
            break
        logging.info(f"DoubleOracle: starting iteration {num_iters} for power {power}")
        steps_since_update += 1
        num_iters += 1

        if not plausible_orders_policy[power] or not all_actions[power]:
            continue

        # print()
        if last_policies is None:
            inner_timings = timing_ctx.TimingCtx()
            last_policies = agent.get_all_power_prob_distributions(
                game, bp_policy=plausible_orders_policy, timings=inner_timings
            )
            timings += inner_timings
        assert len(last_policies[power]) == len(plausible_orders_policy[power])
        actions = sorted(set(all_actions[power]).union(plausible_orders_policy[power]))
        scores = score_actions(
            power,
            actions,
            game,
            critic,
            last_policies,
            max_br_orders=-1,
            max_exact_actions=double_oracle_cfg.max_op_actions,
            timings=timings,
            use_board_state_hashing=double_oracle_cfg.use_board_state_hashing,
            cache=score_action_cache,
        )
        with timings("do.select"):
            res = sorted(zip(actions, scores), key=lambda x: -x[1])
            best_known_ev = max(score for action, score in res if action in last_policies[power])
            best_action, best_score = res[0]
        relative_diff = best_score / max(best_known_ev, 1e-4) * 100 - 100.0
        logging.info(
            f"{power}: Current EV {best_known_ev} Best new EV: {best_score} Diff: {best_score - best_known_ev} Rel diff: {relative_diff:.1f}%"
        )
        if (
            best_score - best_known_ev > double_oracle_cfg.min_diff
            and relative_diff > double_oracle_cfg.min_diff_percentage
        ):
            logging.info(f"{power}: Adding to plausible: {best_action}")
            plausible_orders_policy[power][best_action] = 0
            steps_since_update = 0
            num_changes += 1
            if double_oracle_cfg.regenerate_every_iter:
                score_action_cache.clear()
                all_actions[power] = generate_double_oracle_actions(
                    double_oracle_cfg.generation,
                    game,
                    agent,
                    [power],
                    plausible_orders_policy,
                    last_policies,
                )[power]
            last_policies = None
        # for a, score in res[:20]:
        #     print("ev=%.3f sprob=%.3f %s" % (score, policies[power].get(a, 9.999), compactify(a)))
        # for a in plausible_orders[power]:
        #     score = dict(res)[a]
        #     print("ev=%.3f sprob=%.3f %s" % (score, policies[power].get(a, 9.999), compactify(a)))
        # print(build_matrix(policies)[allowed_powers.index(power)])

    stats = {}
    stats["num_changes"] = num_changes
    stats["num_iters"] = num_iters

    if need_actions_only:
        return plausible_orders_policy, stats

    if last_policies is not None and play_agent is None:
        return last_policies, stats

    inner_timings = timing_ctx.TimingCtx()
    policies = (play_agent or agent).get_all_power_prob_distributions(
        game, bp_policy=plausible_orders_policy, timings=inner_timings
    )
    timings += inner_timings
    return policies, stats
