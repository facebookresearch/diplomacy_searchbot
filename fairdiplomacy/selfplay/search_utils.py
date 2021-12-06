# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple
import time
import torch

from fairdiplomacy.agents.model_wrapper import compute_action_logprobs
from fairdiplomacy.data.dataset import encode_power_actions
from fairdiplomacy.models.consts import POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.diplomacy_model.order_vocabulary import EOS_IDX
from fairdiplomacy.selfplay.rollout import order_logits_to_action_logprobs
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global

import nest


def unparse_device(device: str) -> int:
    if device == "cpu":
        return -1
    assert device.startswith("cuda:"), device
    return int(device.split(":")[-1])


@torch.no_grad()
def create_research_targets_single_rollout(
    is_explore_tensor: torch.Tensor,
    episode_reward: torch.Tensor,
    predicted_values: torch.Tensor,
    alive_powers: torch.Tensor,
    discounting: float = 1.0,
) -> torch.Tensor:
    """Creates a target for value function.

    Args:
        is_explore_tensor: bool tensor [T, power].
        episode_reward: float tensor [power].
        alive_powers: bool tensor [T, power], alive power at the beginning of the end of a phase.
        predicted_values: float tensor [T, power].
        discounting: simplex discounting factor.

    Returns:
        tatgets: float tensor [T, power].
    """
    assert is_explore_tensor.shape[1:] == episode_reward.shape, (
        is_explore_tensor.shape,
        episode_reward.shape,
    )
    assert is_explore_tensor.shape == predicted_values.shape, (
        is_explore_tensor.shape,
        predicted_values.shape,
    )
    # True in production but not in tests
    # assert is_explore_tensor.shape[1] == len(POWERS)
    explore_powers_len = is_explore_tensor.shape[1]

    # Make it so that when any power explores, every power bootstraps at that point
    is_explore_tensor = torch.any(is_explore_tensor, dim=1, keepdim=True)
    is_explore_tensor = torch.repeat_interleave(is_explore_tensor, explore_powers_len, dim=1)

    alive_powers = alive_powers.float()
    # Assuming being alive after 1 phase <-> being alive at the start of the game.
    alive_powers = torch.cat([alive_powers[:1], alive_powers[:-1]], dim=0)
    current_value = episode_reward
    targets = []
    for i in range(len(is_explore_tensor) - 1, -1, -1):
        current_value = torch.where(is_explore_tensor[i], predicted_values[i], current_value)
        targets.append(current_value)
        if discounting < 1.0:
            simplex_center = alive_powers[i] / alive_powers[i].sum()
            simplex_direction = simplex_center - current_value
            current_value = current_value + simplex_direction * (1 - discounting)
    targets = torch.stack(list(reversed(targets))).detach()
    return targets


def all_power_prob_distributions_to_tensors(
    all_power_prob_distributions,
    max_actions: int,
    x_possible_actions: torch.Tensor,
    x_in_adj_phase: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(x_possible_actions.shape) == 3 and x_possible_actions.shape[0] == len(
        POWERS
    ), x_possible_actions.shape
    # Converting the policies to 2 tensors:
    #   orders (7 X max_actions x MAX_SEQ_LEN)
    #   probs (7 x max_actions)
    orders_tensors = torch.full((len(POWERS), max_actions, MAX_SEQ_LEN), EOS_IDX, dtype=torch.long)
    probs_tensor = torch.zeros((len(POWERS), max_actions))
    for power_idx, power in enumerate(POWERS):
        if not all_power_prob_distributions.get(power):
            # Skipping non-acting powers.
            continue
        actions, probs = zip(*all_power_prob_distributions[power].items())
        for action_idx, action in enumerate(actions):
            action_tensor, good = encode_power_actions(
                action, x_possible_actions[power_idx], x_in_adj_phase
            )
            assert good, (power, action, x_possible_actions[power_idx])
            orders_tensors[power_idx, action_idx] = action_tensor
        probs_tensor[power_idx, : len(probs)] = torch.as_tensor(probs)
    return orders_tensors, probs_tensor


def compute_search_policy_cross_entropy(model, obs, search_policy_orders, search_policy_probs):
    """Compute cross entropy loss between model's predictions and SearchBot policy.

    Args:
        model: model
        obs: dict of obs.
        search_policy_orders: Long tensor [T, B, 7, max_actions, MAX_SEQ_LEN]
        search_policy_probs: Float tensor [T, B, 7, max_actions]

    Returns:
        loss, scalar
    """
    # Shape: [T, B, 7, max_actions, MAX_SEQ_LEN]
    time_sz, batch_sz, seven, max_actions, _ = search_policy_orders.shape
    assert seven == 7, search_policy_orders.shape

    def repeat_single_dim(tensor, dim, times):
        repeats = [1 for _ in tensor.shape]
        repeats[dim] = times
        return tensor.repeat(*repeats)

    # make it [ T * B * max_actions, ....].
    flat_repeated_obs = nest.map(
        lambda x: repeat_single_dim(x.flatten(end_dim=1).unsqueeze(1), 1, max_actions).flatten(
            end_dim=1
        ),
        obs,
    )
    # new shape: [T * B * max_actions, 7, MAX_SEQ_LEN]
    flat_search_policy_local_orders = search_policy_orders.transpose(2, 3).flatten(end_dim=2)
    flat_search_policy_global_orders = local_order_idxs_to_global(
        flat_search_policy_local_orders, flat_repeated_obs["x_possible_actions"]
    ).long()
    # new shape: [T * B * max_actions, 7]
    flat_search_policy_probs = search_policy_probs.transpose(2, 3).flatten(end_dim=2)

    # shape: [T * B * max_actions]
    has_any_actions = (flat_search_policy_local_orders != EOS_IDX).any(-1).any(-1)

    flat_repeated_obs = nest.map(lambda x: x[has_any_actions], flat_repeated_obs)
    flat_search_policy_local_orders = flat_search_policy_local_orders[has_any_actions]
    flat_search_policy_global_orders = flat_search_policy_global_orders[has_any_actions]
    flat_search_policy_probs = flat_search_policy_probs[has_any_actions]

    # Logits shape: [T * B, 7, MAX_SEQ_LEN]
    # print(
    #     nest.map(
    #         lambda x: x.shape,
    #         dict(
    #             **nest.map(lambda x: x.flatten(end_dim=1), obs),
    #             teacher_force_orders=flat_search_policy_global_orders.clamp(
    #                 min=0
    #             ),  # EOS_IDX = -1 -> 0
    #         ),
    #     )
    # )
    # _, _, logits, _ = model(
    #     **nest.map(lambda x: x.flatten(end_dim=1), obs),
    #     temperature=1.0,
    #     teacher_force_orders=flat_search_policy_global_orders.clamp(min=0),  # EOS_IDX = -1 -> 0
    # )
    # # shape: [T * B, max_actions, 7, MAX_SEQ_LEN]
    # flat_logits = repeat_single_dim(logits.unsqueeze(1), 1, max_actions)
    # # Shape: [valid subset of (T * B * max_actions), 7, MAX_SEQ_LEN].
    # flat_logits = flat_logits.flatten(end_dim=1)[has_any_actions]
    # # Shape: [valid subset of (T * B * max_actions)].
    # flat_action_logprobs = order_logits_to_action_logprobs(logits, flat_search_policy_local_orders)
    print(
        nest.map(
            lambda x: x.shape,
            dict(**flat_repeated_obs, teacher_force_orders=flat_search_policy_global_orders),
        )
    )
    _, _, logits, _ = model(
        **flat_repeated_obs,
        temperature=1.0,
        teacher_force_orders=flat_search_policy_global_orders.clamp(min=0),  # EOS_IDX = -1 -> 0
        need_value=False,
    )

    # Shape: [valid subset of (T * B * max_actions)].
    flat_action_logprobs = order_logits_to_action_logprobs(logits, flat_search_policy_local_orders)
    loss = -(flat_search_policy_probs * flat_action_logprobs).sum() / (time_sz * batch_sz)
    return loss


def compute_search_policy_cross_entropy_sampled(
    model,
    obs,
    search_policy_orders,
    search_policy_probs,
    blueprint_probs=None,
    mask=None,
    mse_loss=False,
    mse_bp_normalized=False,
    mse_bp_upper_bound=None,
    is_move_phase: Optional[torch.Tensor] = None,
    using_ddp=False,
):
    """Compute cross entropy loss between model's predictions and SearchBot policy.

    Uses a single sample from the model's policy

    Args:
        model: model
        obs: dict of obs.
        search_policy_orders: Long tensor [T, B, 7, max_actions, MAX_SEQ_LEN]
        search_policy_probs: Float tensor [T, B, 7, max_actions]
        blueprint_probs: None or Float tensor [T, B, 7, max_actions]
        mask: Bool tensor [T, B]. True, if the policy actually was
            computed at the position.
        is_move_phase: if set, should be [T, B] bool tensor. Used only for stats.
        using_ddp: for distributed data parallel, set True to optimize tensor padding

    Returns: a tuple (loss, metircs)
        loss, scalar
        metrics: dict
    """
    if mask is not None:
        # We'll gather all valid phases and reshape them into something with
        # time dimension equals 1. This only works for non-recurrent networks.
        obs = nest.map(lambda x: _apply_mask(x, mask), obs)
        search_policy_orders = _apply_mask(search_policy_orders, mask)
        search_policy_probs = _apply_mask(search_policy_probs, mask)
        if blueprint_probs is not None:
            blueprint_probs = _apply_mask(blueprint_probs, mask)
        if is_move_phase is not None:
            is_move_phase = _apply_mask(is_move_phase, mask)

    # Shape: [T, B, 7, max_actions, MAX_SEQ_LEN]
    time_sz, batch_sz, seven, max_actions, _ = search_policy_orders.shape
    assert seven == 7, search_policy_orders.shape

    # Prepare to sample an action
    # new shape: [T * B * 7, max_actions]
    flat_search_policy_probs = search_policy_probs.flatten(end_dim=2)
    # new shape: [T * B * 7, max_actions, MAX_SEQ_LEN]
    flat_search_policy_local_orders = search_policy_orders.flatten(end_dim=2)

    # Sample an action
    # shape: [T * B * 7, 1]
    sampled_action_indcies = torch.multinomial(
        flat_search_policy_probs + (flat_search_policy_probs.sum(-1) < 1e-3).float().unsqueeze(-1),
        num_samples=1,
    )

    # Select the action.
    # new shape: [T * B * 7, 1]
    flat_search_policy_probs = torch.gather(flat_search_policy_probs, 1, sampled_action_indcies)
    # new shape: [T * B * 7, 1, MAX_SEQ_LEN]
    flat_search_policy_local_orders = torch.gather(
        flat_search_policy_local_orders,
        1,
        sampled_action_indcies.unsqueeze(-1).expand(time_sz * batch_sz * seven, 1, MAX_SEQ_LEN),
    )

    # Reshape power dimension back.
    flat_search_policy_probs = flat_search_policy_probs.view(time_sz * batch_sz, seven)
    flat_search_policy_local_orders = flat_search_policy_local_orders.view(
        time_sz * batch_sz, seven, MAX_SEQ_LEN
    )
    flat_obs = nest.map(lambda x: x.flatten(end_dim=1), obs)
    flat_search_policy_global_orders = local_order_idxs_to_global(
        flat_search_policy_local_orders, flat_obs["x_possible_actions"]
    ).long()

    # Logits shape: [T * B, 7, MAX_SEQ_LEN, 469]
    _, _, logits, _ = model(
        **flat_obs,
        temperature=1.0,
        teacher_force_orders=flat_search_policy_global_orders.clamp(min=0),  # EOS_IDX = -1 -> 0
        pad_to_max=(not using_ddp),
        need_value=False,
    )
    logprobs = compute_action_logprobs(flat_search_policy_local_orders, logits)

    # Shape: [T * B, 7].
    valid_actions = (flat_search_policy_local_orders != EOS_IDX).any(-1).float()

    metrics = {}
    metrics["loss_inner/valid_share"] = valid_actions.mean()
    if mse_loss:
        if mse_bp_normalized:
            assert blueprint_probs is not None
            assert mse_bp_upper_bound is not None
            weights = blueprint_probs.view(time_sz * batch_sz, seven, -1).sum(-1) + 1e-10
            weights.clamp_(max=mse_bp_upper_bound)
        else:
            weights = 1.0
        scaled_flat_search_policy_probs = flat_search_policy_probs * weights
        probs = logprobs.exp()
        loss = torch.nn.functional.mse_loss(
            probs, scaled_flat_search_policy_probs, reduction="none"
        )
        loss = (loss * valid_actions).mean() / valid_actions.mean()
        metrics["loss_inner/bp_weight"] = (
            weights.mean() if isinstance(weights, torch.Tensor) else weights
        )
        metrics["loss_inner/bp_weight_valid"] = (
            weights * valid_actions
        ).mean() / valid_actions.mean()
        metrics["loss_inner/scaled_bp"] = scaled_flat_search_policy_probs.mean()
        metrics["loss_inner/predicted_prob"] = probs.mean()
    else:
        assert not mse_bp_normalized
        if is_move_phase is not None:
            with torch.no_grad():
                valid_move_mask = valid_actions * is_move_phase.view(-1, 1).float()
                metrics["loss_inner/loss_moves"] = -(logprobs * valid_move_mask).mean() / (
                    valid_move_mask.mean() + 1e-10
                )
        loss = -(logprobs * valid_actions).mean() / valid_actions.mean()
    return loss, metrics


def _apply_mask(x, mask):
    return x.flatten(end_dim=1)[mask.view(-1)].unsqueeze(0)


def compute_search_policy_entropy(search_policy_orders, search_policy_probs, mask=None):
    """Compute entropy of the search policy.

    Args:
        search_policy_orders: Long tensor [T, B, 7, max_actions, MAX_SEQ_LEN]
        search_policy_probs: Float tensor [T, B, 7, max_actions]
        mask: None or bool tensor [T, B]

    Returns:
        entropy, scalar
    """
    if mask is not None:
        search_policy_orders = _apply_mask(search_policy_orders, mask)
        search_policy_probs = _apply_mask(search_policy_probs, mask)
    search_policy_orders = search_policy_orders.flatten(end_dim=2)
    search_policy_probs = search_policy_probs.flatten(end_dim=2)
    has_actions = (search_policy_orders != EOS_IDX).any(-1).any(-1)
    probs = search_policy_probs[has_actions]
    return -(probs * torch.log(probs + 1e-8)).sum(-1).mean()


def evs_to_policy(search_policy_evs, *, temperature=1.0, use_softmax=True):
    """Compute policy targets from EVs.

    Args:
        search_policy_evs: Float tensor [T, B, 7, max_actions]. Invalid values are marked with -1.
        temperature: temperature for softmax. Ignored if softmax is not used.
        use_softmax: whether to apply exp() before normalizing.

    Returns:
       search_policy_probs: Float tensor [T, B, 7, max_actions].
    """
    search_policy_probs = search_policy_evs.clone().float()
    invalid_mask = search_policy_evs < -0.5
    if use_softmax:
        search_policy_probs /= temperature
        # Using -1e8 instead of -inf so that softmax is defined even if all
        # orders are masked out.
        search_policy_probs[invalid_mask] = -1e8
        search_policy_probs = search_policy_probs.softmax(-1)
    else:
        search_policy_probs.masked_fill_(invalid_mask, 0.0)
        search_policy_probs /= (invalid_mask + 1e-20).sum(-1, keepdim=True)
    return search_policy_probs.to(search_policy_evs.dtype)


def perform_retry_loop(f: Callable, max_tries: int, sleep_seconds: int):
    """Retries f repeatedly if it raises RuntimeError or ValueError.

    Stops after success, or max_tries, sleeping sleep_seconds each failure."""
    tries = 0
    success = False
    while not success:
        tries += 1
        try:
            f()
            success = True
        except (RuntimeError, ValueError) as e:
            if tries >= max_tries:
                raise
            time.sleep(sleep_seconds)
