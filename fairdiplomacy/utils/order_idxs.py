# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpful functions for converting:
- order idxs to/from order strings
- global idxs to/from local idxs
"""
import torch
from typing import List, Sequence

from fairdiplomacy.models.consts import LOCS
from fairdiplomacy.models.diplomacy_model.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}
MAX_VALID_LEN = get_order_vocabulary_idxs_len()


##########################
##  STRING CONVERSIONS  ##
##########################


def global_order_idxs_to_str(order_idxs) -> List[str]:
    """Convert a sequence of global order idxs to a list of order strings

    N.B. returns normal uncombined build orders that are readable by a Game
    object.
    """
    orders = []
    for idx in order_idxs:
        if idx == EOS_IDX:
            continue
        orders.extend(ORDER_VOCABULARY[idx].split(";"))
    return orders


def action_strs_to_global_idxs(
    orders: Sequence[str],
    try_strip_coasts: bool = False,
    ignore_missing: bool = False,
    sort_by_loc: bool = False,
    sort_by_idx: bool = False,
) -> List[int]:
    """Convert a list of order strings to a list of global idxs in ORDER_VOCABULARY

    Args:
    - orders: a list of combined or uncombined order strings
    - try_strip_coasts: if True and order is not in vocabulary, strip coasts from all
        locs and try again
    - ignore_missing: if True, return idxs only for orders found in vocab. If False and
        and an order is not found, raise KeyError
    - sort_by_loc: if True, return order idxs sorted by the actor's location
    - sort_by_idx: if True, return sorted order idxs

    N.B. accepts combined or uncombined order strings
    """
    if type(orders) == str:
        raise ValueError("orders must be a sequence of strings")
    if sort_by_loc and sort_by_idx:
        raise ValueError("can't set both sort_by_loc and sort_by_idx")

    # combine build orders if necessary
    if any(x.split()[2] == "B" and ";" not in x for x in orders):
        assert all(
            x.split()[2] == "B" for x in orders
        ), "Cannot combine a mix of build and non-build orders"
        orders = [";".join(sorted(orders))]

    order_idxs = []
    order_loc_idxs = []
    for order in orders:
        order_idx = ORDER_VOCABULARY_TO_IDX.get(order, None)
        if order_idx is None and try_strip_coasts:
            order_idx = ORDER_VOCABULARY_TO_IDX.get(strip_coasts(order), None)
        if order_idx is None:
            if ignore_missing:
                continue
            else:
                raise KeyError
        order_idxs.append(order_idx)
        order_loc_idxs.append(LOCS.index(order.split()[1]))

    if sort_by_loc:
        order_idxs = [idx for loc, idx in sorted(zip(order_loc_idxs, order_idxs))]
    if sort_by_idx:
        order_idxs.sort()

    return order_idxs


################################
##  GLOBAL/LOCAL CONVERSIONS  ##
################################


def local_order_idxs_to_global(
    local_idxs: torch.Tensor, x_possible_actions: torch.Tensor, clamp_and_mask: bool = True
) -> torch.Tensor:
    """Convert local order indices to global order indices.

    Args:
        local_indices: Long tensor [*, 7, S] of indices [0,469) of x_possible_actions
        x_possible_actions: Long tensor [*, 7, S, 469]. Containing indices in
            ORDER_VOCABULARY.
        clamp_and_mask: if True, handle EOS_IDX inputs and propagate them to outputs. Set
            False to skip as a speed optimization if not necessary.

    Returns:
        global_indices: Long tensor of the same shape as local_indices such that
            x_possible_actions[b, p, s, local_indices[b, p, s]] = global_indices[b, p, s]
    """
    assert (
        EOS_IDX == -1
    ), "the clamp_and_mask path is necessary because EOS_IDX is negative. Is that still true?"

    if clamp_and_mask:
        mask = local_idxs == EOS_IDX
        local_idxs = local_idxs.clamp(0)

    global_idxs = torch.gather(
        x_possible_actions, local_idxs.ndim, local_idxs.unsqueeze(-1)
    ).view_as(local_idxs)

    if clamp_and_mask:
        global_idxs[mask] = EOS_IDX

    return global_idxs


def global_order_idxs_to_local(
    global_indices: torch.Tensor, x_possible_actions: torch.Tensor
) -> torch.Tensor:
    """Convert global order indices to local order indices.

    Args:
        global_indices: Long tensor [B, 7, S] of indices in ORDER_VOCABULARY
        x_possible_actions: Long tensor [B, 7, S, 469]. Containing indices in
            ORDER_VOCABULARY.

    Returns:
        local_indices: Long tensor of the same shape as global_indices such that
            x_possible_actions[b, p, s, local_indices[b, p, s]] = global_indices[b, p, s]
    """
    onehots = x_possible_actions == global_indices.unsqueeze(-1)
    local_indices = onehots.max(-1).indices
    local_indices[global_indices == EOS_IDX] = EOS_IDX
    return local_indices


############
##  MISC  ##
############


def strip_coasts(order: str) -> str:
    """Return order with all locations stripped of their coast suffixes"""
    for suffix in ["/NC", "/EC", "/SC", "/WC"]:
        order = order.strip(suffix)
    return order
