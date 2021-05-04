# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import List

from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import get_valid_orders_impl, DataFields, MAX_VALID_LEN
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOGIT_MASK_VAL
from fairdiplomacy.models.diplomacy_model.load_model import load_diplomacy_model
from fairdiplomacy.models.diplomacy_model.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelSampledAgent(BaseAgent):
    def __init__(self, model_path, temperature, top_p=1.0, device=_DEFAULT_DEVICE):
        self.model = load_diplomacy_model(model_path, map_location=device, eval=True)
        self.temperature = temperature
        self.device = device
        self.top_p = top_p
        self.thread_pool = pydipcc.ThreadPool(
            1, ORDER_VOCABULARY_TO_IDX, get_order_vocabulary_idxs_len()
        )

    def get_orders(self, game, power, *, temperature=None, top_p=None):
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return []

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        inputs = FeatureEncoder().encode_inputs([game])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            order_idxs, cand_idxs, logits, final_scores = self.model(
                **inputs, temperature=temperature, top_p=top_p
            )

        resample_duplicate_disbands_inplace(
            order_idxs, cand_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
        )
        return decode_order_idxs(order_idxs[0, POWERS.index(power), :])

    def get_orders_many_powers(self, game, powers, *, temperature=None, top_p=None):

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        inputs = FeatureEncoder().encode_inputs([game])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            order_idxs, cand_idxs, logits, final_scores = self.model(
                **inputs, temperature=temperature, top_p=top_p
            )

        resample_duplicate_disbands_inplace(
            order_idxs, cand_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
        )
        return {
            power: decode_order_idxs(order_idxs[0, POWERS.index(power), :]) for power in powers
        }


def decode_order_idxs(order_idxs) -> List[str]:
    orders = []
    for idx in order_idxs:
        if idx == EOS_IDX:
            continue
        orders.extend(ORDER_VOCABULARY[idx].split(";"))
    return orders


def resample_duplicate_disbands_inplace(
    order_idxs, sampled_idxs, logits, x_possible_actions, x_in_adj_phase
):
    """Modify order_idxs and sampled_idxs in-place, resampling where there are
    multiple disband orders.
    """
    # Resample all multiple disbands. Since builds are a 1-step decode, any 2+
    # step adj-phase is a disband.
    if sampled_idxs.shape[2] < 2:
        return
    multi_disband_powers_mask = (sampled_idxs[:, :, 1] != -1) & x_in_adj_phase.bool().unsqueeze(1)
    if not multi_disband_powers_mask.any():
        return

    # N.B. we may sample more orders than we need here: we are sampling
    # according to the longest sequence in the batch, not the longest
    # multi-disband sequence. Still, even if there is a 3-unit disband and a
    # 2-unit disband, we will sample the same # for both and mask out the extra
    # orders (see the eos_mask below)
    #
    # Note 1: The longest sequence in the batch may be longer than
    # the # of disband order candidates, so we take the min with
    # logits.shape[3] (#candidates)
    #
    # Note 2: 1e-15 is Laplace smoothing coefficient to make sampling without
    # replacing work for spiky logits which result in 1-hot probs
    #
    # Note 3: Ensure that masked (large negative) logits don't get mixed up
    # with low-prob (but valid) actions due to the smoothing coefficient and
    # the limits of float math.
    try:
        saved_logits_mask = logits[multi_disband_powers_mask][:, 0] == LOGIT_MASK_VAL
        probs = logits[multi_disband_powers_mask][:, 0].softmax(-1) + 1e-12  # See Note 2
        probs[saved_logits_mask] = 1e-18  # See Note 3
        new_sampled_idxs = torch.multinomial(
            probs, min(logits.shape[2], logits.shape[3]), replacement=False  # See Note 1
        )
    except RuntimeError:
        torch.save(
            {
                "order_idxs": order_idxs,
                "sampled_idxs": sampled_idxs,
                "logits": logits,
                "x_possible_actions": x_possible_actions,
                "x_in_adj_phase": x_in_adj_phase,
            },
            "resample_duplicate_disbands_inplace.debug.pt",
        )
        raise

    filler = torch.empty(
        new_sampled_idxs.shape[0],
        sampled_idxs.shape[2] - new_sampled_idxs.shape[1],
        dtype=new_sampled_idxs.dtype,
        device=new_sampled_idxs.device,
    ).fill_(-1)
    eos_mask = sampled_idxs == EOS_IDX
    sampled_idxs[multi_disband_powers_mask] = torch.cat([new_sampled_idxs, filler], dim=1)
    order_idxs[multi_disband_powers_mask] = torch.cat(
        [
            x_possible_actions[multi_disband_powers_mask][:, 0].long().gather(1, new_sampled_idxs),
            filler,
        ],
        dim=1,
    )
    sampled_idxs[eos_mask] = EOS_IDX
    order_idxs[eos_mask] = EOS_IDX
    # copy step-0 logits to other steps, since they were the logits used above
    # to sample all steps
    logits[multi_disband_powers_mask] = logits[multi_disband_powers_mask][:, 0].unsqueeze(1)


def get_valid_orders(game, power, *, all_possible_orders=None, all_orderable_locations=None):
    """Return indices of valid orders

    Returns:
    - a [1, 17, 469] int tensor of valid move indexes (padded with -1)
    - a [1, 81] int8 tensor of orderable locs, described below
    - the actual length of the sequence == the number of orders to submit, <= 17
    """
    if all_possible_orders is None:
        all_possible_orders = game.get_all_possible_orders()
    if all_orderable_locations is None:
        all_orderable_locations = game.get_orderable_locations()

    return get_valid_orders_impl(
        power, all_possible_orders, all_orderable_locations, game.get_state()
    )
