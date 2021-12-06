# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union
import logging
import numpy as np
import torch

from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.models.consts import LOGIT_MASK_VAL, N_SCS
from fairdiplomacy.models.diplomacy_model.load_model import load_diplomacy_model_model, load_diplomacy_model_model_cached
from fairdiplomacy.models.diplomacy_model.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.batching import batched_forward
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.timing_ctx import DummyCtx

Action = Tuple[str, ...]


class ModelWrapper:
    """Provides an easier interface for model inference"""

    def __init__(
        self,
        model_path,
        device="cuda",
        value_model_path=None,
        max_batch_size=int(1e10),
        *,
        half_precision=False,
        skip_model_cache=False,
    ):
        if not torch.cuda.is_available():
            logging.warning("Using cpu because cuda not available")
            device = "cpu"

        def load_model(path):
            if skip_model_cache:
                return load_diplomacy_model_model(checkpoint_path=path, map_location=device, eval=True)
            else:
                return load_diplomacy_model_model_cached(checkpoint_path=path, map_location=device)

        self.model = load_model(model_path)
        self.value_model = load_model(value_model_path) if value_model_path else self.model
        self.device = device
        self.order_encoder = FeatureEncoder()
        self.max_batch_size = max_batch_size
        self.half_precision = half_precision
        if half_precision:
            self.model.half()
            if self.value_model is not self.model:
                self.value_model.half()

    def do_model_request(
        self,
        x: DataFields,
        temperature: float = -1,
        top_p: float = -1,
        values_only: bool = False,
        timings=None,
        batch_repeat_interleave: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[List[List[Action]], torch.Tensor, None]]:
        """Handle max_batch_size constraint here.

        Note, if values_only is True, only values are computed.
        If false, only the policy.
        """
        if timings is None:
            timings = DummyCtx()

        with timings("to_half_precision"):
            if self.half_precision:
                x = x.to_half_precision()

        if values_only:
            assert (
                batch_repeat_interleave is None
            ), "batch_repeat_interleave is pointless with values_only"
            with timings("model"):
                return batched_forward(
                    self._forward_values, x, batch_size=self.max_batch_size, device=self.device
                )

        assert temperature > 0
        assert top_p > 0
        with timings("model"):
            order_idxs, order_logprobs = batched_forward(
                lambda x: self._forward_policy(
                    x,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                ),
                x,
                batch_size=self.max_batch_size,
                device=self.device,
            )

        with timings("model.decode"):
            decoded = self.order_encoder.decode_order_idxs(order_idxs)

        if self.model.all_powers:
            with timings("model.decode.all_powers"):
                x_in_adj_phase_batched = x["x_in_adj_phase"].cpu()
                x_power_batched = x["x_power"].cpu()
                # As documented in diplomacy_model.py, batch_repeat_interleave=N means we compute outputs as
                # if each input element was repeated N times, without explicitly repeating them.
                # So the output "decoded" has an N times larger batch size, but x_in_adj_phase_batched
                # and x_power_batched do not, since they were inputs. So when indexing them, we divide
                # to find out the proper index.
                div = 1 if batch_repeat_interleave is None else div
                assert len(x_in_adj_phase_batched) * div == len(decoded)
                assert len(x_power_batched) * div == len(decoded)
                for (i, powers_orders) in enumerate(decoded):
                    x_in_adj_phase = x_in_adj_phase_batched[i // div]
                    x_power = x_power_batched[i // div]
                    assert len(powers_orders) == 7
                    if x_in_adj_phase:
                        continue
                    assert all(len(orders) == 0 for orders in powers_orders[1:])
                    all_orders = powers_orders[0]  # all powers' orders
                    powers_orders[0] = []
                    for power_idx, order in zip(x_power[0], all_orders):
                        if power_idx == -1:
                            break
                        powers_orders[power_idx].append(order)

        with timings("model.decode.tuple"):
            # NOTE: use of tuples here is a bit inconsistent with what cfr1p
            # agent does, as well as BaseAgent interface, which expect lists
            # instead.
            decoded = [[tuple(orders) for orders in powers_orders] for powers_orders in decoded]

        # Returning None for values. Must call with values_only to get values.
        return (decoded, order_logprobs, None)

    def _forward_values(self, x: DataFields):
        x = x.copy()
        x.update(x_loc_idxs=None, x_possible_actions=None, temperature=None, top_p=None)
        _, _, _, values = self.value_model(**x, need_policy=False)
        return values

    def _forward_policy(
        self,
        x: DataFields,
        temperature: float,
        top_p: float,
        batch_repeat_interleave: Optional[int],
    ):
        x = x.copy()
        x["temperature"] = temperature
        x["top_p"] = top_p
        x["batch_repeat_interleave"] = batch_repeat_interleave
        y = self.model(**x, need_value=False, pad_to_max=True)
        order_idxs, order_logprobs, _ = model_output_transform(
            x, y, batch_repeat_interleave=batch_repeat_interleave
        )
        return order_idxs, order_logprobs

    def get_values(self, game) -> np.ndarray:
        batch_inputs = self.order_encoder.encode_inputs_state_only([game])
        batch_est_final_scores = self.do_model_request(batch_inputs, values_only=True)
        return batch_est_final_scores[0]

    def is_all_powers(self) -> bool:
        return self.model.all_powers


def model_output_transform(x, y, *, batch_repeat_interleave=None):
    global_order_idxs, local_order_idxs, logits, final_sos = y
    resample_duplicate_disbands_inplace(
        global_order_idxs,
        local_order_idxs,
        logits,
        x["x_possible_actions"],
        x["x_in_adj_phase"],
        batch_repeat_interleave=batch_repeat_interleave,
    )
    return global_order_idxs, compute_action_logprobs(local_order_idxs, logits), final_sos


def compute_action_logprobs(local_order_idxs, logits):
    local_order_idxs = local_order_idxs[:, :, : logits.shape[-2]]  # trim off excess seq dim
    invalid_mask = local_order_idxs < 0
    local_order_idxs = local_order_idxs.clamp(
        min=0
    )  # otherwise gather(-1) will blow up. We'll mask these out later
    logprobs = logits.log_softmax(-1)
    sampled_logprobs = logprobs.gather(-1, local_order_idxs.unsqueeze(-1)).squeeze(-1)
    sampled_logprobs[invalid_mask] = 0
    total_logprobs = sampled_logprobs.sum(-1)
    return total_logprobs


def resample_duplicate_disbands_inplace(
    global_order_idxs,
    local_order_idxs,
    logits,
    x_possible_actions,
    x_in_adj_phase,
    *,
    batch_repeat_interleave=None,
):
    """Modify global_order_idxs and local_order_idxs in-place, resampling where there are
    multiple disband orders.
    """
    # Resample all multiple disbands. Since builds are a 1-step decode, any 2+
    # step adj-phase is a disband.
    if local_order_idxs.shape[2] < 2:
        return
    if batch_repeat_interleave is not None:
        assert x_in_adj_phase.size()[0] * batch_repeat_interleave == local_order_idxs.size()[0]
        x_in_adj_phase = x_in_adj_phase.repeat_interleave(batch_repeat_interleave, dim=0)
    multi_disband_powers_mask = (
        local_order_idxs[:, :, 1] != -1
    ) & x_in_adj_phase.bool().unsqueeze(1)
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
        new_local_order_idxs = torch.multinomial(
            probs, min(logits.shape[2], logits.shape[3]), replacement=False  # See Note 1
        )
    except RuntimeError:
        torch.save(
            {
                "global_order_idxs": global_order_idxs,
                "local_order_idxs": local_order_idxs,
                "logits": logits,
                "x_possible_actions": x_possible_actions,
                "x_in_adj_phase": x_in_adj_phase,
                "batch_repeat_interleave": batch_repeat_interleave,
            },
            "resample_duplicate_disbands_inplace.debug.pt",
        )
        raise

    filler = torch.empty(
        new_local_order_idxs.shape[0],
        local_order_idxs.shape[2] - new_local_order_idxs.shape[1],
        dtype=new_local_order_idxs.dtype,
        device=new_local_order_idxs.device,
    ).fill_(-1)
    eos_mask = local_order_idxs == EOS_IDX
    local_order_idxs[multi_disband_powers_mask] = torch.cat([new_local_order_idxs, filler], dim=1)

    if batch_repeat_interleave is not None:
        assert x_possible_actions.size()[0] * batch_repeat_interleave == local_order_idxs.size()[0]
        x_possible_actions = x_possible_actions.repeat_interleave(batch_repeat_interleave, dim=0)
    global_order_idxs[multi_disband_powers_mask] = torch.cat(
        [
            x_possible_actions[multi_disband_powers_mask][:, 0]
            .long()
            .gather(1, new_local_order_idxs),
            filler,
        ],
        dim=1,
    )
    local_order_idxs[eos_mask] = EOS_IDX
    global_order_idxs[eos_mask] = EOS_IDX
    # copy step-0 logits to other steps, since they were the logits used above
    # to sample all steps
    logits[multi_disband_powers_mask] = logits[multi_disband_powers_mask][:, 0].unsqueeze(1)
