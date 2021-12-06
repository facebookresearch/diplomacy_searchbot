# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional, Union
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from fairdiplomacy.models.consts import POWERS, LOCS, LOGIT_MASK_VAL, MAX_SEQ_LEN, N_SCS
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.padded_embedding import PaddedEmbedding
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global
from fairdiplomacy.models.diplomacy_model.order_vocabulary import get_order_vocabulary, EOS_IDX

EOS_TOKEN = get_order_vocabulary()[EOS_IDX]
# If teacher forcing orders have this id, then a sampled order will be used for
# this position.
NO_ORDER_ID = -2


class DiplomacyModel(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # 35
        # prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 60
        season_emb_size,  # 20,
        num_blocks,  # 16
        A,  # 81x81
        master_alignments,
        orders_vocab_size,  # 13k
        lstm_size,  # 200
        order_emb_size,  # 80
        prev_order_emb_size,  # 20
        lstm_dropout=0,
        lstm_layers=1,
        encoder_dropout=0,
        value_dropout,
        learnable_A=False,
        learnable_alignments=False,
        use_simple_alignments=False,
        avg_embedding=False,
        value_decoder_init_scale=1.0,
        featurize_output=False,
        relfeat_output=False,
        featurize_prev_orders=False,
        residual_linear=False,
        merged_gnn=False,
        encoder_layerdrop=0,
        value_softmax=False,
        separate_value_encoder=False,
        use_global_pooling=False,
        encoder_cfg=None,
        pad_spatial_size_to_multiple=1,
        all_powers,
        has_policy=True,
        has_value=True,
    ):
        super().__init__()
        self.orders_vocab_size = orders_vocab_size

        self.featurize_prev_orders = featurize_prev_orders
        self.prev_order_enc_size = prev_order_emb_size
        if has_policy and featurize_prev_orders:
            order_feats, _srcs, _dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            self.prev_order_enc_size += self.order_feats.shape[-1]

        self.separate_value_encoder = separate_value_encoder
        self.value_encoder = None

        self.has_policy = has_policy
        self.has_value = has_value

        self.spatial_size = A.size()[0]

        encoder_kind = encoder_cfg.WhichOneof("encoder")
        if encoder_kind == "transformer":
            if pad_spatial_size_to_multiple > 1:
                self.spatial_size = (
                    (self.spatial_size + pad_spatial_size_to_multiple - 1)
                    // pad_spatial_size_to_multiple
                    * pad_spatial_size_to_multiple
                )
            encoder_cfg = getattr(encoder_cfg, encoder_kind)
            encoder_kwargs = dict(
                board_state_size=board_state_size + len(POWERS) + season_emb_size + 1,
                prev_orders_size=board_state_size
                + self.prev_order_enc_size
                + len(POWERS)
                + season_emb_size
                + 1,
                spatial_size=self.spatial_size,
                inter_emb_size=inter_emb_size,
                encoder_cfg=encoder_cfg,
            )
            if has_policy or has_value and not separate_value_encoder:
                self.encoder = TransformerEncoder(**encoder_kwargs)
            if has_value and separate_value_encoder:
                self.value_encoder = TransformerEncoder(**encoder_kwargs)
        elif encoder_kind is None:  # None == graph encoder
            if pad_spatial_size_to_multiple > 1:
                raise ValueError(
                    "pad_spatial_size_to_multiple > 1 not supported for graph conv encoder"
                )
            encoder_kwargs = dict(
                board_state_size=board_state_size + len(POWERS) + season_emb_size + 1,
                prev_orders_size=board_state_size
                + self.prev_order_enc_size
                + len(POWERS)
                + season_emb_size
                + 1,
                inter_emb_size=inter_emb_size,
                num_blocks=num_blocks,
                A=A,
                dropout=encoder_dropout,
                learnable_A=learnable_A,
                residual_linear=residual_linear,
                merged_gnn=merged_gnn,
                layerdrop=encoder_layerdrop,
                use_global_pooling=use_global_pooling,
            )
            if has_policy or has_value and not separate_value_encoder:
                self.encoder = DiplomacyModelEncoder(**encoder_kwargs)
            if has_value and separate_value_encoder:
                self.value_encoder = DiplomacyModelEncoder(**encoder_kwargs)
        else:
            assert False

        if has_policy:
            self.policy_decoder = LSTMDiplomacyModelDecoder(
                inter_emb_size=inter_emb_size,
                spatial_size=self.spatial_size,
                orders_vocab_size=orders_vocab_size,
                lstm_size=lstm_size,
                order_emb_size=order_emb_size,
                lstm_dropout=lstm_dropout,
                lstm_layers=lstm_layers,
                master_alignments=master_alignments,
                learnable_alignments=learnable_alignments,
                use_simple_alignments=use_simple_alignments,
                avg_embedding=avg_embedding,
                power_emb_size=power_emb_size,
                featurize_output=featurize_output,
                relfeat_output=relfeat_output,
            )

        if has_value:
            self.value_decoder = ValueDecoder(
                inter_emb_size=inter_emb_size,
                spatial_size=self.spatial_size,
                init_scale=value_decoder_init_scale,
                dropout=value_dropout,
                softmax=value_softmax,
            )

        # These are used by both value and policy, regardless of separate_value_encoder
        self.season_lin = nn.Linear(3, season_emb_size)
        self.prev_order_embedding = nn.Embedding(
            orders_vocab_size, prev_order_emb_size, padding_idx=0
        )

        self.all_powers = all_powers

    def forward(
        self,
        *,
        x_board_state,
        x_prev_state,
        x_prev_orders,
        x_season,
        x_in_adj_phase,
        x_build_numbers,
        x_loc_idxs,
        x_possible_actions,
        temperature,
        top_p=1.0,
        batch_repeat_interleave=None,
        teacher_force_orders=None,
        x_power=None,
        need_policy=True,
        need_value=True,
        pad_to_max=False,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        TODO(alerer): fix the docs.
        Arguments:
        - x_bo: [B, 81, 35]
        - x_pb: [B, 2, 100], long
        - x_po: [B, 81, 40]
        - x_season_1h: [B, 3]
        - in_adj_phase: [B], bool
        - x_build_numbers: [B, 7]
        - loc_idxs: int8, [B, 81] or [B, 7, 81]
        - all_cand_idxs: long, [B, S, 469] or [B, 7, S, 469]
        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of [B, 1]
        - top_p: probability mass to samples from, lower = more spiky; must
          be either a float or a tensor of [B, 1]
        - batch_repeat_interleave: if set to a value k, will behave as if [B] dimension was
            was actually [B*k] in size, with each element repeated k times
            (e.g. [1,2,3] k=2 -> [1,1,2,2,3,3]), on all tensors EXCEPT teacher_force_orders
        - teacher_force_orders: [B, S] or [B, num_samples, S] long or None,
            ORDER idxs, NOT candidate idxs, 0-padded. If batch_repeat_interleave is None,
            then the first form expected. Otherwise, the shape is expected
            with num_samples == batch_repeat_interleave.
        - x_power: [B, S] long, [B, 7, S] long, or None
        - need_policy: if not set, global_order_idxs, local_order_idxs, and logits will be None.
        - need_value: if not set, final_sos in Result will be None
        - pad_to_max, if set, will pad all output tensors to [..., MAX_SEQ_LEN, 469]. Use that
            to make torch.nn.DataPatallel to work.

        if x_power is None or [B, 7, 34] Long, the model will decode for all 7 powers.
            - loc_idxs, all_cand_idxs, and teacher_force_orders must have an
              extra axis at dim=1 with size 7
            - global_order_idxs and order_scores will be returned with an extra axis
              at dim=1 with size 7
            - if x_power is [B, 7, 34] Long, non-A phases are expected to be encoded in [:,0,:]
        else x_power must be [B, S] Long and only one power's sequence will be decoded

        Returns:
          - global_order_idxs [B, S] or [B, 7, S]: idx in ORDER_VOCABULARY of sampled
            orders for each power
          - local_order_idxs [B, S] or [B, 7, S]: idx in all_cand_idxs of sampled
            orders for each power
          - logits [B, S, C] or [B, 7, S, C]: masked pre-softmax logits of each
            candidate order, 0 < S <= 17, 0 < C <= 469
          - final_sos [B, 7]: estimated sum of squares share for each power
        """

        # following https://arxiv.org/pdf/2006.04635.pdf , Appendix C
        B, NUM_LOCS, _ = x_board_state.shape

        # Preemptively make sure that dtypes of things match, to try to limit the chance of bugs
        # if the inputs were built in an ad-hoc way when are trying to run in fp16.
        assert x_board_state.dtype == x_prev_state.dtype
        assert x_board_state.dtype == x_build_numbers.dtype
        assert x_board_state.dtype == x_season.dtype

        assert not (need_policy and not self.has_policy)
        assert not (need_value and not self.has_value)

        assert need_policy or need_value

        # A. get season and prev order embs
        x_season_emb = self.season_lin(x_season)

        x_prev_order_emb = self.prev_order_embedding(x_prev_orders[:, 0])
        if self.featurize_prev_orders:
            x_prev_order_emb = torch.cat(
                (x_prev_order_emb, self.order_feats[x_prev_orders[:, 0]]), dim=-1
            )

        # B. insert the prev orders into the correct board location (which is in the second column of x_po)
        x_prev_order_exp = x_board_state.new_zeros(B, NUM_LOCS, self.prev_order_enc_size)
        prev_order_loc_idxs = torch.arange(B, device=x_board_state.device).repeat_interleave(
            x_prev_orders.shape[-1]
        ) * NUM_LOCS + x_prev_orders[:, 1].reshape(-1)
        x_prev_order_exp.view(-1, self.prev_order_enc_size).index_add_(
            0, prev_order_loc_idxs, x_prev_order_emb.view(-1, self.prev_order_enc_size)
        )

        # concatenate the subcomponents into board state and prev state, following the paper
        x_build_numbers_exp = x_build_numbers[:, None].expand(-1, NUM_LOCS, -1)
        x_season_emb_exp = x_season_emb[:, None].expand(-1, NUM_LOCS, -1)
        vestigial_zeros = torch.zeros((B, NUM_LOCS, 1), device=x_board_state.device)
        x_bo_hat = torch.cat(
            (x_board_state, x_build_numbers_exp, x_season_emb_exp, vestigial_zeros), dim=-1
        )
        x_po_hat = torch.cat(
            (
                x_prev_state,
                x_prev_order_exp,
                x_build_numbers_exp,
                x_season_emb_exp,
                vestigial_zeros,
            ),
            dim=-1,
        )

        assert x_bo_hat.size()[1] == x_po_hat.size()[1]
        if self.spatial_size != x_bo_hat.size()[1]:
            # pad (batch, 81, channels) -> (batch, spatial_size, channels)
            assert self.spatial_size > x_bo_hat.size()[1]
            assert len(x_bo_hat.size()) == 3
            x_bo_hat = F.pad(x_bo_hat, (0, 0, 0, self.spatial_size - x_bo_hat.size()[1]))
        if self.spatial_size != x_po_hat.size()[1]:
            # pad (batch, 81, channels) -> (batch, spatial_size, channels)
            assert self.spatial_size > x_po_hat.size()[1]
            assert len(x_po_hat.size()) == 3
            x_po_hat = F.pad(x_po_hat, (0, 0, 0, self.spatial_size - x_po_hat.size()[1]))

        if need_policy:
            encoded_for_policy = self.encoder(x_bo_hat, x_po_hat)
        else:
            encoded_for_policy = None

        if need_value:
            if self.separate_value_encoder:
                encoded_for_value = self.value_encoder(x_bo_hat, x_po_hat)
            elif encoded_for_policy is not None:
                encoded_for_value = encoded_for_policy
            else:
                encoded_for_value = self.encoder(x_bo_hat, x_po_hat)
        else:
            encoded_for_value = None

        if encoded_for_value is not None:
            final_sos = self.value_decoder(encoded_for_value)
            if batch_repeat_interleave is not None:
                final_sos = torch.repeat_interleave(final_sos, batch_repeat_interleave, dim=0)
        else:
            final_sos = None

        all_powers = x_power is not None and len(x_power.shape) == 3

        if not need_policy:
            global_order_idxs = local_order_idxs = logits = None
        elif x_power is None or all_powers:
            global_order_idxs, local_order_idxs, logits = self.forward_all_powers(
                enc=encoded_for_policy,
                in_adj_phase=x_in_adj_phase,
                loc_idxs=x_loc_idxs,
                cand_idxs=x_possible_actions,
                temperature=temperature,
                top_p=top_p,
                batch_repeat_interleave=batch_repeat_interleave,
                teacher_force_orders=teacher_force_orders,
                power=x_power,
            )
        else:
            global_order_idxs, local_order_idxs, logits = self.forward_one_power(
                enc=encoded_for_policy,
                in_adj_phase=x_in_adj_phase,
                loc_idxs=x_loc_idxs,
                cand_idxs=x_possible_actions,
                temperature=temperature,
                top_p=top_p,
                batch_repeat_interleave=batch_repeat_interleave,
                teacher_force_orders=teacher_force_orders,
                power=x_power,
            )
        if pad_to_max and need_policy:
            max_seq_len = N_SCS if all_powers else MAX_SEQ_LEN
            global_order_idxs = _pad_last_dims(global_order_idxs, [max_seq_len], EOS_IDX)
            local_order_idxs = _pad_last_dims(local_order_idxs, [max_seq_len], EOS_IDX)
            logits = _pad_last_dims(logits, [max_seq_len, 469], LOGIT_MASK_VAL)
        return global_order_idxs, local_order_idxs, logits, final_sos

    def forward_one_power(
        self,
        *,
        enc,
        in_adj_phase,
        loc_idxs,
        cand_idxs,
        power,
        temperature,
        top_p,
        batch_repeat_interleave,
        teacher_force_orders,
    ):
        assert len(loc_idxs.shape) == 2, loc_idxs.shape
        assert len(cand_idxs.shape) == 3, cand_idxs.shape

        if batch_repeat_interleave is not None:
            if teacher_force_orders is not None:
                assert (
                    teacher_force_orders.shape[1] == batch_repeat_interleave
                ), teacher_force_orders.shape
                teacher_force_orders = teacher_force_orders.view(
                    -1, *teacher_force_orders.shape[2:]
                )
            (
                enc,
                in_adj_phase,
                loc_idxs,
                cand_idxs,
                power,
                temperature,
                top_p,
            ) = apply_batch_repeat_interleave(
                (
                    enc,
                    in_adj_phase,
                    loc_idxs,
                    cand_idxs,
                    power,
                    temperature,
                    top_p,
                ),
                batch_repeat_interleave,
            )

        global_order_idxs, local_order_idxs, logits = self.policy_decoder(
            enc,
            in_adj_phase,
            loc_idxs,
            cand_idxs,
            temperature=temperature,
            top_p=top_p,
            teacher_force_orders=teacher_force_orders,
            power=power,
        )

        return global_order_idxs, local_order_idxs, logits

    def forward_all_powers(
        self,
        *,
        enc,
        in_adj_phase,
        loc_idxs,
        cand_idxs,
        temperature,
        teacher_force_orders,
        top_p,
        batch_repeat_interleave,
        log_timings=False,
        power=None,
    ):
        timings = TimingCtx()

        assert len(loc_idxs.shape) == 3
        assert len(cand_idxs.shape) == 4

        with timings("policy_decoder_prep"):
            if batch_repeat_interleave is not None:
                if teacher_force_orders is not None:
                    assert (
                        teacher_force_orders.shape[1] == batch_repeat_interleave
                    ), teacher_force_orders.shape
                    teacher_force_orders = teacher_force_orders.view(
                        -1, *teacher_force_orders.shape[2:]
                    )

                (
                    enc,
                    in_adj_phase,
                    loc_idxs,
                    cand_idxs,
                    power,
                    temperature,
                    top_p,
                ) = apply_batch_repeat_interleave(
                    (
                        enc,
                        in_adj_phase,
                        loc_idxs,
                        cand_idxs,
                        power,
                        temperature,
                        top_p,
                    ),
                    batch_repeat_interleave,
                )

            NPOWERS = len(POWERS)
            enc_repeat = enc.repeat_interleave(NPOWERS, dim=0)
            in_adj_phase = in_adj_phase.repeat_interleave(NPOWERS, dim=0)
            loc_idxs = loc_idxs.view(-1, loc_idxs.shape[2])
            cand_idxs = cand_idxs.view(-1, *cand_idxs.shape[2:])
            temperature = repeat_interleave_if_tensor(temperature, NPOWERS, dim=0)
            top_p = repeat_interleave_if_tensor(top_p, NPOWERS, dim=0)
            teacher_force_orders = (
                teacher_force_orders.view(-1, *teacher_force_orders.shape[2:])
                if teacher_force_orders is not None
                else None
            )

            if power is None:
                # N.B. use repeat, not repeat_interleave, for power only. Each
                # batch is contiguous, and we want a sequence of power idxs for each batch
                power = (
                    torch.arange(NPOWERS, device=enc.device)
                    .view(-1, 1)
                    .repeat(enc.shape[0], cand_idxs.shape[1])
                )
            else:
                # This is all-powers encoding: validate shape and use power idxs from input
                assert len(power.shape) == 3, power.shape
                assert power.shape[1] == NPOWERS, power.shape
                assert power.shape[2] == N_SCS, power.shape
                power = power.view(-1, N_SCS)

        with timings("policy_decoder"):
            # [B, 17, 469] -> [B, 17].
            valid_mask = (cand_idxs != EOS_IDX).any(dim=-1)
            # [B, 17] -> [B].
            phase_has_orders = valid_mask.any(-1)

            def pack(maybe_tensor):
                if isinstance(maybe_tensor, torch.Tensor):
                    return maybe_tensor[phase_has_orders]
                return maybe_tensor

            def unpack(tensor, fill_value):
                B = len(phase_has_orders)
                new_tensor = tensor.new_full((B,) + tensor.shape[1:], fill_value)
                new_tensor[phase_has_orders] = tensor
                return new_tensor

            # FIXME(akhti): it is faster to do the packing at the same time as
            # we are doing repeating.
            enc_repeat = pack(enc_repeat)
            in_adj_phase = pack(in_adj_phase)
            loc_idxs = pack(loc_idxs)
            cand_idxs = pack(cand_idxs)
            power = pack(power)
            temperature = pack(temperature)
            top_p = pack(top_p)
            teacher_force_orders = pack(teacher_force_orders)

            global_order_idxs, local_order_idxs, logits = self.policy_decoder(
                enc_repeat,
                in_adj_phase,
                loc_idxs,
                cand_idxs,
                temperature=temperature,
                top_p=top_p,
                teacher_force_orders=teacher_force_orders,
                power=power,
            )
            global_order_idxs = unpack(global_order_idxs, EOS_IDX)
            local_order_idxs = unpack(local_order_idxs, EOS_IDX)
            logits = unpack(logits, LOGIT_MASK_VAL)

        with timings("finish"):
            # reshape
            valid_mask = valid_mask.view(-1, NPOWERS, *valid_mask.shape[1:])
            global_order_idxs = global_order_idxs.view(-1, NPOWERS, *global_order_idxs.shape[1:])
            local_order_idxs = local_order_idxs.view(-1, NPOWERS, *local_order_idxs.shape[1:])
            logits = logits.view(-1, NPOWERS, *logits.shape[1:])

            # mask out garbage outputs
            eos_fill = torch.empty_like(global_order_idxs, requires_grad=False).fill_(EOS_IDX)
            global_order_idxs = torch.where(valid_mask, global_order_idxs, eos_fill)
            local_order_idxs = torch.where(valid_mask, local_order_idxs, eos_fill)

        if log_timings:
            logging.debug(f"Timings[model, B={enc.shape[0]}]: {timings}")

        return global_order_idxs, local_order_idxs, logits


def compute_alignments(loc_idxs, step, A):
    alignments = torch.matmul(((loc_idxs == step) | (loc_idxs == -2)).to(A.dtype), A)
    alignments /= torch.sum(alignments, dim=1, keepdim=True) + 1e-5
    # alignments = torch.where(
    #     torch.isnan(alignments), torch.zeros_like(alignments), alignments
    # )

    return alignments


def repeat_interleave_if_tensor(x, reps, dim):
    if hasattr(x, "repeat_interleave"):
        return x.repeat_interleave(reps, dim=dim)
    return x


def apply_batch_repeat_interleave(tensors, batch_repeat_interleave):
    return tuple(
        repeat_interleave_if_tensor(tensor, batch_repeat_interleave, dim=0) for tensor in tensors
    )


class LSTMDiplomacyModelDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        spatial_size,
        orders_vocab_size,
        lstm_size,
        order_emb_size,
        lstm_dropout,
        lstm_layers,
        master_alignments,
        learnable_alignments=False,
        use_simple_alignments=False,
        avg_embedding=False,
        power_emb_size,
        featurize_output=False,
        relfeat_output=False,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.spatial_size = spatial_size
        self.order_emb_size = order_emb_size
        self.lstm_dropout = lstm_dropout
        self.avg_embedding = avg_embedding
        self.power_emb_size = power_emb_size

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.cand_embedding = PaddedEmbedding(orders_vocab_size, lstm_size, padding_idx=EOS_IDX)
        self.power_lin = nn.Linear(len(POWERS), power_emb_size)

        self.lstm = nn.LSTM(
            2 * inter_emb_size
            + order_emb_size
            + power_emb_size,
            lstm_size,
            batch_first=True,
            num_layers=self.lstm_layers,
        )

        assert not (
            use_simple_alignments and learnable_alignments
        ), "use_simple_alignments and learnable_alignments are incompatible"
        self.use_simple_alignments = use_simple_alignments

        # if avg_embedding is True, alignments are not used, and pytorch
        # `comp`lains about unused parameters, so only set self.master_alignments
        # when avg_embedding is False
        if not avg_embedding:
            self.master_alignments = nn.Parameter(master_alignments).requires_grad_(
                learnable_alignments
            )

        self.featurize_output = featurize_output
        if featurize_output:
            order_feats, srcs, dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            order_decoder_input_sz = self.order_feats.shape[1]
            self.order_feat_lin = nn.Linear(order_decoder_input_sz, order_emb_size)

            # this one has to stay as separate w, b
            # for backwards compatibility
            self.order_decoder_w = nn.Linear(order_decoder_input_sz, lstm_size)  # FIXME
            self.order_decoder_b = nn.Linear(order_decoder_input_sz, 1)

        self.relfeat_output = relfeat_output
        if relfeat_output:
            assert featurize_output, "Can't have relfeat_output without featurize_output (yet)"
            order_feats, srcs, dsts = compute_order_features()
            self.register_buffer("order_srcs", srcs)
            self.register_buffer("order_dsts", dsts)
            order_relfeat_input_sz = 2 * inter_emb_size

            self.order_relfeat_src_decoder_w = nn.Linear(order_relfeat_input_sz, lstm_size + 1)
            self.order_relfeat_dst_decoder_w = nn.Linear(order_relfeat_input_sz, lstm_size + 1)

            self.order_emb_relfeat_src_decoder_w = nn.Linear(order_emb_size, lstm_size + 1)
            self.order_emb_relfeat_dst_decoder_w = nn.Linear(order_emb_size, lstm_size + 1)

    def get_order_loc_feats(self, cand_order_locs, enc_w, out_w, enc_lin=None):
        B, L, D = enc_w.shape
        flat_order_locs = cand_order_locs.view(-1)
        valid = (flat_order_locs > 0).nonzero(as_tuple=False).squeeze(-1)
        # offsets of the order into the flattened enc_w tensor
        order_offsets = (
            cand_order_locs + torch.arange(B, device=cand_order_locs.device).view(B, 1) * L
        )
        valid_order_offsets = order_offsets.view(-1)[valid]
        valid_order_w = enc_w.view(-1, D)[valid_order_offsets]
        if enc_lin:
            valid_order_w = enc_lin(valid_order_w)
        out_w.view(-1, out_w.shape[-1]).index_add_(0, valid, valid_order_w)

    def forward(
        self,
        enc,
        in_adj_phase,
        loc_idxs,
        all_cand_idxs,
        power,
        temperature=1.0,
        top_p=1.0,
        teacher_force_orders=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timings = TimingCtx()
        with timings("dec.prep"):
            device = enc.device

            if (loc_idxs == -1).all():
                return (
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    enc.new_zeros(*all_cand_idxs.shape),
                )

            # embedding for the last decoded order
            order_emb = enc.new_zeros(enc.shape[0], self.order_emb_size)

            # power embeddings for each lstm step
            assert tuple(power.shape) == tuple(
                all_cand_idxs.shape[:2]
            ), f"{power.shape} != {all_cand_idxs.shape[:2]}"

            # clamp power to avoid -1 padding crashing one_hot
            power_1h = torch.nn.functional.one_hot(power.long().clamp(0), len(POWERS)).to(
                enc.dtype
            )
            all_power_embs = self.power_lin(power_1h)

            # return values: chosen order idxs, candidate idxs, and logits
            all_global_order_idxs = []
            all_local_order_idxs = []
            all_logits = []

            order_enc = enc.new_zeros(enc.shape[0], self.spatial_size, self.order_emb_size)

            self.lstm.flatten_parameters()
            hidden = (
                enc.new_zeros(self.lstm_layers, enc.shape[0], self.lstm_size),
                enc.new_zeros(self.lstm_layers, enc.shape[0], self.lstm_size),
            )

            # reuse same dropout weights for all steps
            dropout_in = (
                enc.new_zeros(
                    enc.shape[0],
                    1,
                    enc.shape[2]
                    + self.order_emb_size
                    + self.power_emb_size,
                )
                .bernoulli_(1 - self.lstm_dropout)
                .div_(1 - self.lstm_dropout)
                .requires_grad_(False)
            )
            dropout_out = (
                enc.new_zeros(enc.shape[0], 1, self.lstm_size)
                .bernoulli_(1 - self.lstm_dropout)
                .div_(1 - self.lstm_dropout)
                .requires_grad_(False)
            )

            # find max # of valid cand idxs per step
            max_cand_per_step = (all_cand_idxs != EOS_IDX).sum(dim=2).max(dim=0).values  # [S]

            if self.relfeat_output:
                src_relfeat_w = self.order_relfeat_src_decoder_w(enc)
                dst_relfeat_w = self.order_relfeat_dst_decoder_w(enc)

        for step in range(all_cand_idxs.shape[1]):
            with timings("dec.power_emb"):
                power_emb = all_power_embs[:, step]

            with timings("dec.loc_enc"):
                num_cands = max_cand_per_step[step]
                cand_idxs = all_cand_idxs[:, step, :num_cands].long().contiguous()

                if self.avg_embedding:
                    # no attention: average across loc embeddings
                    loc_enc = torch.mean(enc, dim=1)
                else:
                    if self.use_simple_alignments:
                        alignments = ((loc_idxs == step) | (loc_idxs == -2)).to(enc.dtype)
                    else:
                        # do static attention
                        alignments = compute_alignments(loc_idxs, step, self.master_alignments)

                    if self.spatial_size != alignments.size()[1]:
                        # pad (batch, 81) -> (batch, spatial_size)
                        assert self.spatial_size > alignments.size()[1]
                        assert len(alignments.size()) == 2
                        alignments = F.pad(
                            alignments, (0, self.spatial_size - alignments.size()[1])
                        )

                    # print('alignments', alignments.mean(), alignments.std())
                    loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            with timings("dec.lstm"):
                input_list = [loc_enc, order_emb, power_emb]

                lstm_input = torch.cat(input_list, dim=1).unsqueeze(1)
                if self.training and self.lstm_dropout > 0.0:
                    lstm_input = lstm_input * dropout_in

                out, hidden = self.lstm(lstm_input, hidden)
                if self.training and self.lstm_dropout > 0.0:
                    out = out * dropout_out

                out = out.squeeze(1).unsqueeze(2)

            with timings("dec.cand_emb"):
                cand_emb = self.cand_embedding(cand_idxs)

            with timings("dec.logits"):
                logits = torch.matmul(cand_emb, out).squeeze(2)  # [B, <=469]

                if self.featurize_output:
                    # a) featurize based on one-hot features
                    cand_order_feats = self.order_feats[cand_idxs]
                    order_w = torch.cat(
                        (
                            self.order_decoder_w(cand_order_feats),
                            self.order_decoder_b(cand_order_feats),
                        ),
                        dim=-1,
                    )

                    if self.relfeat_output:
                        cand_srcs = self.order_srcs[cand_idxs]
                        cand_dsts = self.order_dsts[cand_idxs]

                        # b) featurize based on the src and dst encoder features
                        self.get_order_loc_feats(cand_srcs, src_relfeat_w, order_w)
                        self.get_order_loc_feats(cand_dsts, dst_relfeat_w, order_w)

                        # c) featurize based on the src and dst order embeddings
                        self.get_order_loc_feats(
                            cand_srcs,
                            order_enc,
                            order_w,
                            enc_lin=self.order_emb_relfeat_src_decoder_w,
                        )
                        self.get_order_loc_feats(
                            cand_dsts,
                            order_enc,
                            order_w,
                            enc_lin=self.order_emb_relfeat_dst_decoder_w,
                        )

                    # add some ones to out so that the last element of order_w is a bias
                    out_with_ones = torch.cat((out, out.new_ones(out.shape[0], 1, 1)), dim=1)
                    order_scores_featurized = torch.bmm(order_w, out_with_ones)
                    logits += order_scores_featurized.squeeze(-1)

            with timings("dec.invalid_mask"):
                # unmask where there are no actions or the sampling will crash. The
                # losses at these points will be masked out later, so this is safe.
                invalid_mask = ~(cand_idxs != EOS_IDX).any(dim=1)
                if invalid_mask.all():
                    # early exit
                    # logging.debug(f"Breaking at step {step} because no more orders to give")
                    for _step in range(step, all_cand_idxs.shape[1]):  # fill in garbage
                        all_global_order_idxs.append(
                            torch.empty(
                                all_cand_idxs.shape[0],
                                dtype=torch.long,
                                device=all_cand_idxs.device,
                            ).fill_(EOS_IDX)
                        )
                        all_local_order_idxs.append(
                            torch.empty(
                                all_cand_idxs.shape[0],
                                dtype=torch.long,
                                device=all_cand_idxs.device,
                            ).fill_(EOS_IDX)
                        )
                    break

                cand_mask = cand_idxs != EOS_IDX
                cand_mask[invalid_mask] = 1

            with timings("dec.logits_mask"):
                # make logits for invalid actions a large negative
                # We also deliberately call float() here, not to(enc.dtype), because even in fp16
                # once we have logits we want to cast up to fp32 for doing the masking, temperature,
                # and softmax.
                logits = torch.min(logits, cand_mask.float() * 1e9 + LOGIT_MASK_VAL)
                all_logits.append(logits)

            with timings("dec.logits_temp_top_p"):
                with torch.no_grad():
                    filtered_logits = logits.detach().clone()
                    top_p_min = top_p.min().item() if isinstance(top_p, torch.Tensor) else top_p
                    if top_p_min < 0.999:
                        filtered_logits.masked_fill_(
                            top_p_filtering(filtered_logits, top_p=top_p), -1e9
                        )
                    filtered_logits /= temperature

            with timings("dec.sample"):
                local_order_idxs = Categorical(logits=filtered_logits).sample()
                all_local_order_idxs.append(local_order_idxs)

            with timings("dec.order_idxs"):
                # skip clamp_and_mask since it is handled elsewhere and is slow
                global_order_idxs = local_order_idxs_to_global(
                    local_order_idxs, cand_idxs, clamp_and_mask=False
                )
                all_global_order_idxs.append(global_order_idxs)

            with timings("dec.order_emb"):
                sampled_order_input = global_order_idxs.masked_fill(
                    global_order_idxs == EOS_IDX, 0
                )
                if teacher_force_orders is None:
                    order_input = sampled_order_input
                else:
                    order_input = torch.where(
                        teacher_force_orders[:, step] == NO_ORDER_ID,
                        sampled_order_input,
                        teacher_force_orders[:, step],
                    )

                order_emb = self.order_embedding(order_input)
                if self.featurize_output:
                    order_emb += self.order_feat_lin(self.order_feats[order_input])

                if self.relfeat_output:
                    order_enc = order_enc + order_emb[:, None] * alignments[:, :, None]

        with timings("dec.fin"):
            stacked_global_order_idxs = torch.stack(all_global_order_idxs, dim=1)
            stacked_local_order_idxs = torch.stack(all_local_order_idxs, dim=1)
            stacked_logits = cat_pad_sequences(
                [x.unsqueeze(1) for x in all_logits],
                seq_dim=2,
                cat_dim=1,
                pad_value=LOGIT_MASK_VAL,
            )
            r = stacked_global_order_idxs, stacked_local_order_idxs, stacked_logits

        # logging.debug(f"Timings[dec, {enc.shape[0]}x{step}] {timings}")

        return r


def _pad_last_dims(tensor, partial_new_shape, pad_value):
    assert len(tensor.shape) >= len(partial_new_shape), (tensor.shape, partial_new_shape)
    new_shape = list(tensor.shape)[: len(tensor.shape) - len(partial_new_shape)] + list(
        partial_new_shape
    )
    new_tensor = tensor.new_full(new_shape, pad_value)
    new_tensor[[slice(None, D) for D in tensor.shape]].copy_(tensor)
    return new_tensor


def top_p_filtering(
    logits: torch.Tensor, top_p: Union[float, torch.Tensor], min_tokens_to_keep=1
) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) filtering.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
        logits: tensor of shape [batch_size, vocab]. Logits distribution shape
        top_p: float or tensor of shape [batch_size, 1]. Keep the top tokens
            with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al.
            (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep: int, make sure we keep at least
            min_tokens_to_keep per batch example in the output

    Returns:
        top_p_mask: boolean tensor of shape [batch_size, vocab] with elements to remove.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return indices_to_remove


class DiplomacyModelEncoder(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        num_blocks,  # 16
        A,  # 81x81
        dropout,
        learnable_A=False,
        residual_linear=False,
        merged_gnn=False,
        layerdrop=0,
        use_global_pooling=False,
    ):
        super().__init__()

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            DiplomacyModelBlock(
                in_size=board_state_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
                dropout=dropout,
                residual_linear=residual_linear,
                use_global_pooling=use_global_pooling,
            )
        )
        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                DiplomacyModelBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                    dropout=dropout,
                    residual_linear=residual_linear,
                    use_global_pooling=use_global_pooling,
                )
            )

        if layerdrop > 1e-5:
            assert 0 < layerdrop <= 1.0, layerdrop
            self.layerdrop_rng = np.random.RandomState(0)
        else:
            self.layerdrop_rng = None
        self.layerdrop = layerdrop

        # prev orders blocks
        self.prev_orders_blocks = nn.ModuleList()
        self.prev_orders_blocks.append(
            DiplomacyModelBlock(
                in_size=prev_orders_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
                dropout=dropout,
                residual_linear=residual_linear,
                use_global_pooling=use_global_pooling,
            )
        )
        for _ in range(num_blocks - 1):
            self.prev_orders_blocks.append(
                DiplomacyModelBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                    dropout=dropout,
                    residual_linear=residual_linear,
                    use_global_pooling=use_global_pooling,
                )
            )

        self.merged_gnn = merged_gnn
        if self.merged_gnn:
            self.merged_blocks = nn.ModuleList()
            for _ in range(num_blocks // 2):
                self.merged_blocks.append(
                    DiplomacyModelBlock(
                        in_size=2 * inter_emb_size,
                        out_size=2 * inter_emb_size,
                        A=A,
                        residual=True,
                        learnable_A=learnable_A,
                        dropout=dropout,
                        residual_linear=residual_linear,
                        use_global_pooling=use_global_pooling,
                    )
                )

    def forward(self, x_bo, x_po):
        def apply_blocks_with_layerdrop(blocks, tensor):
            for i, block in enumerate(blocks):
                drop = (
                    i > 0
                    and self.training
                    and self.layerdrop_rng is not None
                    and self.layerdrop_rng.uniform() < self.layerdrop
                )
                if drop:
                    # To make distrubited happy we need to have grads for all params.
                    dummy = sum(w.sum() * 0 for w in block.parameters())
                    tensor = dummy + tensor
                else:
                    tensor = block(tensor)
            return tensor

        y_bo = apply_blocks_with_layerdrop(self.board_blocks, x_bo)
        y_po = apply_blocks_with_layerdrop(self.prev_orders_blocks, x_po)
        state_emb = torch.cat([y_bo, y_po], -1)

        if self.merged_gnn:
            state_emb = apply_blocks_with_layerdrop(self.merged_blocks, state_emb)
        return state_emb


class DiplomacyModelBlock(nn.Module):
    def __init__(
        self,
        *,
        in_size,
        out_size,
        A,
        dropout,
        residual=True,
        learnable_A=False,
        residual_linear=False,
        use_global_pooling=False,
    ):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A, learnable_A=learnable_A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.dropout = nn.Dropout(dropout or 0.0)
        self.residual = residual
        self.residual_linear = residual_linear
        if residual_linear:
            self.residual_lin = nn.Linear(in_size, out_size)
        self.use_global_pooling = use_global_pooling
        if use_global_pooling:
            self.post_pool_lin = nn.Linear(out_size, out_size, bias=False)

    def forward(self, x):
        # Shape [batch_idx, location, channel]
        y = self.graph_conv(x)
        if self.residual_linear:
            y += self.residual_lin(x)
        y = self.batch_norm(y)
        if self.use_global_pooling:
            # Global average pool over location
            g = torch.mean(y, dim=1, keepdim=True)
            g = self.dropout(g)
            g = self.post_pool_lin(g)
            # Add back transformed-pooled values as per-channel biases
            y += g
        y = F.relu(y)
        y = self.dropout(y)
        if self.residual:
            y += x
        return y


class TransformerEncoder(nn.Module):
    def __init__(
        self, *, board_state_size, prev_orders_size, spatial_size, inter_emb_size, encoder_cfg
    ):
        super().__init__()
        # Torch's encoder implementation has the restriction that the input size must match
        # the output size and also be equal to the number of heads times the channels per head
        # in the attention layer. That means that the input size must be evenly divisible by
        # the number of heads.

        # Also due to historical accident, inter_emb_size is actually only half of the actual internal
        # number of channels, this is the reason for all the "* 2" everywhere.
        num_heads = encoder_cfg.num_heads
        channels_per_head = inter_emb_size * 2 // num_heads
        assert inter_emb_size * 2 == channels_per_head * num_heads

        self.initial_linear = nn.Linear(
            board_state_size + prev_orders_size, inter_emb_size * 2, bias=False
        )
        self.initial_positional_bias = nn.Parameter(he_init((spatial_size, inter_emb_size * 2)))
        self.blocks = nn.ModuleList()
        for _ in range(encoder_cfg.num_blocks):
            self.blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=inter_emb_size * 2,
                    nhead=encoder_cfg.num_heads,
                    dim_feedforward=encoder_cfg.ff_channels,
                    dropout=encoder_cfg.dropout,
                )
            )

        layerdrop = encoder_cfg.layerdrop
        if layerdrop is not None and layerdrop > 1e-5:
            assert 0 < layerdrop <= 1.0, layerdrop
            self.layerdrop_rng = np.random.RandomState(0)
        else:
            self.layerdrop_rng = None
        self.layerdrop = layerdrop

    def forward(self, x_bo, x_po):
        x = torch.cat([x_bo, x_po], -1)
        x = self.initial_linear(x)
        x = x + self.initial_positional_bias
        # x: Shape [batch_size, spatial_size, inter_emb_size*2]
        # But torch needs [spatial_size, batch_size, inter_emb_size*2]
        x = x.transpose(0, 1).contiguous()

        def apply_blocks_with_layerdrop(blocks, tensor):
            for i, block in enumerate(blocks):
                drop = (
                    self.training
                    and self.layerdrop_rng is not None
                    and self.layerdrop_rng.uniform() < self.layerdrop
                )
                if drop:
                    # To make distributed happy we need to have grads for all params.
                    dummy = sum(w.sum() * 0 for w in block.parameters())
                    tensor = dummy + tensor
                else:
                    tensor = block(tensor)
            return tensor

        x = apply_blocks_with_layerdrop(self.blocks, x)

        x = x.transpose(0, 1).contiguous()
        return x


def he_init(shape):
    fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
    init_range = math.sqrt(2.0 / fan_in)
    return torch.randn(shape) * init_range


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, A, learnable_A=False):
        super().__init__()
        """
        A -> (81, 81)
        """
        self.A = nn.Parameter(A).requires_grad_(learnable_A)
        self.W = nn.Parameter(he_init((len(self.A), in_size, out_size)))
        self.b = nn.Parameter(torch.zeros(1, 1, out_size))

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """

        x = x.transpose(0, 1)  # (b, N, in )               => (N, b, in )
        x = torch.matmul(x, self.W)  # (N, b, in) * (N, in, out) => (N, b, out)
        x = x.transpose(0, 1)  # (N, b, out)               => (b, N, out)
        x = torch.matmul(self.A, x)  # (b, N, N) * (b, N, out)   => (b, N, out)
        x += self.b

        return x


class ValueDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        spatial_size,
        dropout,
        init_scale=1.0,
        softmax=False,
    ):
        super().__init__()
        emb_flat_size = spatial_size * inter_emb_size * 2
        self.prelin = nn.Linear(emb_flat_size, inter_emb_size)
        self.lin = nn.Linear(inter_emb_size, len(POWERS))

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.0)
        self.softmax = softmax

        # scale down init
        torch.nn.init.xavier_normal_(self.lin.weight, gain=init_scale)
        bound = init_scale / (len(POWERS) ** 0.5)
        torch.nn.init.uniform_(self.lin.bias, -bound, bound)

    def forward(self, enc):
        """Returns [B, 7] FloatTensor summing to 1 across dim=1"""
        B = enc.shape[0]
        y = enc.view(B, -1)

        y = self.prelin(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.lin(y)
        if self.softmax:
            y = F.softmax(y, -1)
        else:
            y = y ** 2
            y = y / y.sum(dim=1, keepdim=True)
        # y = nn.functional.softmax(y, dim=1)
        return y


def compute_order_features():
    """Returns a [13k x D] tensor where each row contains (one-hot) features for one order in the vocabulary.
    """

    order_vocabulary = get_order_vocabulary()
    # assert order_vocabulary[0] == EOS_TOKEN
    # order_vocabulary = order_vocabulary[1:]  # we'll fix this up at the end
    order_split = [o.split() for o in order_vocabulary]

    # fixup strange stuff in the dataset
    for s in order_split:
        # fixup "A SIL S A PRU"
        if len(s) == 5 and s[2] == "S":
            s.append("H")
        # fixup "A SMY - ROM VIA"
        if len(s) == 5 and s[-1] == "VIA":
            s.pop()

    loc_idx = {loc: i for i, loc in enumerate(LOCS)}
    unit_idx = {"A": 0, "F": 1}
    order_type_idx = {
        t: i for i, t in enumerate(sorted(list(set([s[2] for s in order_split if len(s) > 2]))))
    }

    num_locs = len(loc_idx)
    feats = []
    srcs, dsts = [], []
    for o in order_split:
        u = o[3:] if len(o) >= 6 else o
        srcT = torch.zeros(num_locs)
        dstT = torch.zeros(num_locs)
        unitT = torch.zeros(len(unit_idx))
        orderT = torch.zeros(len(order_type_idx))
        underlyingT = torch.zeros(len(order_type_idx))

        if not o[2].startswith("B"):  # lets ignore the concatenated builds, they're tricky
            src_loc = loc_idx[u[1]]
            dst_loc = loc_idx[u[3]] if len(u) >= 4 else loc_idx[u[1]]
            srcT[src_loc] = 1
            dstT[dst_loc] = 1
            unitT[
                unit_idx[o[0]]
            ] = 1  # FIXME: this is wrong! should be u[0]. But too late for backwards compatibility
            orderT[order_type_idx[o[2]]] = 1
            underlyingT[order_type_idx[u[2]]] = 1
            srcs.append(src_loc)
            dsts.append(dst_loc)
        else:
            srcs.append(-1)
            dsts.append(-1)

        feats.append(torch.cat((srcT, dstT, unitT, orderT, underlyingT), dim=-1))

    feats = torch.stack(feats, dim=0)
    # FIXME: this could be done better

    return feats, torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)
