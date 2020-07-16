from typing import Union
import logging
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from fairdiplomacy.models.consts import POWERS, N_SCS
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.padded_embedding import PaddedEmbedding
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.models.consts import LOCS
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary, EOS_IDX

EOS_TOKEN = get_order_vocabulary()[EOS_IDX]


class DipNet(nn.Module):
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
        avg_embedding=False,
        value_decoder_init_scale=1.0,
        featurize_output=False,
        relfeat_output=False,
        featurize_prev_orders=False,
    ):
        super().__init__()
        self.orders_vocab_size = orders_vocab_size

        self.featurize_prev_orders = featurize_prev_orders
        self.prev_order_enc_size = prev_order_emb_size
        if featurize_prev_orders:
            order_feats, _srcs, _dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            self.prev_order_enc_size += self.order_feats.shape[-1]

        self.encoder = DipNetEncoder(
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
        )

        self.policy_decoder = LSTMDipNetDecoder(
            inter_emb_size=inter_emb_size,
            orders_vocab_size=orders_vocab_size,
            lstm_size=lstm_size,
            order_emb_size=order_emb_size,
            lstm_dropout=lstm_dropout,
            lstm_layers=lstm_layers,
            master_alignments=master_alignments,
            learnable_alignments=learnable_alignments,
            avg_embedding=avg_embedding,
            power_emb_size=power_emb_size,
            A=A,
            featurize_output=featurize_output,
            relfeat_output=relfeat_output,
        )

        self.value_decoder = ValueDecoder(
            inter_emb_size=inter_emb_size,
            init_scale=value_decoder_init_scale,
            dropout=value_dropout,
        )

        self.season_lin = nn.Linear(3, season_emb_size)
        self.prev_order_embedding = nn.Embedding(
            orders_vocab_size, prev_order_emb_size, padding_idx=0
        )

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
        teacher_force_orders=None,
        x_power=None,
        x_has_press=None,
    ):
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
        - teacher_force_orders: [B, S] long or None, ORDER idxs, NOT candidate idxs, 0-padded
        - x_power: [B, 7] or None
        - x_has_press: [B, 1] or None

        if x_power is None, the model will decode for all 7 powers.
            - loc_idxs, all_cand_idxs, and teacher_force_orders must have an
              extra axis at dim=1 with size 7
            - order_idxs and order_scores will be returned with an extra axis
              at dim=1 with size 7
        if x_power is not None, it must be [B, 7] 1-hot, and only that power
              will be decoded

        Returns:
          - order_idxs [B, S] or [B, 7, S]: idx in ORDER_VOCABULARY of sampled
            orders for each power
          - sampled_idxs [B, S] or [B, 7, S]: idx in all_cand_idxs of sampled
            orders for each power
          - logits [B, S, C] or [B, 7, S, C]: masked pre-softmax logits of each
            candidate order, 0 < S <= 17, 0 < C <= 469
          - final_sos [B, 7]: estimated sum of squares share for each power
        """

        # following https://arxiv.org/pdf/2006.04635.pdf , Appendix C
        B, NUM_LOCS, _ = x_board_state.shape

        # A. get season and prev order embs
        x_season_emb = self.season_lin(x_season)

        x_prev_order_emb = self.prev_order_embedding(x_prev_orders[:, 0])
        if self.featurize_prev_orders:
            x_prev_order_emb = torch.cat(
                (x_prev_order_emb, self.order_feats[x_prev_orders[:, 0]]), dim=-1
            )

        # B. insert the prev orders into the correct board location (which is in the second column of x_po)
        x_prev_order_exp = torch.zeros(
            (B, NUM_LOCS, self.prev_order_enc_size), device=x_board_state.device
        )
        prev_order_loc_idxs = torch.arange(B, device=x_board_state.device).repeat_interleave(
            x_prev_orders.shape[-1]
        ) * NUM_LOCS + x_prev_orders[:, 1].reshape(-1)
        x_prev_order_exp.view(-1, self.prev_order_enc_size).index_add_(
            0, prev_order_loc_idxs, x_prev_order_emb.view(-1, self.prev_order_enc_size)
        )

        # concatenate the subcomponents into board state and prev state, following the paper
        x_build_numbers_exp = x_build_numbers[:, None].expand(-1, NUM_LOCS, -1)
        x_season_emb_exp = x_season_emb[:, None].expand(-1, NUM_LOCS, -1)
        if x_has_press is not None:
            x_has_press_exp = x_has_press[:, None].expand(-1, NUM_LOCS, 1)
        else:
            x_has_press_exp = torch.zeros((B, NUM_LOCS, 1), device=x_board_state.device)
        x_bo_hat = torch.cat(
            (x_board_state, x_build_numbers_exp, x_season_emb_exp, x_has_press_exp), dim=-1
        )
        x_po_hat = torch.cat(
            (
                x_prev_state,
                x_prev_order_exp,
                x_build_numbers_exp,
                x_season_emb_exp,
                x_has_press_exp,
            ),
            dim=-1,
        )

        if x_power is None:
            return self.forward_all_powers(
                x_bo=x_bo_hat,
                x_po=x_po_hat,
                in_adj_phase=x_in_adj_phase,
                loc_idxs=x_loc_idxs,
                cand_idxs=x_possible_actions,
                temperature=temperature,
                top_p=top_p,
                teacher_force_orders=teacher_force_orders,
                x_has_press=x_has_press,
            )
        else:
            return self.forward_one_power(
                x_bo=x_bo_hat,
                x_po=x_po_hat,
                in_adj_phase=x_in_adj_phase,
                loc_idxs=x_loc_idxs,
                cand_idxs=x_possible_actions,
                temperature=temperature,
                top_p=top_p,
                teacher_force_orders=teacher_force_orders,
                x_power=x_power,
                x_has_press=x_has_press,
            )

    def forward_one_power(
        self,
        *,
        x_bo,
        x_po,
        in_adj_phase,
        loc_idxs,
        cand_idxs,
        x_power,
        temperature,
        top_p,
        teacher_force_orders,
        x_has_press,
    ):
        assert len(loc_idxs.shape) == 2
        assert len(cand_idxs.shape) == 3

        enc = self.encoder(x_bo, x_po)  # [B, 81, 240]
        order_idxs, sampled_idxs, logits = self.policy_decoder(
            enc,
            in_adj_phase,
            loc_idxs,
            cand_idxs,
            temperature=temperature,
            teacher_force_orders=teacher_force_orders,
            power_1h=x_power,
        )
        final_sos = self.value_decoder(enc)

        return order_idxs, sampled_idxs, logits, final_sos

    def forward_all_powers(
        self,
        *,
        x_bo,
        x_po,
        in_adj_phase,
        loc_idxs,
        cand_idxs,
        temperature,
        teacher_force_orders,
        top_p,
        x_has_press,
        log_timings=False,
    ):
        timings = TimingCtx()

        assert len(loc_idxs.shape) == 3
        assert len(cand_idxs.shape) == 4

        with timings("enc"):
            enc = self.encoder(x_bo, x_po)  # [B, 81, 240]

        with timings("policy_decoder_prep"):
            enc_repeat = enc.repeat_interleave(7, dim=0)
            in_adj_phase = in_adj_phase.repeat_interleave(7, dim=0)
            loc_idxs = loc_idxs.view(-1, loc_idxs.shape[2])
            cand_idxs = cand_idxs.view(-1, *cand_idxs.shape[2:])
            temperature = (
                temperature.repeat_interleave(7, dim=0)
                if hasattr(temperature, "repeat_interleave")
                else temperature
            )
            top_p = (
                top_p.repeat_interleave(7, dim=0) if hasattr(top_p, "repeat_interleave") else top_p
            )
            teacher_force_orders = (
                teacher_force_orders.view(-1, *teacher_force_orders.shape[2:])
                if teacher_force_orders is not None
                else None
            )
            # N.B. use repeat, not repeat_interleave, for power_1h only. Each
            # batch is contiguous, and we want a 7x7 identity matrix for each batch.
            power_1h = torch.eye(7, device=enc.device).repeat((enc.shape[0], 1))

        with timings("policy_decoder"):
            order_idxs, sampled_idxs, logits = self.policy_decoder(
                enc_repeat,
                in_adj_phase,
                loc_idxs,
                cand_idxs,
                temperature=temperature,
                top_p=top_p,
                teacher_force_orders=teacher_force_orders,
                power_1h=power_1h,
            )

        with timings("value_decoder"):
            final_sos = self.value_decoder(enc)

        with timings("finish"):
            # reshape
            order_idxs = order_idxs.view(-1, 7, *order_idxs.shape[1:])
            sampled_idxs = sampled_idxs.view(-1, 7, *sampled_idxs.shape[1:])
            cand_idxs = cand_idxs.view(-1, 7, *cand_idxs.shape[1:])
            logits = logits.view(-1, 7, *logits.shape[1:])

            # mask out garbage outputs
            valid_mask = (cand_idxs != EOS_IDX).any(dim=-1)
            eos_fill = torch.empty_like(order_idxs, requires_grad=False).fill_(EOS_IDX)
            order_idxs = torch.where(valid_mask, order_idxs, eos_fill)
            sampled_idxs = torch.where(valid_mask, sampled_idxs, eos_fill)

        if log_timings:
            logging.debug(f"Timings[model, B={x_bo.shape[0]}]: {timings}")

        return order_idxs, sampled_idxs, logits, final_sos


def compute_alignments(loc_idxs, step, A):
    alignments = torch.matmul(((loc_idxs == step) | (loc_idxs == -2)).float(), A)
    alignments /= torch.sum(alignments, dim=1, keepdim=True) + 1e-5
    # alignments = torch.where(
    #     torch.isnan(alignments), torch.zeros_like(alignments), alignments
    # )

    return alignments


class LSTMDipNetDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        orders_vocab_size,
        lstm_size,
        order_emb_size,
        lstm_dropout,
        lstm_layers,
        master_alignments,
        learnable_alignments=False,
        avg_embedding=False,
        power_emb_size,
        A=None,
        featurize_output=False,
        relfeat_output=False,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.order_emb_size = order_emb_size
        self.lstm_dropout = lstm_dropout
        self.avg_embedding = avg_embedding
        self.power_emb_size = power_emb_size

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.cand_embedding = PaddedEmbedding(orders_vocab_size, lstm_size, padding_idx=EOS_IDX)
        self.power_lin = nn.Linear(len(POWERS), power_emb_size)

        self.lstm = nn.LSTM(
            2 * inter_emb_size + order_emb_size + power_emb_size,
            lstm_size,
            batch_first=True,
            num_layers=self.lstm_layers,
        )

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
        valid = (flat_order_locs > 0).nonzero().squeeze(-1)
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
        power_1h,
        temperature=1.0,
        top_p=1.0,
        teacher_force_orders=None,
    ):
        timings = TimingCtx()
        with timings("dec.prep"):
            device = next(self.parameters()).device

            if (loc_idxs == -1).all():
                return (
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    torch.zeros(*all_cand_idxs.shape, device=device),
                )

            # embedding for the last decoded order
            order_emb = torch.zeros(enc.shape[0], self.order_emb_size, device=device)

            # power embedding, constant for each lstm step
            assert len(power_1h.shape) == 2 and power_1h.shape[1] == 7, power_1h.shape
            power_emb = self.power_lin(power_1h)

            # return values: chosen order idxs, candidate idxs, and logits
            all_order_idxs = []
            all_sampled_idxs = []
            all_logits = []

            order_enc = torch.zeros(enc.shape[0], 81, self.order_emb_size, device=enc.device)

            self.lstm.flatten_parameters()
            hidden = (
                torch.zeros(self.lstm_layers, enc.shape[0], self.lstm_size, device=device),
                torch.zeros(self.lstm_layers, enc.shape[0], self.lstm_size, device=device),
            )

            # reuse same dropout weights for all steps
            dropout_in = (
                torch.zeros(
                    enc.shape[0],
                    1,
                    enc.shape[2] + self.order_emb_size + self.power_emb_size,
                    device=enc.device,
                )
                .bernoulli_(1 - self.lstm_dropout)
                .div_(1 - self.lstm_dropout)
                .requires_grad_(False)
            )
            dropout_out = (
                torch.zeros(enc.shape[0], 1, self.lstm_size, device=enc.device)
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
            with timings("dec.loc_enc"):
                num_cands = max_cand_per_step[step]
                cand_idxs = all_cand_idxs[:, step, :num_cands].long().contiguous()

                if self.avg_embedding:
                    # no attention: average across loc embeddings
                    loc_enc = torch.mean(enc, dim=1)
                else:
                    # do static attention; set alignments to:
                    # - master_alignments for the right loc_idx when not in_adj_phase
                    in_adj_phase = in_adj_phase.view(-1, 1)
                    alignments = compute_alignments(loc_idxs, step, self.master_alignments)
                    # print('alignments', alignments.mean(), alignments.std())
                    loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            with timings("dec.lstm"):
                lstm_input = torch.cat((loc_enc, order_emb, power_emb), dim=1).unsqueeze(1)
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
                    out_with_ones = torch.cat(
                        (out, torch.ones((out.shape[0], 1, 1), device=out.device)), dim=1
                    )
                    order_scores_featurized = torch.bmm(order_w, out_with_ones)
                    logits += order_scores_featurized.squeeze(-1)

            with timings("dec.invalid_mask"):
                # unmask where there are no actions or the sampling will crash. The
                # losses at these points will be masked out later, so this is safe.
                invalid_mask = ~(cand_idxs != EOS_IDX).any(dim=1)
                if invalid_mask.all():
                    # early exit
                    logging.debug(f"Breaking at step {step} because no more orders to give")
                    for _step in range(step, all_cand_idxs.shape[1]):  # fill in garbage
                        all_order_idxs.append(
                            torch.empty(
                                all_cand_idxs.shape[0],
                                dtype=torch.long,
                                device=all_cand_idxs.device,
                            ).fill_(EOS_IDX)
                        )
                        all_sampled_idxs.append(
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
                logits = torch.min(logits, cand_mask.float() * 1e9 - 1e8)
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
                sampled_idxs = Categorical(logits=filtered_logits).sample()
                all_sampled_idxs.append(sampled_idxs)

            with timings("dec.order_idxs"):
                order_idxs = torch.gather(cand_idxs, 1, sampled_idxs.view(-1, 1)).view(-1)
                all_order_idxs.append(order_idxs)

            with timings("dec.order_emb"):
                order_input = (
                    teacher_force_orders[:, step]
                    if teacher_force_orders is not None
                    else order_idxs.masked_fill(order_idxs == EOS_IDX, 0)
                )

                order_emb = self.order_embedding(order_input)
                if self.featurize_output:
                    order_emb += self.order_feat_lin(self.order_feats[order_input])

                if self.relfeat_output:
                    order_enc = order_enc + order_emb[:, None] * alignments[:, :, None]

        with timings("dec.fin"):
            stacked_order_idxs = torch.stack(all_order_idxs, dim=1)
            stacked_sampled_idxs = torch.stack(all_sampled_idxs, dim=1)
            stacked_logits = cat_pad_sequences(
                [x.unsqueeze(1) for x in all_logits], seq_dim=2, cat_dim=1, pad_value=-1e8
            )[0]
            r = stacked_order_idxs, stacked_sampled_idxs, stacked_logits

        logging.debug(f"Timings[dec, {enc.shape[0]}x{step}] {timings}")

        return r


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


class DipNetEncoder(nn.Module):
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
    ):
        super().__init__()

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            DipNetBlock(
                in_size=board_state_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
                dropout=dropout,
            )
        )
        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                DipNetBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                    dropout=dropout,
                )
            )

        # prev orders blocks
        self.prev_orders_blocks = nn.ModuleList()
        self.prev_orders_blocks.append(
            DipNetBlock(
                in_size=prev_orders_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
                dropout=dropout,
            )
        )
        for _ in range(num_blocks - 1):
            self.prev_orders_blocks.append(
                DipNetBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                    dropout=dropout,
                )
            )

    def forward(self, x_bo, x_po):

        y_bo = x_bo
        for block in self.board_blocks:
            y_bo = block(y_bo)

        y_po = x_po
        for block in self.prev_orders_blocks:
            y_po = block(y_po)

        state_emb = torch.cat([y_bo, y_po], -1)
        return state_emb


class DipNetBlock(nn.Module):
    def __init__(self, *, in_size, out_size, A, dropout, residual=True, learnable_A=False):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A, learnable_A=learnable_A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.batch_norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.residual:
            y += x
        return y


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
    def __init__(self, *, inter_emb_size, dropout, init_scale=1.0):
        super().__init__()
        emb_flat_size = 81 * inter_emb_size * 2
        self.prelin = nn.Linear(emb_flat_size, inter_emb_size)
        self.lin = nn.Linear(inter_emb_size, len(POWERS))

        self.dropout = nn.Dropout(dropout)

        # scale down init
        torch.nn.init.xavier_normal_(self.lin.weight, gain=init_scale)
        bound = init_scale / (len(POWERS) ** 0.5)
        torch.nn.init.uniform_(self.lin.bias, -bound, bound)

    def forward(self, enc):
        """Returns [B, 7] FloatTensor summing to 1 across dim=1"""
        y = enc.view(enc.shape[0], -1)
        y = self.prelin(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.lin(y)
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
