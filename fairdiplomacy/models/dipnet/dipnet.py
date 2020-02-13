import logging
import math
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from fairdiplomacy.models.dipnet.order_vocabulary import get_incompatible_build_idxs_map


class DipNet(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 60
        season_emb_size,  # 20
        num_blocks,  # 16
        A,  # 81x81
        master_alignments,
        orders_vocab_size,  # 13k
        lstm_size,  # 200
        order_emb_size,  # 80
        lstm_dropout=0,
        learnable_A=False,
        learnable_alignments=False,
        avg_embedding=False,
    ):
        super().__init__()
        self.orders_vocab_size = orders_vocab_size
        self.encoder = DipNetEncoder(
            board_state_size=board_state_size,
            prev_orders_size=prev_orders_size,
            inter_emb_size=inter_emb_size,
            power_emb_size=power_emb_size,
            season_emb_size=season_emb_size,
            num_blocks=num_blocks,
            A=A,
            learnable_A=learnable_A,
        )

        self.decoder = LSTMDipNetDecoder(
            inter_emb_size=inter_emb_size,
            orders_vocab_size=orders_vocab_size,
            lstm_size=lstm_size,
            order_emb_size=order_emb_size,
            lstm_dropout=lstm_dropout,
            master_alignments=master_alignments,
            learnable_alignments=learnable_alignments,
            avg_embedding=avg_embedding,
        )

    def forward(
        self,
        x_bo,
        x_po,
        x_power_1h,
        x_season_1h,
        in_adj_phase,
        loc_idxs,
        valid_order_idxs,
        temperature=1.0,
        teacher_force_orders=None,
    ):
        """
        Arguments:
        - x_bo: shape [B, 81, 35]
        - x_po: shape [B, 81, 40]
        - x_power_1h: shape [B, 7]
        - x_season_1h: shape [B, 3]
        - in_adj_phase: shape [B], bool
        - loc_idxs: shape [B, 81], int8
        - valid_order_idxs: shape [B, S, 469], long
        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of shape [B, 1]
        - teacher_force_orders: shape [B, S] int or None

        Returns:
        - order_idxs [B, S]: idx of sampled orders
        - order_scores [B, S, 13k]: masked pre-softmax logits of each order
        """
        if (valid_order_idxs == 0).all():
            logging.warning("foward called with all valid_order_idxs == 0")
            return (
                torch.zeros(
                    *(valid_order_idxs.shape[:2]),
                    dtype=valid_order_idxs.dtype,
                    device=valid_order_idxs.device,
                ),
                torch.zeros(
                    *valid_order_idxs.shape, self.orders_vocab_size, device=valid_order_idxs.device
                ),
            )

        enc = self.encoder(x_bo, x_po, x_power_1h, x_season_1h)  # [B, 81, 240]
        res = self.decoder(
            enc,
            in_adj_phase,
            loc_idxs,
            self.valid_order_idxs_to_mask(valid_order_idxs),
            temperature=temperature,
            teacher_force_orders=teacher_force_orders,
        )
        return res

    def valid_order_idxs_to_mask(self, valid_order_idxs):
        """
        Arguments:
        - valid_order_idxs: [B, S, 469] torch.int32, idxs into ORDER_VOCABULARY

        Returns [B, S, 13k] bool mask
        """
        valid_order_mask = torch.zeros(
            (valid_order_idxs.shape[0], valid_order_idxs.shape[1], self.orders_vocab_size),
            dtype=torch.bool,
            device=valid_order_idxs.device,
        )
        valid_order_mask.scatter_(-1, valid_order_idxs.long(), 1)
        valid_order_mask[:, :, 0] = 0  # remove EOS_TOKEN
        return valid_order_mask


class LSTMDipNetDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        orders_vocab_size,
        lstm_size,
        order_emb_size,
        lstm_dropout,
        master_alignments,
        learnable_alignments=False,
        avg_embedding=False,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.order_emb_size = order_emb_size
        self.lstm_dropout = lstm_dropout
        self.avg_embedding = avg_embedding

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.lstm = nn.LSTM(2 * inter_emb_size + order_emb_size, lstm_size, batch_first=True)
        self.lstm_out_linear = nn.Linear(lstm_size, orders_vocab_size)

        # if avg_embedding is True, alignments are not used, and pytorch
        # complains about unused parameters, so only set self.master_alignments
        # when avg_embedding is False
        if not avg_embedding:
            self.master_alignments = nn.Parameter(master_alignments).requires_grad_(
                learnable_alignments
            )

        # 13k x 13k table of compatible orders
        self.compatible_orders_table = ~(torch.eye(orders_vocab_size).bool())
        incompatible_orders = get_incompatible_build_idxs_map()
        assert len(incompatible_orders) > 0
        for order, v in incompatible_orders.items():
            for incomp_order in v:
                self.compatible_orders_table[order, incomp_order] = 0

    def forward(
        self, enc, in_adj_phase, loc_idxs, order_masks, temperature=1.0, teacher_force_orders=None
    ):
        totaltic = time.time()
        device = next(self.parameters()).device
        self.lstm.flatten_parameters()

        hidden = (
            torch.zeros(1, enc.shape[0], self.lstm_size).to(device),
            torch.zeros(1, enc.shape[0], self.lstm_size).to(device),
        )

        # embedding for the last decoded order
        order_emb = torch.zeros(enc.shape[0], self.order_emb_size).to(device)

        # return values: chosen order idxs, and scores (logits)
        all_order_idxs = []
        all_order_scores = []

        # reuse same dropout weights for all steps
        dropout_in = (
            torch.zeros(enc.shape[0], 1, enc.shape[2] + self.order_emb_size, device=enc.device)
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

        for step in range(order_masks.shape[1]):
            order_mask = order_masks[:, step].contiguous()

            if self.avg_embedding:
                # no attention: average across loc embeddings
                loc_enc = torch.mean(enc, dim=1)
            else:
                # do static attention; set alignments to:
                # - master_alignments for the right loc_idx when not in_adj_phase
                in_adj_phase = in_adj_phase.view(-1, 1)
                alignments = torch.matmul(
                    ((loc_idxs == step) | (loc_idxs == -2)).float(), self.master_alignments
                )
                alignments /= torch.sum(alignments, dim=1, keepdim=True)
                alignments[alignments != alignments] = 0  # remove nans, caused by loc_idxs == -1
                loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            lstm_input = torch.cat((loc_enc, order_emb), dim=1).unsqueeze(1)  # [B, 1, 320]
            if self.training and self.lstm_dropout < 1.0:
                lstm_input = lstm_input * dropout_in

            out, hidden = self.lstm(lstm_input, hidden)
            if self.training and self.lstm_dropout < 1.0:
                out = out * dropout_out
            order_scores = self.lstm_out_linear(out.squeeze(1))  # [B, 13k]

            # unmask where there are no actions or the sampling will crash. The
            # losses at these points will be masked out later, so this is safe.
            invalid_mask = ~(loc_idxs != -1).any(dim=1)
            if invalid_mask.all():
                # early exit
                logging.debug(f"Breaking at step {step} because no more orders to give")
                for _step in range(step, order_masks.shape[1]):  # fill in garbage
                    all_order_idxs.append(
                        torch.zeros(
                            order_masks.shape[0], dtype=torch.long, device=order_masks.device
                        )
                    )
                break
            order_mask[invalid_mask] = 1

            # make scores for invalid actions 0. This is faster than
            # order_scores[~order_mask] = float("-inf") use 1e9 instead of inf
            # because 0*inf=nan
            order_scores = torch.min(order_scores, order_mask.float() * 1e9 - 1e8)
            all_order_scores.append(order_scores)

            order_idxs = Categorical(logits=order_scores / temperature).sample()
            all_order_idxs.append(order_idxs)

            if teacher_force_orders is not None:
                order_emb = self.order_embedding(teacher_force_orders[:, step])
            else:
                order_emb = self.order_embedding(order_idxs).squeeze(1)

            # Mask out chosen actions in future steps to prevent the same
            # order from occuring twice in a single turn
            dont_repeat_orders = (
                teacher_force_orders[:, step] if teacher_force_orders is not None else order_idxs
            )

            # ugh, hack because I don't want to make compatible_orders_table a
            # buffer (backwards-incompatible)
            self.compatible_orders_table = self.compatible_orders_table.to(order_mask.device)

            compatible_mask = self.compatible_orders_table[dont_repeat_orders]  # B x 13k
            order_masks[:, step:] *= compatible_mask.unsqueeze(1)

        res = torch.stack(all_order_idxs, dim=1), torch.stack(all_order_scores, dim=1)
        # logging.debug("total: {}".format(time.time() - totaltic))
        return res


class DipNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 60
        season_emb_size,  # 20
        num_blocks,  # 16
        A,  # 81x81
        learnable_A=False,
    ):
        super().__init__()

        # power/season embeddings
        self.power_lin = nn.Linear(7, power_emb_size)
        self.season_lin = nn.Linear(3, season_emb_size)

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            DipNetBlock(
                in_size=board_state_size,
                out_size=inter_emb_size,
                power_emb_size=power_emb_size,
                season_emb_size=season_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
            )
        )
        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                DipNetBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    power_emb_size=power_emb_size,
                    season_emb_size=season_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                )
            )

        # prev orders blocks
        self.prev_orders_blocks = nn.ModuleList()
        self.prev_orders_blocks.append(
            DipNetBlock(
                in_size=prev_orders_size,
                out_size=inter_emb_size,
                power_emb_size=power_emb_size,
                season_emb_size=season_emb_size,
                A=A,
                residual=False,
                learnable_A=learnable_A,
            )
        )
        for _ in range(num_blocks - 1):
            self.prev_orders_blocks.append(
                DipNetBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    power_emb_size=power_emb_size,
                    season_emb_size=season_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                )
            )

    def forward(self, x_bo, x_po, x_power_1h, x_season_1h):
        power_emb = self.power_lin(x_power_1h)
        season_emb = self.season_lin(x_season_1h)

        y_bo = x_bo
        for block in self.board_blocks:
            y_bo = block(y_bo, power_emb, season_emb)

        y_po = x_po
        for block in self.prev_orders_blocks:
            y_po = block(y_po, power_emb, season_emb)

        state_emb = torch.cat([y_bo, y_po], -1)
        return state_emb


class DipNetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_size,
        out_size,
        power_emb_size,
        season_emb_size,
        A,
        residual=True,
        learnable_A=False,
    ):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A, learnable_A=learnable_A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.film = FiLM(power_emb_size, season_emb_size, out_size)
        self.residual = residual

    def forward(self, x, power_emb, season_emb):
        y = self.graph_conv(x)
        y = self.batch_norm(y)
        y = self.film(y, power_emb, season_emb)
        y = F.relu(y)
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


class FiLM(nn.Module):
    def __init__(self, power_emb_size, season_emb_size, out_size):
        super().__init__()
        self.W_gamma = nn.Linear(power_emb_size + season_emb_size, out_size)
        self.W_beta = nn.Linear(power_emb_size + season_emb_size, out_size)

    def forward(self, x, power_emb, season_emb):
        """Modulate x by gamma/beta calculated from power/season embeddings

        x -> (B, out_size)
        power_emb -> (B, power_emb_size)
        season_emb -> (B, season_emb_size)
        """
        mod_emb = torch.cat([power_emb, season_emb], -1)
        gamma = self.W_gamma(mod_emb).unsqueeze(1)
        beta = self.W_beta(mod_emb).unsqueeze(1)
        return gamma * x + beta
