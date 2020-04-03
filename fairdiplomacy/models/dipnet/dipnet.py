import logging
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from fairdiplomacy.models.consts import POWERS, N_SCS
from fairdiplomacy.utils.timing_ctx import TimingCtx


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
        encoder_dropout=0,
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
            season_emb_size=season_emb_size,
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
            master_alignments=master_alignments,
            learnable_alignments=learnable_alignments,
            avg_embedding=avg_embedding,
            power_emb_size=power_emb_size,
        )

        self.value_decoder = ValueDecoder(inter_emb_size)

    def forward(
        self,
        x_bo,
        x_po,
        x_season_1h,
        in_adj_phase,
        loc_idxs,
        valid_order_idxs,
        temperature,
        teacher_force_orders=None,
        x_power=None,
    ):
        """
        Arguments:
        - x_bo: [B, 81, 35]
        - x_po: [B, 81, 40]
        - x_season_1h: [B, 3]
        - in_adj_phase: [B], bool
        - loc_idxs: int8, [B, 81] or [B, 7, 81]
        - valid_order_idxs: long, [B, S, 469] or [B, 7, S, 469]
        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of [B, 1]
        - teacher_force_orders: [B, S] int or None
        - x_power: [B, 7] or None

        if x_power is None, the model will decode for all 7 powers.
            - loc_idxs, valid_order_idxs, and teacher_force_orders must have an
              extra axis at dim=1 with size 7
            - order_idxs and order_scores will be returned with an extra axis
              at dim=1 with size 7
        if x_power is not None, it must be [B, 7] 1-hot, and only that power
              will be decoded

        Returns:
          - order_idxs [B, S] or [B, 7, S]: idx of sampled orders for each power
          - order_scores [B, S, 13k] or [B, 7, S, 13k]: masked pre-softmax
            logits of each order, for each power
          - final_scores [B, 7]: estimated final SC counts per power
        """
        if x_power is None:
            return self.forward_all_powers(
                x_bo=x_bo,
                x_po=x_po,
                x_season_1h=x_season_1h,
                in_adj_phase=in_adj_phase,
                loc_idxs=loc_idxs,
                valid_order_idxs=valid_order_idxs,
                temperature=temperature,
                teacher_force_orders=teacher_force_orders,
            )
        else:
            return self.forward_one_power(
                x_bo=x_bo,
                x_po=x_po,
                x_season_1h=x_season_1h,
                in_adj_phase=in_adj_phase,
                loc_idxs=loc_idxs,
                valid_order_idxs=valid_order_idxs,
                temperature=temperature,
                teacher_force_orders=teacher_force_orders,
                x_power=x_power,
            )

    def forward_one_power(
        self,
        *,
        x_bo,
        x_po,
        x_season_1h,
        in_adj_phase,
        loc_idxs,
        valid_order_idxs,
        x_power,
        temperature,
        teacher_force_orders,
    ):
        assert len(loc_idxs.shape) == 2
        assert len(valid_order_idxs.shape) == 3

        enc = self.encoder(x_bo, x_po, x_season_1h)  # [B, 81, 240]
        order_idxs, order_scores = self.policy_decoder(
            enc,
            in_adj_phase,
            loc_idxs,
            self.valid_order_idxs_to_mask(valid_order_idxs),
            temperature=temperature,
            teacher_force_orders=teacher_force_orders,
            power_1h=x_power,
        )
        final_scores = self.value_decoder(enc)

        return order_idxs, order_scores, final_scores

    def forward_all_powers(
        self,
        *,
        x_bo,
        x_po,
        x_season_1h,
        in_adj_phase,
        loc_idxs,
        valid_order_idxs,
        temperature,
        teacher_force_orders,
        log_timings=False,
    ):
        timings = TimingCtx()

        assert len(loc_idxs.shape) == 3
        assert len(valid_order_idxs.shape) == 4

        with timings("enc"):
            enc = self.encoder(x_bo, x_po, x_season_1h)  # [B, 81, 240]

        with timings("policy_decoder_prep"):
            enc = enc.repeat_interleave(7, dim=0)
            in_adj_phase = in_adj_phase.repeat_interleave(7, dim=0)
            loc_idxs = loc_idxs.view(-1, loc_idxs.shape[2])
            valid_order_masks = self.valid_order_idxs_to_mask(
                valid_order_idxs.view(-1, *valid_order_idxs.shape[2:])
            )
            temperature = (
                temperature.repeat_interleave(7, dim=0)
                if hasattr(temperature, "repeat_interleave")
                else temperature
            )
            teacher_force_orders = (
                teacher_force_orders.view(-1, *teacher_force_orders.shape[2:])
                if teacher_force_orders is not None
                else None
            )
            # N.B. use repeat, not repeat_interleave, for power_1h only. Each
            # batch is contiguous, and we want a 7x7 identity matrix for each batch.
            power_1h = torch.eye(7, device=enc.device).repeat((enc.shape[0] // 7, 1))

        with timings("policy_decoder"):
            order_idxs, order_scores = self.policy_decoder(
                enc,
                in_adj_phase,
                loc_idxs,
                valid_order_masks,
                temperature=temperature,
                teacher_force_orders=teacher_force_orders,
                power_1h=power_1h,
            )

        with timings("value_decoder"):
            final_scores = self.value_decoder(enc)

        with timings("finish"):
            order_idxs = order_idxs.view(-1, 7, *order_idxs.shape[1:])
            order_idxs *= (valid_order_idxs != 0).any(dim=-1).long()
            order_scores = order_scores.view(-1, 7, *order_scores.shape[1:])

        if log_timings:
            logging.debug(f"Timings[model, B={x_bo.shape[0]}]: {timings}")

        return order_idxs, order_scores, final_scores

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
        power_emb_size,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.order_emb_size = order_emb_size
        self.lstm_dropout = lstm_dropout
        self.avg_embedding = avg_embedding
        self.power_emb_size = power_emb_size

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.power_lin = nn.Linear(len(POWERS), power_emb_size)
        self.lstm = nn.LSTM(
            2 * inter_emb_size + order_emb_size + power_emb_size, lstm_size, batch_first=True
        )
        self.lstm_out_linear = nn.Linear(lstm_size, orders_vocab_size)

        # if avg_embedding is True, alignments are not used, and pytorch
        # complains about unused parameters, so only set self.master_alignments
        # when avg_embedding is False
        if not avg_embedding:
            self.master_alignments = nn.Parameter(master_alignments).requires_grad_(
                learnable_alignments
            )

    def forward(
        self,
        enc,
        in_adj_phase,
        loc_idxs,
        order_masks,
        power_1h,
        temperature=1.0,
        teacher_force_orders=None,
    ):
        device = next(self.parameters()).device

        if (loc_idxs == -1).all():
            return (
                torch.zeros(*order_masks.shape[:2], dtype=torch.long, device=device),
                torch.zeros(*order_masks.shape, device=device),
            )

        self.lstm.flatten_parameters()

        hidden = (
            torch.zeros(1, enc.shape[0], self.lstm_size).to(device),
            torch.zeros(1, enc.shape[0], self.lstm_size).to(device),
        )

        # embedding for the last decoded order
        order_emb = torch.zeros(enc.shape[0], self.order_emb_size).to(device)

        # power embedding, constant for each lstm step
        assert len(power_1h.shape) == 2 and power_1h.shape[1] == 7, power_1h.shape
        power_emb = self.power_lin(power_1h)

        # return values: chosen order idxs, and scores (logits)
        all_order_idxs = []
        all_order_scores = []

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
                alignments = torch.where(
                    torch.isnan(alignments), torch.zeros_like(alignments), alignments
                )
                loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            lstm_input = torch.cat((loc_enc, order_emb, power_emb), dim=1).unsqueeze(1)
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

        return torch.stack(all_order_idxs, dim=1), torch.stack(all_order_scores, dim=1)


class DipNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        season_emb_size,  # 20
        num_blocks,  # 16
        A,  # 81x81
        dropout,
        learnable_A=False,
    ):
        super().__init__()

        # power/season embeddings
        self.season_lin = nn.Linear(3, season_emb_size)

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            DipNetBlock(
                in_size=board_state_size,
                out_size=inter_emb_size,
                season_emb_size=season_emb_size,
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
                    season_emb_size=season_emb_size,
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
                season_emb_size=season_emb_size,
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
                    season_emb_size=season_emb_size,
                    A=A,
                    residual=True,
                    learnable_A=learnable_A,
                    dropout=dropout,
                )
            )

    def forward(self, x_bo, x_po, x_season_1h):
        season_emb = self.season_lin(x_season_1h)

        y_bo = x_bo
        for block in self.board_blocks:
            y_bo = block(y_bo, season_emb)

        y_po = x_po
        for block in self.prev_orders_blocks:
            y_po = block(y_po, season_emb)

        state_emb = torch.cat([y_bo, y_po], -1)
        return state_emb


class DipNetBlock(nn.Module):
    def __init__(
        self, *, in_size, out_size, season_emb_size, A, dropout, residual=True, learnable_A=False
    ):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A, learnable_A=learnable_A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.film = FiLM(season_emb_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, season_emb):
        y = self.graph_conv(x)
        y = self.batch_norm(y)
        y = self.film(y, season_emb)
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


class FiLM(nn.Module):
    def __init__(self, season_emb_size, out_size):
        super().__init__()
        self.W_gamma = nn.Linear(season_emb_size, out_size)
        self.W_beta = nn.Linear(season_emb_size, out_size)

    def forward(self, x, season_emb):
        """Modulate x by gamma/beta calculated from power/season embeddings

        x -> (B, out_size)
        season_emb -> (B, season_emb_size)
        """
        gamma = self.W_gamma(season_emb).unsqueeze(1)
        beta = self.W_beta(season_emb).unsqueeze(1)
        return gamma * x + beta


class ValueDecoder(nn.Module):
    def __init__(self, inter_emb_size):
        super().__init__()
        self.lin = nn.Linear(81 * inter_emb_size * 2, len(POWERS))

    def forward(self, enc):
        """Returns [B, 7] FloatTensor summing to 34 across dim=1"""
        y = self.lin(enc.view(enc.shape[0], -1))  # [B, 7]
        return y / torch.sum(y, dim=1, keepdim=True) * N_SCS
