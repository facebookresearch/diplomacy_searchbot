import torch
import time
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_incompatible_build_idxs_map,
    get_order_vocabulary,
)


def maybe_print(*args):
    pass


class DipNet(nn.Module):
    def __init__(
        self,
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
            board_state_size,
            prev_orders_size,
            inter_emb_size,
            power_emb_size,
            season_emb_size,
            num_blocks,
            A,
            learnable_A=learnable_A,
        )

        self.decoder = LSTMDipNetDecoder(
            inter_emb_size,
            orders_vocab_size,
            lstm_size,
            order_emb_size,
            lstm_dropout,
            master_alignments,
            learnable_alignments=learnable_alignments,
            avg_embedding=avg_embedding,
        )

    def valid_order_idxs_to_mask(self, valid_order_idxs, loc_idxs):
        assert valid_order_idxs.dtype == torch.int32
        valid_order_mask = torch.zeros(
            (valid_order_idxs.shape[0], valid_order_idxs.shape[1], self.orders_vocab_size),
            dtype=torch.bool,
            device=valid_order_idxs.device,
        )
        valid_order_mask.scatter_(-1, valid_order_idxs.long(), 1)
        valid_order_mask[:, :, 0] = 0  # remove EOS_TOKEN
        return valid_order_mask

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
        - loc_idxs: shape [B, S], long, 0 <= idx < 81
        - order_idxs: shape [B, S, 468]  padded with -1
        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of shape [B, 1]
        - teacher_force_orders: shape [B, S] int or None

        Returns:
        - order_idxs [B, S]: idx of sampled orders
        - order_scores [B, S, 13k]: masked pre-softmax logits of each order
        """
        tic = time.time()
        enc = self.encoder(x_bo, x_po, x_power_1h, x_season_1h)  # [B, 81, 240]
        res = self.decoder(
            enc,
            in_adj_phase,
            loc_idxs,
            self.valid_order_idxs_to_mask(valid_order_idxs, loc_idxs),
            temperature=temperature,
            teacher_force_orders=teacher_force_orders,
        )
        # print(f"DipNet forward time: {time.time() - tic}")
        return res


class LSTMDipNetDecoder(nn.Module):
    def __init__(
        self,
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
        self.avg_embedding = avg_embedding

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.lstm = nn.LSTM(
            2 * inter_emb_size + order_emb_size, lstm_size, batch_first=True, dropout=lstm_dropout
        )
        self.lstm_out_linear = nn.Linear(lstm_size, orders_vocab_size)

        # if avg_embedding is True, alignments are not used, and pytorch
        # complains about unused parameters, so only set self.master_alignments
        # when avg_embedding is False
        if not avg_embedding:
            # add a zero row, to make master_alignments 82 x 81, where alignments[-1] is zeros
            master_alignments = torch.cat(
                [
                    master_alignments,
                    torch.zeros(1, master_alignments.shape[1], dtype=master_alignments.dtype),
                ]
            )
            self.master_alignments = nn.Parameter(master_alignments).requires_grad_(
                learnable_alignments
            )

        # 13k x 13k table of compatible orders
        self.compatible_orders_table = ~(torch.eye(orders_vocab_size).bool())
        incompatible_orders = get_incompatible_build_idxs_map()
        for order, v in incompatible_orders.items():
            for incomp_order in enumerate(v):
                self.compatible_orders_table[order, incomp_order] = 0

    def forward(
        self, enc, in_adj_phase, loc_idxs, order_masks, temperature=1.0, teacher_force_orders=None
    ):
        totaltic = time.time()
        tic = time.time()
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

        for step in range(order_masks.shape[1]):
            order_mask = order_masks[:, step].contiguous()

            if self.avg_embedding:
                # no attention: average across loc embeddings
                loc_enc = torch.mean(enc, dim=1)
            else:
                # do static attention

                # adj_alignments averages over all locations in all steps
                # (hence we use all of loc_idxs to index at every step, unlike
                # not_adj_alignments). However, some sequences are longer than
                # others, and loc_idxs = -1 where y_actions = EOS_IDX. This
                # indexing/summation works only because loc_idxs = -1 and we
                # added an 82nd row (index -1) of zeros to master_alignments
                adj_alignments = torch.sum(self.master_alignments[loc_idxs], dim=1)
                not_adj_alignments = self.master_alignments[loc_idxs[:, step]]

                # gather row from adj_alignments where in_adj_phase = 1, else not_adj_alignments
                alignments = torch.gather(
                    torch.stack([not_adj_alignments, adj_alignments]),
                    0,
                    in_adj_phase.long().reshape(1, -1, 1).repeat(1, 1, adj_alignments.shape[-1]),
                ).squeeze(0)
                alignments /= torch.sum(alignments, dim=1, keepdim=True)
                alignments[alignments != alignments] = 0  # remove nans, caused by loc_idxs == -1
                loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            lstm_input = torch.cat((loc_enc, order_emb), dim=1).unsqueeze(1)  # [B, 1, 320]

            out, hidden = self.lstm(lstm_input, hidden)

            order_scores = self.lstm_out_linear(out.squeeze(1))  # [B, 13k]
            all_order_scores.append(order_scores)

            # unmask where there are no actions or the sampling will crash. The
            # losses at these points will be masked out later, so this is safe.
            invalid_mask = loc_idxs[:, step] == -1
            if invalid_mask.sum() == loc_idxs.shape[0]:
                maybe_print(f"Breaking at step {step} because no more orders to give")
                assert step > 0
                # early exit
                for _step in range(step, order_masks.shape[1]):  # fill in garbage
                    all_order_idxs.append(all_order_idxs[-1])
                    # FIXME(alerer): shouldn't we be filling in 0? (i.e. <EOS>)
                break
            order_mask[invalid_mask] = 1

            # make scores for invalid actions 0. This is faster  than order_scores[~order_mask] = float("-inf")
            # use 1e9 instead of inf because 0*inf=nan
            order_scores -= (1 - order_mask.float()) * 1e9

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

            # ugh, hack because I don't want to make compatible_orders_table a buffer (backwards-incompatible)
            self.compatible_orders_table = self.compatible_orders_table.to(order_mask.device)

            compatible_mask = self.compatible_orders_table[dont_repeat_orders]  # B x 13k
            order_masks[:, step:] *= compatible_mask.unsqueeze(1)

        res = torch.stack(all_order_idxs, dim=1), torch.stack(all_order_scores, dim=1)
        maybe_print("total: ", time.time() - totaltic)
        return res


class DipNetEncoder(nn.Module):
    def __init__(
        self,
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
                board_state_size,
                inter_emb_size,
                power_emb_size,
                season_emb_size,
                A,
                residual=False,
                learnable_A=learnable_A,
            )
        )
        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                DipNetBlock(
                    inter_emb_size,
                    inter_emb_size,
                    power_emb_size,
                    season_emb_size,
                    A,
                    residual=True,
                    learnable_A=learnable_A,
                )
            )

        # prev orders blocks
        self.prev_orders_blocks = nn.ModuleList()
        self.prev_orders_blocks.append(
            DipNetBlock(
                prev_orders_size,
                inter_emb_size,
                power_emb_size,
                season_emb_size,
                A,
                residual=False,
                learnable_A=learnable_A,
            )
        )
        for _ in range(num_blocks - 1):
            self.prev_orders_blocks.append(
                DipNetBlock(
                    inter_emb_size,
                    inter_emb_size,
                    power_emb_size,
                    season_emb_size,
                    A,
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


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, A, learnable_A=False):
        super().__init__()
        """
        A -> (81, 81)
        """
        self.A = nn.Parameter(A).requires_grad_(learnable_A)
        self.W = nn.Linear(in_size, out_size)

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """
        Wx = self.W(x)

        # following 3 lines are equivalent to (but faster than)
        # res = torch.matmul(self.A, Wx)
        Wx_shape = Wx.shape
        res = self.A @ Wx.transpose(0, 1).reshape(Wx_shape[1], -1)
        res = res.view(Wx_shape[1], Wx_shape[0], Wx_shape[2]).transpose(0, 1).contiguous()

        return res


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
