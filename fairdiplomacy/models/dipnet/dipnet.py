import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical
import logging


class DipNet(nn.Module):
    def __init__(
        self,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 7 (one-hot)
        season_emb_size,  # 3 (one-hot)
        num_blocks,  # 16
        A,  # 81x81
        orders_vocab_size,  # 13k
        lstm_size,  # 200
        order_emb_size,  # 80
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.order_emb_size = order_emb_size

        self.encoder = DipNetEncoder(
            board_state_size,
            prev_orders_size,
            inter_emb_size,
            power_emb_size,
            season_emb_size,
            num_blocks,
            A,
        )
        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.lstm = nn.LSTM(2 * inter_emb_size + order_emb_size, lstm_size, batch_first=True)
        self.lstm_out_linear = nn.Linear(lstm_size, orders_vocab_size)

    def forward(self, x_bo, x_po, power_emb, season_emb, order_masks, temperature=0.1):
        """
        Arguments:
        - x_bo: shape [B, 81, 35]
        - x_po: shape [B, 81, 40]
        - power_emb: shape [B, 7]
        - season_emb: shape [B, 3]
        - order_masks: shape [B, S, 13k]
        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of shape [B, 1]

        Returns:
        - order_idxs [B, S]: idx of sampled orders
        - order_scores [B, S, 13k]: masked pre-softmax logits of each order
        """
        device = next(self.parameters()).device

        self.lstm.flatten_parameters()

        enc = self.encoder(x_bo, x_po, power_emb, season_emb)  # [B, 81, 240]
        avg_enc = torch.mean(enc, dim=1, keepdim=False)  # [B, enc_size]

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
            order_mask = order_masks[:, step, :]  # [B, 13k]
            lstm_input = torch.cat((avg_enc, order_emb), dim=1).unsqueeze(1)  # [B, 1, 320]
            out, hidden = self.lstm(lstm_input, hidden)
            order_scores = self.lstm_out_linear(out.squeeze(1))  # [B, 13k]
            all_order_scores.append(order_scores)

            # masked softmax to choose order_idxs
            # N.B. if an entire row is masked out (probaby during an <EOS>
            # token) then unmask it, or the sampling will crash. The loss for
            # that row will be masked out later, so it doesn't matter
            order_mask[~torch.any(order_mask, dim=1)] = 1
            order_scores[~order_mask] = float("-inf")
            order_idxs = Categorical(logits=order_scores / temperature).sample()
            all_order_idxs.append(order_idxs)
            order_emb = self.order_embedding(order_idxs).squeeze(1)

        return torch.stack(all_order_idxs, dim=1), torch.stack(all_order_scores, dim=1)


class SimpleDipNetDecoder(nn.Module):
    def __init__(self, enc_size, orders_vocab_size):
        super().__init__()
        self.linear = nn.Linear(enc_size, orders_vocab_size)

    def forward(self, x):
        """x should have shape [B, 81, enc_size]"""
        x = torch.mean(x, dim=1, keepdim=False)  # [B, enc_size]
        return self.linear(x)


class DipNetEncoder(nn.Module):
    def __init__(
        self,
        board_state_size,  # 35
        prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 7 (one-hot)
        season_emb_size,  # 3 (one-hot)
        num_blocks,  # 16
        A,  # 81x81
    ):
        super().__init__()

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
                )
            )

    def forward(self, x_bo, x_po, power_emb, season_emb):
        y_bo = x_bo
        for block in self.board_blocks:
            y_bo = block(y_bo, power_emb, season_emb)

        y_po = x_po
        for block in self.prev_orders_blocks:
            y_po = block(y_po, power_emb, season_emb)

        state_emb = torch.cat([y_bo, y_po], -1)
        return state_emb


class DipNetBlock(nn.Module):
    def __init__(self, in_size, out_size, power_emb_size, season_emb_size, A, residual=True):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A)
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
    def __init__(self, in_size, out_size, A):
        super().__init__()
        """
        A -> (81, 81)
        """
        self.A = nn.Parameter(A)
        self.W = nn.Linear(in_size, out_size)

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """
        Ax = torch.matmul(self.A, x)
        return self.W(Ax)


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
