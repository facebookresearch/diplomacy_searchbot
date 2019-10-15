import torch
import torch.nn.functional as F
from torch import nn


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
    ):
        super().__init__()
        self.encoder = DipNetEncoder(
            board_state_size,
            prev_orders_size,
            inter_emb_size,
            power_emb_size,
            season_emb_size,
            num_blocks,
            A,
        )
        self.decoder = SimpleDipNetDecoder(81 * inter_emb_size * 2, orders_vocab_size)

    def forward(self, x_bo, x_po, power_emb, season_emb):
        enc = self.encoder(x_bo, x_po, power_emb, season_emb)
        enc = enc.reshape(enc.shape[0], -1)
        y = self.decoder(enc)
        return y


class SimpleDipNetDecoder(nn.Module):
    def __init__(self, encoder_out_size, orders_vocab_size):
        super().__init__()
        self.linear = nn.Linear(encoder_out_size, orders_vocab_size)

    def forward(self, x):
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
        self.A = A
        self.W = nn.Linear(in_size, out_size)

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """
        return self.W(torch.matmul(self.A, x))


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
