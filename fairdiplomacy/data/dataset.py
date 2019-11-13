import diplomacy
import logging
import numpy as np
import torch

from fairdiplomacy.data.get_game_lengths import get_all_game_lengths
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary, EOS_IDX
from fairdiplomacy.models.dipnet.standard_topo_locs import STANDARD_TOPO_LOCS


ORDER_VOCABULARY = get_order_vocabulary()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, game_json_paths, game_json_lengths=None):
        self.game_json_paths = game_json_paths

        if game_json_lengths is not None:
            self.game_json_lengths = game_json_lengths
        else:
            logging.info("Calculating dataset size...")
            self.game_json_lengths = get_all_game_lengths(game_json_paths)

        self.len = sum(self.game_json_lengths)
        self.cumsum = np.cumsum(self.game_json_lengths)

    def __getitem__(self, idx):
        game_idx = np.searchsorted(self.cumsum, idx, side="right")
        phase_idx = idx - (self.cumsum[game_idx - 1] if game_idx > 0 else 0)

        path = self.game_json_paths[game_idx]
        game = diplomacy.utils.export.load_saved_games_from_disk(path)[0]
        try:
            return encode_phase(game, phase_idx)
        except Exception:
            logging.exception(
                "Skipping dataset game_idx={} path={} phase_idx={}".format(
                    game_idx, path, phase_idx
                )
            )
            return None

    def __len__(self):
        return self.len


def encode_phase(game, phase_idx, only_winners=True):
    """Return five tensors:

    board_state: shape=(7, 81, 35)
    prev_orders: shape=(7, 81, 40)
    power: shape=(7, 7)
    season: shape=(7, 3)
    actions: shape=(7, 17) int order idxs
    """
    phase = str(list(game.state_history.keys())[phase_idx])

    # encode board state
    x_board_state = board_state_to_np(game.state_history[phase])
    x_board_state = torch.from_numpy(x_board_state).repeat(len(POWERS), 1, 1)

    # encode prev movement orders
    # TODO: does MILA condition on last move orders, or last orders of any kind?
    prev_move_phases = [
        p
        for i, p in enumerate(game.state_history.keys())
        if i < phase_idx and str(p).endswith("M")
    ]
    if len(prev_move_phases) > 0:
        move_phase = prev_move_phases[-1]
        x_prev_orders = prev_orders_to_np(
            game.state_history[move_phase], game.order_history[move_phase]
        )
        x_prev_orders = torch.from_numpy(x_prev_orders).repeat(len(POWERS), 1, 1)
    else:
        x_prev_orders = torch.zeros(len(POWERS), 81, 40)

    # encode powers 1-hot
    x_power = torch.eye(len(POWERS))

    # encode season 1-hot
    x_season = torch.zeros(len(POWERS), len(SEASONS))
    season_idx = [s[0] for s in SEASONS].index(phase[0])
    x_season[:, season_idx] = 1

    # encode actions
    y_actions = torch.zeros(len(POWERS), MAX_SEQ_LEN, dtype=torch.int64).fill_(EOS_IDX)
    for power_i, power in enumerate(POWERS):
        orders = game.order_history[phase].get(power, [])
        order_idxs = []
        for order in orders:
            try:
                order_idxs.append(smarter_order_index(order))
            except ValueError:
                # logging.warn(
                #     'Order "{}" not in vocab, game_id={} phase_idx={} phase={}'.format(
                #         order, game.game_id, phase_idx, phase
                #     )
                # )
                continue

        # sort by topo order
        order_idxs.sort(key=lambda idx: STANDARD_TOPO_LOCS.index(ORDER_VOCABULARY[idx].split()[1]))
        for i, order_idx in enumerate(order_idxs):
            y_actions[power_i, i] = order_idx

    if only_winners:
        final_score = {k: len(v) for k, v in game.get_state()["centers"].items()}
        winner_idxs = [i for i, power in enumerate(POWERS) if final_score.get(power, 0) > 0]
        return tuple(
            t[winner_idxs] for t in (x_board_state, x_prev_orders, x_power, x_season, y_actions)
        )
    else:
        return x_board_state, x_prev_orders, x_power, x_season, y_actions


def smarter_order_index(order):
    try:
        return ORDER_VOCABULARY.index(order)
    except ValueError:
        for suffix in ["/NC", "/EC", "/SC", "/WC"]:
            order = order.replace(suffix, "")
        return ORDER_VOCABULARY.index(order)


def collate_fn(items):
    # items is a list of 5-tuples, one tuple from each dataset
    # lsts is a 5-tuple of lists of tensors
    lsts = zip(*(x for x in items if x is not None))
    try:
        return tuple(torch.cat(lst, dim=0) for lst in lsts)
    except:
        logging.exception([tuple(t.shape for t in item) for item in items])
        raise


if __name__ == "__main__":
    game = diplomacy.utils.export.load_saved_games_from_disk("out/game_3232.json")[0]
    for t in encode_phase(game, 3):
        print(t.shape)
