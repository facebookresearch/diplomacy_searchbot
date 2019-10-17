import diplomacy
import logging
import numpy as np
import torch

from fairdiplomacy.data.get_game_lengths import get_all_game_lengths
from fairdiplomacy.models.consts import SEASONS, POWERS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary


ORDER_VOCABULARY = get_order_vocabulary()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, game_json_paths):
        self.game_json_paths = game_json_paths

        logging.info("Calculating dataset size...")
        self.game_json_lengths = get_all_game_lengths(game_json_paths)
        self.len = sum(self.game_json_lengths)
        self.cumsum = np.cumsum(self.game_json_lengths)

    def __getitem__(self, idx):
        game_idx = np.searchsorted(self.cumsum, idx, side="right")
        phase_idx = idx - (self.cumsum[game_idx - 1] if game_idx > 0 else 0)

        path = self.game_json_paths[game_idx]
        game = diplomacy.utils.export.load_saved_games_from_disk(path)[0]
        return encode_phase(game, phase_idx)

    def __len__(self):
        return self.len


def encode_phase(game, phase_idx):
    """Return five tensors:

    board_state: shape=(7, 81, 35)
    prev_orders: shape=(7, 81, 40)
    power: shape=(7, 7)
    season: shape=(7, 3)
    actions: shape=(7, 13030)
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
        x_prev_orders = prev_orders_to_np(
            game.state_history[phase], game.order_history[prev_move_phases[-1]]
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
    y_actions = torch.zeros(len(POWERS), len(ORDER_VOCABULARY))
    for power_i, power in enumerate(POWERS):
        for order in game.order_history[phase][power]:
            try:
                action_idx = smarter_order_index(order)
            except ValueError:
                logging.warn('Order "{}" not in vocab'.format(order))
                continue
            y_actions[power_i, action_idx] = 1

    return x_board_state, x_prev_orders, x_power, x_season, y_actions


def encode_game(game):
    """Return five tensors:

    board_state: shape=(L, 81, 35)
    prev_orders: shape=(L, 81, 40)
    power: shape=(L, 7)
    season: shape=(L, 3)
    actions: shape=(L, 13030)

    where L is the number of actions in the game
    """
    L = len(game.state_history) * len(POWERS)

    # return values
    x_board_state = torch.zeros(L, 81, 35)
    x_prev_orders = torch.zeros(L, 81, 40)
    x_power = torch.zeros(L, 7)
    x_season = torch.zeros(L, 3)
    y_actions = torch.zeros(L, len(ORDER_VOCABULARY))

    i = 0
    prev_orders_np = np.zeros((81, 40))
    for phase, state in game.state_history.items():
        phase = str(phase)
        orders_by_power = game.order_history[phase]
        state_np = board_state_to_np(state)
        season_idx = [s[0] for s in SEASONS].index(phase[0])

        for power_idx, power in enumerate(POWERS):
            # set values in returned tensors
            x_board_state[i, :, :] = torch.from_numpy(state_np)
            x_prev_orders[i, :, :] = torch.from_numpy(prev_orders_np)
            x_power[i, power_idx] = 1
            x_season[i, season_idx] = 1
            for order in orders_by_power[power]:
                try:
                    action_idx = smarter_order_index(order)
                except ValueError:
                    # Skip i += 1, so this row is overwritten
                    logging.warn('Order "{}" not in vocab'.format(order))
                    continue
                y_actions[i, action_idx] = 1

            i += 1

        # TODO: does MILA condition on last move orders, or last orders of any kind?
        if phase.endswith("M"):
            prev_orders_np = prev_orders_to_np(state, orders_by_power)

    if i != L:
        logging.warn("Skipped {} bad orders this game".format(L - i))

    return x_board_state[:i], x_prev_orders[:i], x_power[:i], x_season[:i], y_actions[:i]


def smarter_order_index(order):
    try:
        return ORDER_VOCABULARY.index(order)
    except ValueError:
        for suffix in ["/NC", "/EC", "/SC", "/WC"]:
            order = order.replace(suffix, "")
        return ORDER_VOCABULARY.index(order)


def collate_fn(items):
    # Items is a list of 5-tuples, one tuple from each dataset
    return tuple(torch.cat(tensors) for tensors in zip(*items))


if __name__ == "__main__":
    game = diplomacy.utils.export.load_saved_games_from_disk("out/game_3232.json")[0]
    for t in encode_phase(game, 3):
        print(t.shape)
