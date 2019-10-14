import diplomacy
import logging
import numpy as np
import torch

from fairdiplomacy.models.consts import SEASONS, POWERS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary

ORDER_VOCABULARY = get_order_vocabulary()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, game_json_paths):
        self.game_json_paths = game_json_paths

    def __getitem__(self, idx):
        path = self.game_json_paths[idx]
        game = diplomacy.utils.export.load_saved_games_from_disk(path)[0]
        return encode_game(game)

    def __len__(self):
        return len(self.game_json_paths)


def collate_fn(items):
    # Items is a list of 5-tuples, one tuple from each dataset
    return tuple(torch.cat(tensors) for tensors in zip(*items))

def encode_game(game):
    """Return five tensors:

    board_state: shape=(L, 81, 35)
    prev_orders: shape=(L, 81, 40)
    power: shape=(L, 7)
    season: shape=(L, 3)
    actions: shape=(L, 13030)

    where L is the number of action in the game
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
                    logging.warn("Order \"{}\" not in vocab".format(order))
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


if __name__ == "__main__":
    game = diplomacy.utils.export.load_saved_games_from_disk("out/game_3232.json")[0]
    for t in encode_game(game):
        print(t.shape)
