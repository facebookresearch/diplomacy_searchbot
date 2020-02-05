import copy
import diplomacy
import logging
import numpy as np
import torch

from fairdiplomacy.data.get_game_lengths import get_all_game_lengths
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)


ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(ORDER_VOCABULARY)}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, game_json_paths, game_json_lengths=None, debug_only_opening_phase=False):
        self.game_json_paths = game_json_paths
        self.debug_only_opening_phase = debug_only_opening_phase

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
            if self.debug_only_opening_phase:
                return encode_phase(game, 0)
            else:
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


def encode_phase(game, phase_idx, only_with_min_final_score=7):
    """
    Arguments:
    - game: diplomacy.Game object
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.

    Return: tuple of eight tensors
    - board_state: shape=(7, 81, 35)
    - prev_orders: shape=(7, 81, 40)
    - power: shape=(7, 7)
    - season: shape=(7, 3)
    - in_adj_phase: shape=(7,)
    - possible_actions: shape=(7, 17, 469), pad=-1
    - loc_idxs: shape=(7, 17), 0 <= idx < 81, pad=-1
    - actions: shape=(7, 17) int order idxs
    """
    phase_name = str(list(game.state_history.keys())[phase_idx])
    phase_state = game.state_history[phase_name]

    # encode board state
    x_board_state = board_state_to_np(phase_state)
    x_board_state = torch.from_numpy(x_board_state).repeat(len(POWERS), 1, 1)

    # encode prev movement orders
    prev_move_phases = [
        p
        for i, p in enumerate(game.get_phase_history())
        if i < phase_idx and str(p.name).endswith("M")
    ]
    if len(prev_move_phases) > 0:
        move_phase = prev_move_phases[-1]
        x_prev_orders = prev_orders_to_np(move_phase)
        x_prev_orders = torch.from_numpy(x_prev_orders).repeat(len(POWERS), 1, 1)
    else:
        x_prev_orders = torch.zeros(len(POWERS), 81, 40)

    # encode powers 1-hot
    x_power = torch.eye(len(POWERS))

    # encode season 1-hot
    x_season = torch.zeros(len(POWERS), len(SEASONS))
    season_idx = [s[0] for s in SEASONS].index(phase_name[0])
    x_season[:, season_idx] = 1

    # encode adjustment phase
    x_in_adj_phase = torch.zeros(len(POWERS), dtype=torch.bool).fill_(phase_name[-1] == "A")

    # encode possible actions
    tmp_game = copy.deepcopy(game)
    tmp_game.set_state(phase_state)
    all_possible_orders = tmp_game.get_all_possible_orders()
    all_orderable_locations = tmp_game.get_orderable_locations()
    x_possible_actions, x_loc_idxs = [
        torch.stack(ts).squeeze(1)
        for ts in zip(
            *[
                get_valid_orders_impl(
                    power, all_possible_orders, all_orderable_locations, phase_state
                )[:2]
                for power in POWERS
            ]
        )
    ]

    # encode actions
    y_actions = torch.zeros(len(POWERS), MAX_SEQ_LEN, dtype=torch.int64).fill_(EOS_IDX)
    for power_i, power in enumerate(POWERS):
        orders = game.order_history[phase_name].get(power, [])
        order_idxs = []
        for order in orders:
            try:
                order_idxs.append(smarter_order_index(order))
            except KeyError:
                continue

        # fill in Hold for invalid or missing orders
        if not x_in_adj_phase[0]:
            filled_locs = set(ORDER_VOCABULARY[x].split()[1].split("/")[0] for x in order_idxs)
            unfilled_locs = {LOCS[x] for _, x in x_loc_idxs[power_i, :, :].nonzero()} - filled_locs
            for loc in unfilled_locs:
                unit = next(unit for unit in phase_state["units"][power] if loc in unit)
                if "*" in unit:
                    order_idxs.append(smarter_order_index(f"{unit.strip('*')} D"))
                else:
                    order_idxs.append(smarter_order_index(f"{unit} H"))

        # sort by topo order
        order_idxs.sort(key=lambda idx: LOCS.index(ORDER_VOCABULARY[idx].split()[1]))
        for i, order_idx in enumerate(order_idxs):
            y_actions[power_i, i] = order_idx

    # filter away powers that have no orders
    power_idxs = torch.nonzero(torch.any(y_actions != EOS_IDX, dim=1)).squeeze(1).tolist()

    # filter away powers whose orders are not in valid_orders
    # most common reasons why this happens:
    # - actual garbage orders (e.g. moves between non-adjacent locations)
    # - too many orders (e.g. three build orders with only two allowed builds)
    for power_idx in (
        ~(
            ((y_actions.unsqueeze(2) == x_possible_actions) | (y_actions.unsqueeze(2) == 0))
            .any(dim=2)
            .all(dim=1)
        )
    ).nonzero():
        power_idxs = [idx for idx in power_idxs if idx != power_idx]  # rm from output
        # logging.warning(
        #     "Order not possible: {} {} {} {}".format(
        #         y_actions[power_idx],
        #         game.order_history[phase_name].get(POWERS[power_idx]),
        #         x_possible_actions[power_idx, :5, :50],
        #         phase_state,
        #         all_possible_orders,
        #     )
        # )

    # maybe filter away powers that don't finish with enough SC
    if only_with_min_final_score is not None:
        final_score = {k: len(v) for k, v in game.get_state()["centers"].items()}
        winner_idxs = [
            i
            for i, power in enumerate(POWERS)
            if final_score.get(power, 0) >= only_with_min_final_score
        ]
        power_idxs = [idx for idx in power_idxs if idx in winner_idxs]

    return tuple(
        t[power_idxs]
        for t in (
            x_board_state,
            x_prev_orders,
            x_power,
            x_season,
            x_in_adj_phase,
            x_possible_actions,
            x_loc_idxs,
            y_actions,
        )
    )


def get_valid_orders_impl(power, all_possible_orders, all_orderable_locations, game_state):
    """Return a boolean mask of valid orders

    Returns:
    - a [1, 17, 469] int tensor of valid move indexes (padded with -1)
    - a [1, 17, 81] bool tensor of orderable locs
    - the actual length of the sequence == the number of orders to submit, <= 17
    """
    all_order_idxs = torch.zeros(
        1, MAX_SEQ_LEN, get_order_vocabulary_idxs_len(), dtype=torch.int32
    )
    loc_idxs = torch.zeros(1, MAX_SEQ_LEN, len(LOCS), dtype=torch.bool)

    if power not in all_orderable_locations:
        return all_order_idxs, loc_idxs, 0

    # strip "WAIVE" from possible orders
    all_possible_orders = {
        k: [x for x in v if x != "WAIVE"] for k, v in all_possible_orders.items()
    }

    # sort by index in LOCS using the right coastal variant!
    # all_orderable_locations may give the root loc even if the possible orders
    # are from a coastal fleet
    orderable_locs = sorted(
        all_orderable_locations[power],
        key=lambda loc: LOCS.index(all_possible_orders[loc][0].split()[1]),
    )

    power_possible_orders = [x for loc in orderable_locs for x in all_possible_orders[loc]]
    n_builds = game_state["builds"][power]["count"]

    if n_builds > 0:
        # build phase: all possible build orders, up to the number of allowed builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        all_order_idxs[0, :n_builds, : len(order_idxs)] = order_idxs.unsqueeze(0)
        loc_idxs[0, :n_builds, [LOCS.index(l) for l in orderable_locs]] = 1
        return all_order_idxs, loc_idxs, n_builds

    if n_builds < 0:
        # disbands: all possible disband orders, up to the number of required disbands
        n_disbands = -n_builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        all_order_idxs[0, :n_builds, : len(order_idxs)] = order_idxs.unsqueeze(0)
        loc_idxs[0, :n_builds, [LOCS.index(l) for l in orderable_locs]] = 1
        return all_order_idxs, loc_idxs, n_disbands

    # move phase: iterate through orderable_locs in topo order
    for i, loc in enumerate(orderable_locs):
        orders, order_idxs = filter_orders_in_vocab(all_possible_orders[loc])
        all_order_idxs[0, i, : len(order_idxs)] = order_idxs
        loc_idxs[0, i, LOCS.index(loc)] = 1

    return all_order_idxs, loc_idxs, len(orderable_locs)


def filter_orders_in_vocab(orders):
    """Return the subset of orders that are found in the vocab, and their idxs"""
    ret, idxs = [], []
    for order in orders:
        try:
            idx = smarter_order_index(order)
            ret.append(order)
            idxs.append(idx)
        except KeyError:
            continue
    return ret, torch.tensor(idxs, dtype=torch.int32)


def smarter_order_index(order):
    try:
        return ORDER_VOCABULARY_TO_IDX[order]
    except KeyError:
        for suffix in ["/NC", "/EC", "/SC", "/WC"]:
            order = order.replace(suffix, "")
        return ORDER_VOCABULARY_TO_IDX[order]


def collate_fn(items):
    # items is a list of 6-tuples, one tuple from each dataset
    # lsts is a 6-tuple of lists of tensors
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
