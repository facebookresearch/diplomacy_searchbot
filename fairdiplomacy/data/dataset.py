import diplomacy
import joblib
import logging
import torch
from itertools import combinations, product
from typing import Union, List, Optional

from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
from fairdiplomacy.utils.tensorlist import TensorList


ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(ORDER_VOCABULARY)}
MAX_VALID_LEN = get_order_vocabulary_idxs_len()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        game_json_paths: List[str],
        debug_only_opening_phase=False,
        fill_missing_orders=False,
        n_jobs=20,
    ):
        self.game_json_paths = game_json_paths
        assert not debug_only_opening_phase, "FIXME"

        encoded_games = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(encode_game)(p, fill_missing_orders=fill_missing_orders)
            for p in game_json_paths
        )

        logging.info(f"Got {len(encoded_games)} games")

        encoded_games = [g for g in encoded_games if g[-1][0].any()]
        logging.info(f"{len(encoded_games)} games had data for at least one power")

        game_idxs, phase_idxs, power_idxs, x_idxs = [], [], [], []
        x_idx = 0
        for game_idx, encoded_game in enumerate(encoded_games):
            for phase_idx, valid_power_idxs in enumerate(encoded_game[-1]):
                assert valid_power_idxs.nelement() == 7
                for power_idx in valid_power_idxs.nonzero()[:, 0]:
                    game_idxs.append(game_idx)
                    phase_idxs.append(phase_idx)
                    power_idxs.append(power_idx)
                    x_idxs.append(x_idx)
                x_idx += 1

        self.game_idxs = torch.tensor(game_idxs, dtype=torch.long)
        self.phase_idxs = torch.tensor(phase_idxs, dtype=torch.long)
        self.power_idxs = torch.tensor(power_idxs, dtype=torch.long)
        self.x_idxs = torch.tensor(x_idxs, dtype=torch.long)

        # now collate the data into giant tensors!
        self.encoded_games = [_cat(x) for x in zip(*encoded_games)]

        self.num_games = len(encoded_games)
        self.num_phases = len(self.encoded_games[0])
        self.num_elements = len(self.x_idxs)

        for i, e in enumerate(self.encoded_games):
            if isinstance(e, TensorList):
                assert len(e) == self.num_phases * len(POWERS) * MAX_SEQ_LEN
            else:
                assert len(e) == self.num_phases

    def stats_str(self):
        return f"Dataset: {self.num_games} games, {self.num_phases} phases, and {self.num_elements} elements."

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)

        assert isinstance(idx, torch.Tensor) and idx.dtype == torch.long
        assert idx.max() < self.num_elements

        x_idx = self.x_idxs[idx]
        power_idx = self.power_idxs[idx]

        fields = [x[x_idx] for x in self.encoded_games[:-1]]

        # unpack the possible_actions, insert it into fields[6]
        possible_actions_idx = ((x_idx * len(POWERS) + power_idx) * MAX_SEQ_LEN).unsqueeze(
            1
        ) + torch.arange(MAX_SEQ_LEN).unsqueeze(0)
        x_possible_actions = self.encoded_games[6][possible_actions_idx.view(-1)]
        x_possible_actions_padded = x_possible_actions.to_padded(total_length=MAX_VALID_LEN)
        fields[6] = x_possible_actions_padded.view(len(idx), MAX_SEQ_LEN, MAX_VALID_LEN)

        assert (
            len(fields) == 9
        ), "If you've changed the fields, make sure to fix all the changed indices"

        # for these fields we need to select out the correct power
        for f in [2, 7, 8]:
            fields[f] = fields[f][torch.arange(len(fields[f])), power_idx]

        # cast fields
        for i in range(len(fields) - 4):
            fields[i] = fields[i].to(torch.float32)
        for i in [6, 8]:
            fields[i] = fields[i].to(torch.long)

        return fields

    def __len__(self):
        return self.num_elements


def encode_game(
    game: Union[str, diplomacy.Game], only_with_min_final_score=7, fill_missing_orders=False
):
    """
    Arguments:
    - game: diplomacy.Game object
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.
    - fill_missing_orders: if True, missing orders for orderable locs will be
      explicitly set as H or D (depending on the phase). If False, powers with
      missing orders will be marked invalid.

    Return: tuple of tensors
    L is game length, P is # of powers above min_final_score
    [0] board_state: shape=(L, 81, 35)
    [1] prev_orders: shape=(L, 81, 40)
    [2] power: shape=(L, 7, 7)
    [3] season: shape=(L, 3)
    [4] in_adj_phase: shape=(L, 1)
    [5] final_scores: shape=(L, 7)
    [6] possible_actions: TensorList shape=(L x 7, 17 x 469)
    [7] loc_idxs: shape=(L, 7, 81), int8
    [8] actions: shape=(L, 7, 17) int order idxs
    [9] valid_power_idxs: shape=(L, 7) bool mask of valid powers at each phase
    """

    if isinstance(game, str):
        logging.debug(f"Encoding {game}")
        game = diplomacy.utils.export.load_saved_games_from_disk(game)[0]

    num_phases = len(game.state_history)
    phase_encodings = [
        encode_phase(game, phase_idx, only_with_min_final_score, fill_missing_orders)
        for phase_idx in range(num_phases)
    ]

    stacked_encodings = [_stack(x) for x in zip(*phase_encodings)]

    return stacked_encodings


def encode_phase(
    game, phase_idx: int, only_with_min_final_score: Optional[int], fill_missing_orders=False
):
    """
    Arguments:
    - game: diplomacy.Game object
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.
    - fill_missing_orders: if True, missing orders for orderable locs will be
      explicitly set as H or D (depending on the phase). If False, powers with
      missing orders will be marked invalid.

    Return: tuple of tensors
    - board_state: shape=(81, 35)
    - prev_orders: shape=(81, 40)
    - power: shape=(7, 7)
    - season: shape=(3,)
    - in_adj_phase: shape=(1,)
    - final_scores: shape=(7,) int8
    - possible_actions: Tensorlist shape=(7 x 17, 469)
    - loc_idxs: shape=(7, 81), int8
    - actions: shape=(7, 17) int order idxs
    - valid_power_idxs: shape=(7,) bool mask of valid powers at this phase
    """
    phase_name = str(list(game.state_history.keys())[phase_idx])
    phase_state = game.state_history[phase_name]

    # encode board state
    x_board_state = board_state_to_np(game.state_history[phase_name])
    x_board_state = torch.from_numpy(x_board_state).to(bool)

    # encode prev movement orders
    prev_move_phases = [
        p
        for i, p in enumerate(game.get_phase_history())
        if i < phase_idx and str(p.name).endswith("M")
    ]
    if len(prev_move_phases) > 0:
        move_phase = prev_move_phases[-1]
        x_prev_orders = prev_orders_to_np(move_phase)
        x_prev_orders = torch.from_numpy(x_prev_orders).to(bool)
    else:
        x_prev_orders = torch.zeros(81, 40, dtype=torch.bool)

    # encode powers one-hot
    x_power = torch.eye(len(POWERS), dtype=torch.bool)

    # encode season 1-hot
    x_season = torch.zeros(len(SEASONS), dtype=torch.bool)
    season_idx = [s[0] for s in SEASONS].index(phase_name[0])
    x_season[season_idx] = 1

    # encode adjustment phase
    x_in_adj_phase = torch.zeros(1, dtype=torch.bool).fill_(phase_name[-1] == "A")

    # encode final scores
    y_final_scores = torch.zeros(1, 7, dtype=torch.int8)
    final_centers = game.get_state()["centers"]
    for power_i, power in enumerate(POWERS):
        y_final_scores[:, power_i] = len(final_centers.get(power, []))

    # encode possible actions
    tmp_game = diplomacy.Game(map_name=game.map_name)
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
    valid_power_idxs = torch.ones(len(POWERS), dtype=torch.bool)
    y_actions = torch.zeros(len(POWERS), MAX_SEQ_LEN, dtype=torch.int32).fill_(EOS_IDX)
    for power_i, power in enumerate(POWERS):
        orders = game.order_history[phase_name].get(power, [])
        order_idxs = []

        if any(len(order.split()) < 3 for order in orders):
            # skip over power with unparseably short order
            valid_power_idxs[power_i] = 0
            continue
        elif any(order.split()[2] == "B" for order in orders):
            # builds are represented as a single ;-separated order
            assert all(order.split()[2] == "B" for order in orders), orders
            order = ";".join(sorted(orders))
            try:
                order_idx = ORDER_VOCABULARY_TO_IDX[order]
            except KeyError:
                logging.warning(f"Invalid build order: {order}")
                valid_power_idxs[power_i] = 0
                continue
            order_idxs.append(order_idx)
        else:
            for order in orders:
                try:
                    order_idxs.append(smarter_order_index(order))
                except KeyError:
                    # skip over invalid orders; we may fill them in later
                    continue

            # maybe fill in Hold/Disband for invalid or missing orders
            all_locs = {LOCS[x] for x in (x_loc_idxs[power_i] != -1).nonzero().squeeze(1)}
            filled_locs = {ORDER_VOCABULARY[x].split()[1].split("/")[0] for x in order_idxs}
            unfilled_locs = all_locs - filled_locs
            if fill_missing_orders and not x_in_adj_phase[0] and unfilled_locs:
                try:
                    for loc in unfilled_locs:
                        unit = next(unit for unit in phase_state["units"][power] if loc in unit)
                        if "*" in unit:
                            # FIXME: unit may be e.g. "F STP" when it should be "F STP/NC", and this will fail
                            order_idxs.append(smarter_order_index(f"{unit.strip('*')} D"))
                        else:
                            order_idxs.append(smarter_order_index(f"{unit} H"))
                except Exception:
                    logging.exception("Error filling missing orders")
                    valid_power_idxs[power_i] = 0
                    continue

        # sort by topo order
        order_idxs.sort(key=lambda idx: LOCS.index(ORDER_VOCABULARY[idx].split()[1]))
        for i, order_idx in enumerate(order_idxs):
            y_actions[power_i, i] = order_idx

    # filter away powers that have no orders
    valid_power_idxs &= torch.any(y_actions != EOS_IDX, dim=1)
    assert valid_power_idxs.ndimension() == 1

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
        valid_power_idxs[power_idx] = 0
        # logging.warning(
        #     "Order not possible: {} {} {} {}".format(
        #         y_actions[power_idx],
        #         game.order_history[phase_name].get(POWERS[power_idx]),
        #         x_possible_actions[power_idx, :5, :50],
        #         phase_state,
        #         all_possible_orders,
        #     )
        # )

    # Maybe filter away powers that don't finish with enough SC.
    # If all players finish with fewer SC, include everybody.
    # cf. get_top_victors() in mila's state_space.py
    if only_with_min_final_score is not None:
        final_score = {k: len(v) for k, v in game.get_state()["centers"].items()}
        if max(final_score.values()) >= only_with_min_final_score:
            for i, power in enumerate(POWERS):
                if final_score.get(power, 0) < only_with_min_final_score:
                    valid_power_idxs[i] = 0

    x_possible_actions = TensorList.from_padded(
        x_possible_actions.view(len(POWERS) * MAX_SEQ_LEN, MAX_VALID_LEN)
    )

    return (
        x_board_state,
        x_prev_orders,
        x_power,
        x_season,
        x_in_adj_phase,
        y_final_scores,
        x_possible_actions,
        x_loc_idxs,
        y_actions,
        valid_power_idxs,
    )


def get_valid_orders_impl(power, all_possible_orders, all_orderable_locations, game_state):
    """Return a list of valid orders

    Returns:
    - a [1, 17, 469] int tensor of valid move indexes (padded with 0)
    - a [1, 81] int8 tensor of orderable locs, described below
    - the actual length of the sequence == the number of orders to submit, <= 17

    loc_idxs:
    - not adj phase: X[i] = s if LOCS[i] is orderable at step s (0 <= s < 17), -1 otherwise
    - in adj phase: X[i] = -2 if LOCS[i] is orderable this phase, -1 otherwise
    """
    all_order_idxs = torch.zeros(1, MAX_SEQ_LEN, MAX_VALID_LEN, dtype=torch.int32)
    loc_idxs = torch.zeros(1, len(LOCS), dtype=torch.int8).fill_(-1)

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
        # build phase: represented as a single ;-separated string combining all
        # units to be built.
        orders = [
            ";".join(sorted(x))
            for c in combinations([all_possible_orders[loc] for loc in orderable_locs], n_builds)
            for x in product(*c)
        ]
        try:
            order_idxs = torch.tensor(
                [ORDER_VOCABULARY_TO_IDX[x] for x in orders], dtype=torch.int32
            )
        except KeyError:
            import ipdb

            ipdb.set_trace()
        all_order_idxs[0, :1, : len(order_idxs)] = order_idxs.unsqueeze(0)
        loc_idxs[0, [LOCS.index(l) for l in orderable_locs]] = -2
        return all_order_idxs, loc_idxs, n_builds

    if n_builds < 0:
        # disbands: all possible disband orders, up to the number of required disbands
        n_disbands = -n_builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        all_order_idxs[0, :n_disbands, : len(order_idxs)] = order_idxs.unsqueeze(0)
        loc_idxs[0, [LOCS.index(l) for l in orderable_locs]] = -2
        return all_order_idxs, loc_idxs, n_disbands

    # move phase: iterate through orderable_locs in topo order
    for i, loc in enumerate(orderable_locs):
        orders, order_idxs = filter_orders_in_vocab(all_possible_orders[loc])
        all_order_idxs[0, i, : len(order_idxs)] = order_idxs
        loc_idxs[0, LOCS.index(loc)] = i

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


def _cat(x):
    return TensorList.cat(x) if isinstance(x[0], TensorList) else torch.cat(x)


def _stack(x):
    return TensorList.cat(x) if isinstance(x[0], TensorList) else torch.stack(x)
