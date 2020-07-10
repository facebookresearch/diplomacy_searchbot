import diplomacy
import joblib
import json
import logging
import os
import torch
from itertools import combinations, product
from typing import Any, Dict, Union, List, Optional, Tuple

from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state
from fairdiplomacy.utils.tensorlist import TensorList
from fairdiplomacy.utils.sampling import sample_p_dict


ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(ORDER_VOCABULARY)}
LOC_IDX = {loc: idx for idx, loc in enumerate(LOCS)}
MAX_VALID_LEN = get_order_vocabulary_idxs_len()


class DataFields(dict):
    BOOL_STORAGE_FIELDS = [
        "x_board_state",
        "x_prev_state",
    ]

    def select(self, idx):
        return DataFields({k: v[idx] for k, v in self.items()})

    @classmethod
    def cat(cls, L: list):
        if len(L) > 0:
            return cls({k: _cat([x[k] for x in L]) for k in L[0]})
        else:
            return cls()

    @classmethod
    def stack(cls, L: list, dim: int = 0):
        if len(L) > 0:
            return cls({k: torch.stack([x[k] for x in L], dim) for k in L[0]})
        else:
            return cls()

    def to_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.bool)
        return self

    def from_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.float32)
        return self


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        game_ids: List[int],
        data_dir: str,
        game_metadata: Dict[int, Any],
        debug_only_opening_phase=False,
        only_with_min_final_score=7,
        n_jobs=20,
        value_decay_alpha=1.0,
        cf_agent=None,
        n_cf_agent_samples=1,
        min_rating=None,
        exclude_n_holds=-1,
    ):
        self.game_ids = game_ids
        self.data_dir = data_dir
        self.game_metadata = game_metadata
        self.n_cf_agent_samples = n_cf_agent_samples
        self.min_rating = min_rating
        assert not debug_only_opening_phase, "FIXME"

        logging.info(
            f"Building dataset from {len(game_ids)} games, only_with_min_final_score={only_with_min_final_score} value_decay_alpha={value_decay_alpha} cf_agent={cf_agent}"
        )

        torch.set_num_threads(1)
        encoded_games = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(encode_game)(
                f"{data_dir}/game_{game_id}.json",
                only_with_min_final_score=only_with_min_final_score,
                cf_agent=cf_agent,
                n_cf_agent_samples=n_cf_agent_samples,
                value_decay_alpha=value_decay_alpha,
                input_valid_power_idxs=self.get_valid_power_idxs(game_id),
                game_metadata=self.game_metadata[game_id],
                exclude_n_holds=exclude_n_holds,
            )
            for game_id in game_ids
        )
        encoded_games = [
            g for g in encoded_games if g is not None
        ]  # remove "empty" games (e.g. json didn't exist)

        logging.info(f"Found data for {len(encoded_games)} / {len(game_ids)} games")

        encoded_games = [g for g in encoded_games if g["valid_power_idxs"][0].any()]
        logging.info(f"{len(encoded_games)} games had data for at least one power")

        game_idxs, phase_idxs, power_idxs, x_idxs = [], [], [], []
        x_idx = 0
        for game_idx, encoded_game in enumerate(encoded_games):
            for phase_idx, valid_power_idxs in enumerate(encoded_game["valid_power_idxs"]):
                assert valid_power_idxs.nelement() == len(POWERS), (
                    encoded_game["valid_power_idxs"].shape,
                    valid_power_idxs.shape,
                )
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
        self.encoded_games = DataFields.cat(encoded_games)

        self.num_games = len(encoded_games)
        self.num_phases = len(self.encoded_games["x_board_state"]) if self.encoded_games else 0
        self.num_elements = len(self.x_idxs)

        for i, e in enumerate(self.encoded_games.values()):
            if isinstance(e, TensorList):
                assert len(e) == self.num_phases * len(POWERS) * MAX_SEQ_LEN
            else:
                assert len(e) == self.num_phases

    def get_valid_power_idxs(self, game_id):
        return [
            self.game_metadata[game_id][pwr]["logit_rating"] >= self.min_rating for pwr in POWERS
        ]

    def stats_str(self):
        return f"Dataset: {self.num_games} games, {self.num_phases} phases, and {self.num_elements} elements."

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)

        assert isinstance(idx, torch.Tensor) and idx.dtype == torch.long
        assert idx.max() < len(self)

        sample_idxs = idx % self.n_cf_agent_samples
        idx //= self.n_cf_agent_samples

        x_idx = self.x_idxs[idx]
        power_idx = self.power_idxs[idx]

        fields = self.encoded_games.select(x_idx)  # [x[x_idx] for x in self.encoded_games[:-1]]

        # unpack the possible_actions
        possible_actions_idx = ((x_idx * len(POWERS) + power_idx) * MAX_SEQ_LEN).unsqueeze(
            1
        ) + torch.arange(MAX_SEQ_LEN).unsqueeze(0)
        x_possible_actions = self.encoded_games["x_possible_actions"][
            possible_actions_idx.view(-1)
        ]
        x_possible_actions_padded = x_possible_actions.to_padded(
            total_length=MAX_VALID_LEN, padding_value=EOS_IDX
        )
        fields["x_possible_actions"] = x_possible_actions_padded.view(
            len(idx), MAX_SEQ_LEN, MAX_VALID_LEN
        )

        # for these fields we need to select out the correct power
        for f in ("x_power", "x_loc_idxs", "y_actions"):
            fields[f] = fields[f][torch.arange(len(fields[f])), power_idx]

        # for y_actions, select out the correct power
        fields["y_actions"] = (
            fields["y_actions"]
            .gather(1, sample_idxs.view(-1, 1, 1).repeat((1, 1, fields["y_actions"].shape[2])))
            .squeeze(1)
        )

        # cast fields
        for k in fields:
            if k in ("x_possible_actions", "y_actions", "x_prev_orders"):
                fields[k] = fields[k].to(torch.long)
            elif k != "prev_orders":
                fields[k] = fields[k].to(torch.float32)

        return fields.from_storage_fmt_()

    def __len__(self):
        return self.num_elements * self.n_cf_agent_samples


def encode_game(
    game: Union[str, diplomacy.Game],
    only_with_min_final_score=7,
    *,
    cf_agent=None,
    n_cf_agent_samples=1,
    input_valid_power_idxs,
    value_decay_alpha,
    game_metadata,
    exclude_n_holds,
):
    """
    Arguments:
    - game: diplomacy.Game object
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.
    - input_valid_power_idxs: bool tensor, true if power should a priori be included in
      the dataset based on e.g. player rating)

    Return: DataFields dict of tensors
    L is game length, P is # of powers above min_final_score, N is n_cf_agent_samples
    - board_state: shape=(L, 81, 35)
    - prev_state: shape=(L, 81, 35)
    - prev_orders: shape=(L, 2, 100), dtype=long
    - power: shape=(L, 7, 7)
    - season: shape=(L, 3)
    - in_adj_phase: shape=(L, 1)
    - has_press: shape=(L, 1)
    - build_numbers: shape=(L, 7)
    - final_scores: shape=(L, 7)
    - possible_actions: TensorList shape=(L x 7, 17 x 469)
    - loc_idxs: shape=(L, 7, 81), int8
    - actions: shape=(L, 7, N, 17) int order idxs, N=n_cf_agent_samples
    - valid_power_idxs: shape=(L, 7) bool mask of valid powers at each phase
    """

    torch.set_num_threads(1)
    if isinstance(game, str):
        try:
            with open(game) as f:
                j = json.load(f)
            game = Game.from_saved_game_format(j)
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f"Error while loading game at {game}: {e}")
            return None

    num_phases = len(game.state_history)
    logging.info(f"Encoding {game.game_id} with {num_phases} phases")
    phase_encodings = [
        encode_phase(
            game,
            phase_idx,
            only_with_min_final_score=only_with_min_final_score,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            value_decay_alpha=value_decay_alpha,
            input_valid_power_idxs=input_valid_power_idxs,
            exclude_n_holds=exclude_n_holds,
        )
        for phase_idx in range(num_phases)
    ]

    stacked_encodings = DataFields.cat(phase_encodings)

    has_press = torch.zeros(num_phases, 1) + (1 if game_metadata["press_type"] != "NoPress" else 0)
    stacked_encodings["x_has_press"] = has_press

    return stacked_encodings.to_storage_fmt_()


def encode_state(game: diplomacy.Game, phase_idx: Optional[int] = None):
    # encode board state
    phase_history = game.get_phase_history()
    if phase_idx is None:
        state = game.get_state()
        phase_idx = len(phase_history)
        season_char = game.phase[0]
    else:
        phase = phase_history[phase_idx]
        state = phase_history[phase_idx].state
        season_char = phase.name[0]

    x_board_state = torch.from_numpy(board_state_to_np(state)).unsqueeze(0)

    prev_phases = []
    for i in range(phase_idx - 1, -1, -1):
        p = phase_history[i]
        prev_phases.append(p)
        if p.name.endswith("M"):
            break

    x_prev_orders = torch.zeros(1, 2, 100, dtype=torch.long)
    if len(prev_phases) > 0:
        prev_orders, prev_order_locs = [], []
        x_prev_state = torch.from_numpy(board_state_to_np(prev_phases[-1].state)).unsqueeze(0)
        for phase in prev_phases:
            # print(phase.name, phase.orders)
            for pwr in phase.orders:
                prev_orders += [
                    ORDER_VOCABULARY_TO_IDX[o]
                    for o in phase.orders[pwr]
                    if o in ORDER_VOCABULARY_TO_IDX
                ]
                prev_order_locs += [
                    LOC_IDX[o.split()[1]]
                    for o in phase.orders[pwr]
                    if o in ORDER_VOCABULARY_TO_IDX
                ]
        num_orders = len(prev_orders)
        x_prev_orders[0, 0, :num_orders] = torch.tensor(prev_orders, dtype=torch.long)
        x_prev_orders[0, 1, :num_orders] = torch.tensor(prev_order_locs, dtype=torch.long)
    else:
        x_prev_state = torch.zeros(1, 81, 35)

    x_season = torch.zeros(1, len(SEASONS))
    x_season[0, ("S", "F", "W").index(season_char)] = 1

    x_in_adj_phase = torch.zeros(1).fill_(state["name"][-1] == "A")

    builds = state["builds"]
    x_build_numbers = torch.tensor(
        [[builds[p]["count"] if p in builds else 0 for p in POWERS]], dtype=torch.float32
    )

    return DataFields(
        x_board_state=x_board_state,
        x_prev_state=x_prev_state,
        x_prev_orders=x_prev_orders,
        x_season=x_season,
        x_in_adj_phase=x_in_adj_phase,
        x_build_numbers=x_build_numbers,
    )


def encode_phase(
    game,
    phase_idx: int,
    *,
    only_with_min_final_score: Optional[int],
    cf_agent=None,
    n_cf_agent_samples=1,
    value_decay_alpha,
    input_valid_power_idxs,
    exclude_n_holds,
):
    """
    Arguments:
    - game: diplomacy.Game object
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.

    Return: DataFields dict of tensors
    - board_state: shape=(81, 35)
    - prev_state: shape=(81, 35)
    - prev_orders: shape=(2, 100)
    - power: shape=(7, 7)
    - season: shape=(3,)
    - in_adj_phase: shape=(1,)
    - has_press: shape=(L, 1)
    - build_numbers: shape=(7,)
    - final_scores: shape=(7,) int8
    - possible_actions: Tensorlist shape=(7 x 17, 469)
    - loc_idxs: shape=(7, 81), int8
    - actions: shape=(7, N, 17) int order idxs, N=n_cf_agent_samples
    - valid_power_idxs: shape=(7,) bool mask of valid powers at this phase
    """
    phase_name = str(list(game.state_history.keys())[phase_idx])
    phase_state = game.state_history[phase_name]

    data_fields = encode_state(game, phase_idx)

    # encode final scores
    y_final_scores = encode_weighted_sos_scores(game, phase_idx, value_decay_alpha)

    # encode possible actions
    tmp_game = Game.clone_from(game, up_to_phase=phase_name)
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
    valid_power_idxs = torch.tensor(input_valid_power_idxs, dtype=torch.bool)
    # print('valid_power_idxs', valid_power_idxs)
    y_actions_lst = []
    power_orders_samples = (
        {power: [game.order_history[phase_name].get(power, [])] for power in POWERS}
        if cf_agent is None
        else get_cf_agent_order_samples(tmp_game, phase_name, cf_agent, n_cf_agent_samples)
    )
    for power_i, power in enumerate(POWERS):
        orders_samples = power_orders_samples[power]
        if len(orders_samples) == 0:
            valid_power_idxs[0, power_i] = False
            y_actions_lst.append(
                torch.empty(n_cf_agent_samples, MAX_SEQ_LEN, dtype=torch.int32).fill_(EOS_IDX)
            )
            continue
        encoded_power_actions_lst = []
        for orders in orders_samples:
            encoded_power_actions, valid = encode_power_actions(
                orders, x_possible_actions[power_i]
            )
            encoded_power_actions_lst.append(encoded_power_actions)
            if exclude_n_holds >= 0 and len(orders) >= exclude_n_holds:
                if all(o.endswith(" H") for o in orders):
                    valid = 0
            valid_power_idxs[power_i] &= valid
        y_actions_lst.append(torch.stack(encoded_power_actions_lst, dim=0))  # [N, 17]

    y_actions = torch.stack(y_actions_lst, dim=0)  # [7, N, 17]

    # filter away powers that have no orders
    valid_power_idxs &= (y_actions != EOS_IDX).any(dim=2).all(dim=1)
    assert valid_power_idxs.ndimension() == 1

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
        x_possible_actions.view(len(POWERS) * MAX_SEQ_LEN, MAX_VALID_LEN), padding_value=EOS_IDX
    )

    data_fields["x_power"] = torch.eye(len(POWERS), dtype=torch.bool).unsqueeze(0)
    data_fields["y_final_scores"] = y_final_scores.unsqueeze(0)
    data_fields["x_possible_actions"] = x_possible_actions
    data_fields["x_loc_idxs"] = x_loc_idxs.unsqueeze(0)
    data_fields["y_actions"] = y_actions.unsqueeze(0)
    data_fields["valid_power_idxs"] = valid_power_idxs.unsqueeze(0)

    return data_fields


def get_valid_orders_impl(power, all_possible_orders, all_orderable_locations, game_state):
    """Return a list of valid orders

    Returns:
    - a [1, 17, 469] int tensor of valid move indexes (padded with EOS_IDX)
    - a [1, 81] int8 tensor of orderable locs, described below
    - the actual length of the sequence == the number of orders to submit, <= 17

    loc_idxs:
    - not adj phase: X[i] = s if LOCS[i] is orderable at step s (0 <= s < 17), -1 otherwise
    - in adj phase: X[i] = -2 if LOCS[i] is orderable this phase, -1 otherwise
    """
    all_order_idxs = torch.empty(1, MAX_SEQ_LEN, MAX_VALID_LEN, dtype=torch.int32).fill_(EOS_IDX)
    loc_idxs = torch.empty(1, len(LOCS), dtype=torch.int8).fill_(-1)

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
        order_idxs = torch.tensor([ORDER_VOCABULARY_TO_IDX[x] for x in orders], dtype=torch.int32)
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


def get_cf_agent_order_samples(game, phase_name, cf_agent, n_cf_agent_samples):
    assert game.get_state()["name"] == phase_name, f"{game.get_state()['name']} != {phase_name}"

    if hasattr(cf_agent, "get_all_power_prob_distributions"):
        power_action_ps = cf_agent.get_all_power_prob_distributions(game)
        logging.info(f"get_all_power_prob_distributions: {power_action_ps}")
        return {
            power: (
                [sample_p_dict(power_action_ps[power]) for _ in range(n_cf_agent_samples)]
                if power_action_ps[power]
                else []
            )
            for power in POWERS
        }
    else:
        return {
            power: [cf_agent.get_orders(game, power) for _ in range(n_cf_agent_samples)]
            for power in POWERS
        }


def encode_power_actions(orders: List[str], x_possible_actions) -> Tuple[torch.LongTensor, bool]:
    """
    Arguments:
    - a list of orders, e.g. ["F APU - ION", "A NAP H"]
    - x_possible_actions, a LongTensor of valid actions for this power-phase, shape=[17, 469]

    Returns a tuple:
    - MAX_SEQ_LEN-len 1d-tensor, pad=EOS_IDX
    - True/False is valid
    """
    y_actions = torch.empty(MAX_SEQ_LEN, dtype=torch.int32).fill_(EOS_IDX)
    order_idxs = []

    if any(len(order.split()) < 3 for order in orders):
        # skip over power with unparseably short order
        return y_actions, False
    elif any(order.split()[2] == "B" for order in orders):
        # builds are represented as a single ;-separated order
        assert all(order.split()[2] == "B" for order in orders), orders
        order = ";".join(sorted(orders))
        try:
            order_idx = ORDER_VOCABULARY_TO_IDX[order]
        except KeyError:
            logging.warning(f"Invalid build order: {order}")
            return y_actions, False
        order_idxs.append(order_idx)
    else:
        for order in orders:
            try:
                order_idxs.append(smarter_order_index(order))
            except KeyError:
                # skip over invalid orders; we may fill them in later
                continue

    # sort by topo order
    order_idxs.sort(key=lambda idx: LOCS.index(ORDER_VOCABULARY[idx].split()[1]))
    for i, order_idx in enumerate(order_idxs):
        try:
            cand_idx = (x_possible_actions[i] == order_idx).nonzero()[0, 0]
            y_actions[i] = cand_idx
        except IndexError:
            # filter away powers whose orders are not in valid_orders
            # most common reasons why this happens:
            # - actual garbage orders (e.g. moves between non-adjacent locations)
            # - too many orders (e.g. three build orders with only two allowed builds)
            return y_actions, False

    return y_actions, True


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


def encode_weighted_sos_scores(game, phase_idx, value_decay_alpha):
    y_final_scores = torch.zeros(1, 7, dtype=torch.float)
    phases = game.get_phase_history()

    if phase_idx == len(phases) - 1:
        # end of game
        phase = phases[phase_idx]
        y_final_scores[0, :] = torch.FloatTensor(
            [
                compute_game_scores_from_state(p, phase.state).square_score
                for p in range(len(POWERS))
            ]
        )
        return y_final_scores

    # only weight scores at end of year, noting that not all years have a
    # winter adjustment phase
    end_of_year_phases = {}
    for phase in phases:
        end_of_year_phases[int(phase.name[1:-1])] = phase.name
    end_of_year_phases = set(end_of_year_phases.values())

    remaining = 1.0
    weight = 1.0 - value_decay_alpha
    for phase in phases[phase_idx + 1 :]:
        if phase.name not in end_of_year_phases:
            continue

        # calculate sos score at this phase
        sq_scores = torch.FloatTensor(
            [
                compute_game_scores_from_state(p, phase.state).square_score
                for p in range(len(POWERS))
            ]
        )

        # accumulate exp. weighted average
        y_final_scores[0, :] += weight * sq_scores
        remaining -= weight
        weight *= value_decay_alpha

    # fill in remaining weight with final score
    y_final_scores[0, :] += remaining * sq_scores

    return y_final_scores


def _cat(x):
    return TensorList.cat(x) if isinstance(x[0], TensorList) else torch.cat(x)
