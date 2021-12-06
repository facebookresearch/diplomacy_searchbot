# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
from typing import Any, Dict, Union, List, Optional, Sequence, Tuple

import joblib
import torch
from tqdm import tqdm

import heyhi
from conf import conf_cfgs
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.models.consts import POWERS, MAX_SEQ_LEN, LOCS, N_SCS
from fairdiplomacy.models.diplomacy_model.order_vocabulary import EOS_IDX
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import JointAction
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state
from fairdiplomacy.utils.order_idxs import (
    action_strs_to_global_idxs,
    global_order_idxs_to_local,
    global_order_idxs_to_str,
    MAX_VALID_LEN,
    OrderIdxConversionException,
)
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.tensorlist import TensorList
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder

LOC_IDX = {loc: idx for idx, loc in enumerate(LOCS)}


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, cfg: conf_cfgs.NoPressDatasetParams, *, use_validation: bool, all_powers=False
    ):
        torch.set_num_threads(1)

        self.debug_only_opening_phase = cfg.debug_only_opening_phase
        self.exclude_n_holds = cfg.exclude_n_holds
        self.n_cf_agent_samples = cfg.n_cf_agent_samples
        self.only_with_min_final_score = cfg.only_with_min_final_score
        self.value_decay_alpha = cfg.value_decay_alpha
        self.value_dir = cfg.value_dir
        self.limit_n_games = cfg.limit_n_games
        self.all_powers = all_powers

        game_data_path = cfg.val_set_path if use_validation else cfg.train_set_path

        # FIXME: uncomment when cfg has stabilized
        # if cfg.cf_agent.agent is not None:
        #     from fairdiplomacy.agents import build_agent_from_cfg
        #     self.cf_agent = build_agent_from_cfg(cfg.cf_agent)
        # else:
        #     self.cf_agent = None
        self.cf_agent = None

        logging.info(f"Reading metadata from {cfg.metadata_path}")
        with open(cfg.metadata_path) as meta_f:
            self.game_metadata = json.load(meta_f)

        # Metadata keys are sometimes paths, sometimes int. Be consistent here.
        extract_game_id_fn = (
            extract_game_id_int
            if is_int(next(k for k in self.game_metadata.keys()))
            else extract_game_id
        )
        self.game_metadata = {extract_game_id_fn(k): v for k, v in self.game_metadata.items()}

        self.min_rating = compute_min_rating(self.game_metadata, cfg.min_rating_percentile)
        logging.info(f"Only training on powers with min rating {self.min_rating}")

        if self.limit_n_games > 0:
            logging.info(f"Using only first {self.limit_n_games} games")
            n_games = self.limit_n_games
        else:
            logging.info("Skimming games data to get # games")
            with open(game_data_path) as f:
                n_games = 0
                for _ in f:
                    n_games += 1
            logging.info(f"Found {n_games} games")

        def read_data_lines(lines):
            for i, line in enumerate(lines):
                if self.limit_n_games > 0 and i >= self.limit_n_games:
                    return
                game_path, game_json = line.split(" ", 1)
                if game_path in self.game_metadata:
                    yield game_path, game_json
                else:
                    game_id = extract_game_id_fn(game_path)
                    if game_id in self.game_metadata:
                        yield game_id, game_json
                    else:
                        logging.debug(f"Skipping game id not in metadata: {game_id}")

        with open(game_data_path) as f:
            encoded_game_tuples = joblib.Parallel(n_jobs=cfg.num_dataloader_workers)(
                joblib.delayed(encode_game)(
                    game_id,
                    game_json,
                    value_dir=self.value_dir,
                    only_with_min_final_score=self.only_with_min_final_score,
                    cf_agent=self.cf_agent,
                    n_cf_agent_samples=self.n_cf_agent_samples,
                    value_decay_alpha=self.value_decay_alpha,
                    input_valid_power_idxs=self.get_valid_power_idxs(game_id),
                    game_metadata=self.game_metadata[game_id],
                    exclude_n_holds=self.exclude_n_holds,
                    all_powers=self.all_powers,
                )
                for game_id, game_json in tqdm(read_data_lines(f), total=n_games)
            )

        # Filter for games with valid phases
        encoded_games = [
            g for g in encoded_game_tuples if g is not None and g["valid_power_idxs"][0].any()
        ]
        logging.info(f"Found valid data for {len(encoded_games)} / {n_games} games")

        # Build x_idx and power_idx tensors used for indexing
        power_idxs, x_idxs = [], []
        x_idx = 0
        for encoded_game in encoded_games:
            for valid_power_idxs in encoded_game["valid_power_idxs"]:
                assert valid_power_idxs.nelement() == len(POWERS), (
                    encoded_game["valid_power_idxs"].shape,
                    valid_power_idxs.shape,
                )
                for power_idx in valid_power_idxs.nonzero(as_tuple=False)[:, 0]:
                    power_idxs.append(power_idx)
                    x_idxs.append(x_idx)
                x_idx += 1

        self.power_idxs = torch.tensor(power_idxs, dtype=torch.long)
        self.x_idxs = torch.tensor(x_idxs, dtype=torch.long)

        # now collate the data into giant tensors!
        self.encoded_games = DataFields.cat(encoded_games)

        self.num_games = len(encoded_games)
        self.num_phases = len(self.encoded_games["x_board_state"]) if self.encoded_games else 0
        self.num_elements = len(self.x_idxs)

        self.validate_dataset()
        logging.info("Validated dataset, returning")

    def stats_str(self):
        return f"Dataset: {self.num_games} games, {self.num_phases} phases, and {self.num_elements} elements."

    def validate_dataset(self):
        assert len(self) > 0
        for e in self.encoded_games.values():
            if isinstance(e, TensorList):
                max_seq_len = N_SCS if self.all_powers else MAX_SEQ_LEN
                assert len(e) == self.num_phases * len(POWERS) * max_seq_len
            else:
                assert len(e) == self.num_phases

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)

        max_seq_len = N_SCS if self.all_powers else MAX_SEQ_LEN

        assert isinstance(idx, torch.Tensor) and idx.dtype == torch.long
        assert idx.max() < len(self)

        sample_idxs = idx % self.n_cf_agent_samples
        idx //= self.n_cf_agent_samples

        x_idx = self.x_idxs[idx]
        power_idx = self.power_idxs[idx]

        fields = self.encoded_games.select(x_idx)  # [x[x_idx] for x in self.encoded_games[:-1]]

        if self.all_powers:
            # non-A phases are encoded in power idx 0
            power_idx[~fields["x_in_adj_phase"].bool()] = 0

        # unpack the possible_actions
        possible_actions_idx = ((x_idx * len(POWERS) + power_idx) * max_seq_len).unsqueeze(
            1
        ) + torch.arange(max_seq_len).unsqueeze(0)
        x_possible_actions = self.encoded_games["x_possible_actions"][
            possible_actions_idx.view(-1)
        ]
        x_possible_actions_padded = x_possible_actions.to_padded(
            total_length=MAX_VALID_LEN, padding_value=EOS_IDX
        )
        fields["x_possible_actions"] = x_possible_actions_padded.view(
            len(idx), max_seq_len, MAX_VALID_LEN
        ).long()

        # for these fields we need to select out the correct power
        for f in ("x_power", "x_loc_idxs", "y_actions"):
            fields[f] = fields[f][torch.arange(len(fields[f])), power_idx]

        # for y_actions, select out the correct sample
        fields["y_actions"] = (
            fields["y_actions"]
            .gather(1, sample_idxs.view(-1, 1, 1).repeat((1, 1, fields["y_actions"].shape[2])))
            .squeeze(1)
        )

        # FIXME: either do this in c++ encoding, or (preferably?) ensure that
        # it can be long in the model code
        fields["x_loc_idxs"] = fields["x_loc_idxs"].float()

        # set all_powers
        fields["all_powers"] = self.all_powers

        return fields

    def __len__(self):
        return self.num_elements * self.n_cf_agent_samples

    def get_valid_power_idxs(self, game_id):
        return [
            (self.game_metadata[game_id][pwr]["logit_rating"] >= self.min_rating)
            if self.min_rating is not None
            else True
            for pwr in POWERS
        ]


def encode_game(
    game_id,
    game_json: str,
    *,
    only_with_min_final_score=7,
    value_dir: Optional[str],
    cf_agent=None,
    n_cf_agent_samples=1,
    input_valid_power_idxs,
    value_decay_alpha,
    game_metadata,
    exclude_n_holds,
    all_powers: bool,
) -> Optional[DataFields]:
    torch.set_num_threads(1)
    encoder = FeatureEncoder()
    try:
        game = Game.from_json(game_json)
    except RuntimeError:
        logging.debug(f"RuntimeError (json decoding) while loading game id {game_id}")
        return None

    phase_names = [phase.name for phase in game.get_phase_history()]
    power_values = load_power_values(value_dir, game_id)

    num_phases = len(phase_names)
    logging.info(f"Encoding {game.game_id} with {num_phases} phases")
    phase_encodings = [
        encode_phase(
            encoder,
            game,
            game_id,
            phase_idx,
            only_with_min_final_score=only_with_min_final_score,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            value_decay_alpha=value_decay_alpha,
            input_valid_power_idxs=input_valid_power_idxs,
            exclude_n_holds=exclude_n_holds,
            power_values=power_values[phase_names[phase_idx]] if power_values else None,
            all_powers=all_powers,
        )
        for phase_idx in range(num_phases)
    ]

    stacked_encodings = DataFields.cat(phase_encodings)

    return stacked_encodings


def load_power_values(value_dir: Optional[str], game_id: int) -> Dict[str, Dict[str, float]]:
    """If value_dir is not None, tries to load values from a json values file.

    Returns:
        - A map from phase name to {pwr: value}
    """
    if value_dir:
        value_path = os.path.join(value_dir, f"game_{game_id}.json")
        try:
            with open(value_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f"Error while loading values at {value_path}: {e}")
    return None


def encode_phase(
    encoder: FeatureEncoder,
    game: Game,
    game_id: str,
    phase_idx: int,
    *,
    only_with_min_final_score: Optional[int],
    cf_agent=None,
    n_cf_agent_samples=1,
    value_decay_alpha,
    input_valid_power_idxs,
    exclude_n_holds,
    power_values=None,
    all_powers: bool,
):
    """
    Arguments:
    - game: Game object
    - game_id: unique id for game
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners).

    Returns: DataFields, including y_actions and y_final_score
    """

    # keep track of which powers are invalid, and which powers are valid but
    # weak (by min-rating and min-score)
    strong_power_idxs = torch.tensor(input_valid_power_idxs, dtype=torch.bool)
    valid_power_idxs = torch.ones_like(strong_power_idxs, dtype=torch.bool)

    phase = game.get_phase_history()[phase_idx]

    rolled_back_game = game.rolled_back_to_phase_start(phase.name)

    encode_fn = encoder.encode_inputs_all_powers if all_powers else encoder.encode_inputs
    data_fields = encode_fn([rolled_back_game])

    # encode final scores
    if power_values is not None:
        y_final_scores = torch.tensor([power_values[power] for power in POWERS])
    else:
        y_final_scores = encode_weighted_sos_scores(game, phase_idx, value_decay_alpha)

    # get actions from phase, or from cf_agent if set
    joint_action_samples = (
        {power: [phase.orders.get(power, [])] for power in POWERS}
        if cf_agent is None
        else get_cf_agent_order_samples(rolled_back_game, phase.name, cf_agent, n_cf_agent_samples)
    )

    # encode y_actions
    max_seq_len = N_SCS if all_powers else MAX_SEQ_LEN
    y_actions = torch.full(
        (len(POWERS), n_cf_agent_samples, max_seq_len), EOS_IDX, dtype=torch.long
    )
    for sample_i in range(n_cf_agent_samples):
        joint_action = {
            power: action_samples[sample_i]
            for power, action_samples in joint_action_samples.items()
        }
        if data_fields["x_in_adj_phase"] or not all_powers:
            for power_i, power in enumerate(POWERS):
                y_actions[power_i, sample_i, :], valid = encode_power_actions(
                    joint_action.get(power, []),
                    data_fields["x_possible_actions"][0, power_i],
                    data_fields["x_in_adj_phase"][0],
                    max_seq_len=max_seq_len,
                )
                valid_power_idxs[power_i] &= valid
        else:
            # do all_powers encoding
            y_actions[0, sample_i], valid_mask = encode_all_powers_action(
                joint_action,
                data_fields["x_possible_actions"][0],
                data_fields["x_power"][0],
                data_fields["x_in_adj_phase"][0],
            )
            valid_power_idxs &= valid_mask

    # Check for all-holds, no orders
    for power_i, power in enumerate(POWERS):
        orders = joint_action.get(power, [])
        if len(orders) == 0 or (
            0 <= exclude_n_holds <= len(orders) and all(order.endswith(" H") for order in orders)
        ):
            valid_power_idxs[power_i] = 0

    # Maybe filter away powers that don't finish with enough SC.
    # If all players finish with fewer SC, include everybody.
    if only_with_min_final_score is not None:
        final_score = {k: len(v) for k, v in game.get_state()["centers"].items()}
        if max(final_score.values()) >= only_with_min_final_score:
            for i, power in enumerate(POWERS):
                if final_score.get(power, 0) < only_with_min_final_score:
                    strong_power_idxs[i] = 0

    data_fields["y_final_scores"] = y_final_scores.unsqueeze(0)
    data_fields["y_actions"] = y_actions.unsqueeze(0)
    data_fields["x_possible_actions"] = TensorList.from_padded(
        data_fields["x_possible_actions"].view(-1, MAX_VALID_LEN), padding_value=EOS_IDX
    )
    data_fields["valid_power_idxs"] = valid_power_idxs.unsqueeze(0) & strong_power_idxs
    data_fields["valid_power_idxs_any_strength"] = valid_power_idxs.unsqueeze(0)

    if not all_powers:
        data_fields["x_power"] = torch.arange(len(POWERS)).view(1, -1, 1).repeat(1, 1, MAX_SEQ_LEN)

    return data_fields


def encode_power_actions(
    orders: List[str], x_possible_actions, x_in_adj_phase, *, max_seq_len=MAX_SEQ_LEN
) -> Tuple[torch.LongTensor, bool]:
    """
    Arguments:
    - a list of orders, e.g. ["F APU - ION", "A NAP H"]
    - x_possible_actions, a LongTensor of valid actions for this power-phase, shape=[17, 469]
    Returns a tuple:
    - max_seq_len-len 1d-tensor, pad=EOS_IDX
    - True/False is valid
    """
    y_actions = torch.full((max_seq_len,), EOS_IDX, dtype=torch.int32)
    order_idxs = []

    # Check for missing unit orders
    if not x_in_adj_phase:
        n_expected = len([x for x in x_possible_actions[:, 0].tolist() if x != -1])
        if n_expected != len(orders):
            logging.debug(f"Missing orders: {orders}, n_expected={n_expected}")
            return y_actions, False

    if any(len(order.split()) < 3 for order in orders):
        # skip over power with unparseably short order
        return y_actions, False
    elif any(order.split()[2] == "B" for order in orders):
        try:
            order_idxs.extend(action_strs_to_global_idxs(orders))
        except Exception:
            logging.debug(f"Invalid build orders: {orders}")
            return y_actions, False
    else:
        try:
            order_idxs.extend(
                action_strs_to_global_idxs(
                    orders, try_strip_coasts=True, ignore_missing=False, sort_by_loc=True
                )
            )
        except OrderIdxConversionException:
            logging.debug(f"Bad order in: {orders}")
            return y_actions, False

    for i, order_idx in enumerate(order_idxs):
        try:
            cand_idx = (x_possible_actions[i] == order_idx).nonzero(as_tuple=False)[0, 0]
            y_actions[i] = cand_idx
        except IndexError:
            # filter away powers whose orders are not in valid_orders
            # most common reasons why this happens:
            # - actual garbage orders (e.g. moves between non-adjacent locations)
            # - too many orders (e.g. three build orders with only two allowed builds)
            return y_actions, False

    return y_actions, True


def encode_all_powers_action(
    joint_action: JointAction,
    x_possible_actions: torch.LongTensor,
    x_power: torch.LongTensor,
    x_in_adj_phase,
):
    """
    Encode y_actions with all_powers=True. Some powers may be invalid due e.g.
    to missing orders, but we take care to ensure that y_actions lines up with
    x_possible_actions despite possible missing or invalid orders.

    Returns a tuple:
    - y_actions, a 1-d sequence of local order idxs in global LOC order
    - valid_mask: a [7]-shaped bool tensor indicating whether each power's actions were valid
    """
    max_seq_len = N_SCS
    valid_mask = torch.ones(7, dtype=torch.bool)
    y_actions = torch.full((max_seq_len,), EOS_IDX, dtype=torch.long)

    assert not x_in_adj_phase, "Use encode_power_actions"
    assert x_possible_actions.shape[0] == len(POWERS), x_possible_actions.shape
    assert x_possible_actions.shape[1] == max_seq_len, x_possible_actions.shape

    assert (x_possible_actions[1:] == EOS_IDX).all()
    x_possible_actions = x_possible_actions[0]

    orders_by_root_loc = {
        order.split()[1][:3]: order for power, orders in joint_action.items() for order in orders
    }
    expected_root_locs_sorted = [
        order.split()[1][:3] for order in global_order_idxs_to_str(x_possible_actions[:, 0])
    ]
    orders_sorted = [orders_by_root_loc.get(loc) for loc in expected_root_locs_sorted]
    global_idxs = action_strs_to_global_idxs(
        orders_sorted, try_strip_coasts=True, return_none_for_missing=True
    )
    for step, (order, loc, global_idx) in enumerate(
        zip(orders_sorted, expected_root_locs_sorted, global_idxs)
    ):
        if order is None:
            # Missing order
            valid_mask[x_power[0, step]] = 0
            continue

        if global_idx is None:
            # Handled OrderIdxConversionException
            valid_mask[x_power[0, step]] = 0
            continue

        nz = (x_possible_actions[step] == global_idx).nonzero(as_tuple=True)
        if len(nz) != 1:
            got_loc = order.split()[1][:3]
            assert got_loc == loc, f"{got_loc} != {loc}"
            logging.debug(f"Unexpected order for {loc}: {order}")
            valid_mask[x_power[0, step]] = 0
            continue
        local_idx = nz[0][0].item()
        y_actions[step] = local_idx

    return y_actions, valid_mask


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
    final_sq_scores = torch.FloatTensor(game.get_square_scores())
    y_final_scores[0, :] += remaining * final_sq_scores

    return y_final_scores


# imported by RL code
def cat_pad_inputs(xs: List[DataFields]) -> DataFields:
    batch = DataFields({k: [x[k] for x in xs] for k in xs[0].keys()})
    for k, v in batch.items():
        if k == "x_possible_actions":
            batch[k] = cat_pad_sequences(v, pad_value=-1, pad_to_len=MAX_SEQ_LEN)
        elif k == "x_loc_idxs":
            batch[k] = cat_pad_sequences(v, pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN)
        else:
            batch[k] = torch.cat(v)

    return batch


def shuffle_locations(batch: DataFields) -> DataFields:
    """Change location order in the batch randomly."""
    x_loc_idxs = batch["x_loc_idxs"]
    *batch_dims, _ = x_loc_idxs.shape
    device = x_loc_idxs.device

    loc_priority = torch.rand((*batch_dims, MAX_SEQ_LEN), device=device)
    # Shape: [batch_dims, 1]. Note, this will return 0 in case of adjustment
    # phase. That's a safe bet as we don't know how many actions will actually
    # present in y_actions.
    num_locs = (x_loc_idxs >= 0).sum(-1, keepdim=True)
    unsqueeze_shape = [1] * len(batch_dims) + [-1]
    invalid_mask = (
        torch.arange(MAX_SEQ_LEN, device=device).view(unsqueeze_shape).expand_as(loc_priority)
        >= num_locs
    )
    loc_priority += (1000 + torch.arange(MAX_SEQ_LEN, device=device)) * invalid_mask.float()
    perm = loc_priority.sort(dim=-1).indices
    return reorder_locations(batch, perm)


def reorder_locations(batch: DataFields, perm: torch.Tensor) -> DataFields:
    """Change location order in the batch according to the permutation."""
    batch = batch.copy()
    if "y_actions" in batch:
        y_actions = batch["y_actions"]
        batch["y_actions"] = y_actions.gather(-1, perm[..., : y_actions.shape[-1]])
        assert (
            (y_actions == -1) == (batch["y_actions"] == -1)
        ).all(), "permutation must keep undefined locations in place"

    if len(batch["x_possible_actions"].shape) == 3:
        batch["x_possible_actions"] = batch["x_possible_actions"].gather(
            -2, perm.unsqueeze(-1).repeat(1, 1, 469)
        )
    else:
        assert len(batch["x_possible_actions"].shape) == 4
        batch["x_possible_actions"] = batch["x_possible_actions"].gather(
            -2, perm.unsqueeze(-1).repeat(1, 1, 1, 469)
        )

    # In case if a move phase, x_loc_idxs is B x 81 or B x 7 x 81, where the
    # value in each loc is which order in the sequence it is (or -1 if not in
    # the sequence).
    new_x_loc_idxs = batch["x_loc_idxs"].clone()
    for lidx in range(perm.shape[-1]):
        mask = batch["x_loc_idxs"] == perm[..., lidx].unsqueeze(-1)
        new_x_loc_idxs[mask] = lidx
    batch["x_loc_idxs"] = new_x_loc_idxs
    return batch


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


def compute_min_rating(metadata: Dict, min_rating_percentile: float) -> float:
    if min_rating_percentile > 0:
        ratings = torch.tensor(
            [
                game[pwr]["logit_rating"]
                for game in metadata.values()
                for pwr in POWERS
                if pwr in game
            ]
        )
        return ratings.sort()[0][int(len(ratings) * min_rating_percentile)]
    else:
        return -1e9


def extract_game_id(path):
    # "/path/to/game_1234.json" -> "game_1234.json"
    return path.rsplit("/", 1)[-1]


def extract_game_id_int(path):
    # "/path/to/game_1234.json" -> 1234
    return int(re.findall(r"([0-9]+)", path)[-1])


def is_int(k):
    try:
        int(k)
        return True
    except ValueError:
        return False

