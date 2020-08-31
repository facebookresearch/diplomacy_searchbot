import diplomacy
import torch
import numpy as np
from typing import List

from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import (
    get_valid_orders_impl,
    encode_state,
    DataFields,
    MAX_VALID_LEN,
)
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
import pydipcc

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DipnetAgent(BaseAgent):
    def __init__(
        self, model_path, temperature, top_p=1.0, has_press=False, device=_DEFAULT_DEVICE
    ):
        self.model = load_dipnet_model(model_path, map_location=device, eval=True)
        self.temperature = temperature
        self.device = device
        self.top_p = top_p
        self.thread_pool = pydipcc.ThreadPool(
            1, ORDER_VOCABULARY_TO_IDX, get_order_vocabulary_idxs_len()
        )
        self.has_press = has_press

    def get_orders(self, game, power, *, temperature=None, top_p=None):
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return []

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        inputs = encode_inputs(game)
        if self.has_press:
            inputs["x_has_press"] = torch.ones((1, 1))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            order_idxs, cand_idxs, logits, final_scores = self.model(
                **inputs, temperature=temperature, top_p=top_p
            )

        resample_duplicate_disbands_inplace(
            order_idxs, cand_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
        )
        print(order_idxs)
        return decode_order_idxs(order_idxs[0, POWERS.index(power), :])

    def get_orders_all_powers(self, game, *, temperature=None, top_p=None):

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        inputs = encode_inputs(game)
        if self.has_press:
            inputs["x_has_press"] = torch.ones((1, 1))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            order_idxs, cand_idxs, logits, final_scores = self.model(
                **inputs, temperature=temperature, top_p=top_p
            )

        resample_duplicate_disbands_inplace(
            order_idxs, cand_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
        )
        return {
            power: decode_order_idxs(order_idxs[0, POWERS.index(power), :]) for power in POWERS
        }


def decode_order_idxs(order_idxs) -> List[str]:
    orders = []
    for idx in order_idxs:
        if idx == EOS_IDX:
            continue
        orders.extend(ORDER_VOCABULARY[idx].split(";"))
    return orders


def resample_duplicate_disbands_inplace(
    order_idxs, sampled_idxs, logits, x_possible_actions, x_in_adj_phase
):
    """Modify order_idxs and sampled_idxs in-place, resampling where there are
    multiple disband orders.
    """
    # Resample all multiple disbands. Since builds are a 1-step decode, any 2+
    # step adj-phase is a disband.
    if sampled_idxs.shape[2] < 2:
        return
    mask = (sampled_idxs[:, :, 1] != -1) & x_in_adj_phase.bool().unsqueeze(1)
    if not mask.any():
        return

    # N.B. we may sample more orders than we need here: we are sampling
    # according to the longest sequence in the batch, not the longest
    # multi-disband sequence. Still, even if there is a 3-unit disband and a
    # 2-unit disband, we will sample the same # for both and mask out the extra
    # orders (see the eos_mask below)
    #
    # Note 1 (see below): The longest sequence in the batch may be longer than
    # the # of disband order candidates, so we take the min with
    # logits.shape[3] (#candidates)
    try:
        new_sampled_idxs = torch.multinomial(
            logits[mask][:, 0].exp() + 1e-7,
            min(logits.shape[2], logits.shape[3]),  # See Note 1
            replacement=False,
        )
    except RuntimeError:
        # if you're reading this after Aug 25: remove this whole except block
        torch.save(
            {
                order_idxs: order_idxs,
                sampled_idxs: sampled_idxs,
                logits: logits,
                x_possible_actions: x_possible_actions,
                x_in_adj_phase: x_in_adj_phase,
            },
            "resample_duplicate_disbands_inplace.debug.pt",
        )
        raise

    filler = torch.empty(
        new_sampled_idxs.shape[0],
        sampled_idxs.shape[2] - new_sampled_idxs.shape[1],
        dtype=new_sampled_idxs.dtype,
        device=new_sampled_idxs.device,
    ).fill_(-1)

    eos_mask = sampled_idxs == EOS_IDX

    sampled_idxs[mask] = torch.cat([new_sampled_idxs, filler], dim=1)
    order_idxs[mask] = torch.cat(
        [x_possible_actions[mask][:, 0].long().gather(1, new_sampled_idxs), filler], dim=1
    )

    sampled_idxs[eos_mask] = EOS_IDX
    order_idxs[eos_mask] = EOS_IDX

    # copy step-0 logits to other steps, since they were the logits used above
    # to sample all steps
    logits[mask] = logits[mask][:, 0].unsqueeze(1)


def encode_batch_inputs(thread_pool, games) -> DataFields:
    B = len(games)

    batch = DataFields(
        x_board_state=np.empty((B, 81, 35), dtype=np.float32),
        x_prev_state=np.empty((B, 81, 35), dtype=np.float32),
        x_prev_orders=np.empty((B, 2, 100), dtype=np.long),
        x_season=np.empty((B, 3), dtype=np.float32),
        x_in_adj_phase=np.empty((B, 1), dtype=np.float32),
        x_build_numbers=np.empty((B, 7), dtype=np.float32),
        x_loc_idxs=np.empty((B, 7, 81), dtype=np.int8),
        x_possible_actions=np.empty((B, 7, MAX_SEQ_LEN, MAX_VALID_LEN), dtype=np.int32),
        x_max_seq_len=np.empty((B, 1), dtype=np.int32),
    )

    inputs = [
        [batch[key][b] for b in range(B)]
        for key in [
            "x_board_state",
            "x_prev_state",
            "x_prev_orders",
            "x_season",
            "x_in_adj_phase",
            "x_build_numbers",
            "x_loc_idxs",
            "x_possible_actions",
            "x_max_seq_len",
        ]
    ]

    thread_pool.encode_inputs_multi(games, *inputs)

    # FIXME: just remove x_max_seq_len entirely
    del batch["x_max_seq_len"]
    batch["x_in_adj_phase"] = batch["x_in_adj_phase"].squeeze(1)
    for k, v in batch.items():
        if type(v) == np.ndarray:
            batch[k] = torch.from_numpy(v)

    return batch


# TODO: deprecated, migrate callers to encode_batch_inputs
def encode_inputs_multi(thread_pool, games) -> List[DataFields]:
    all_data_fields = []
    for game in games:
        x_board_state = np.empty((1, 81, 35), dtype=np.float32)
        x_prev_state = np.empty((1, 81, 35), dtype=np.float32)
        x_prev_orders = np.empty((1, 2, 100), dtype=np.long)
        x_season = np.empty((1, 3), dtype=np.float32)
        x_in_adj_phase = np.empty((1,), dtype=np.float32)
        x_build_numbers = np.empty((1, 7), dtype=np.float32)
        x_loc_idxs = np.empty((1, 7, 81), dtype=np.int8)
        x_possible_actions = np.empty((1, 7, MAX_SEQ_LEN, MAX_VALID_LEN), dtype=np.int32)
        x_max_seq_len = np.empty((1,), dtype=np.int32)

        all_data_fields.append(
            DataFields(
                x_board_state=x_board_state,
                x_prev_state=x_prev_state,
                x_prev_orders=x_prev_orders,
                x_season=x_season,
                x_in_adj_phase=x_in_adj_phase,
                x_build_numbers=x_build_numbers,
                x_loc_idxs=x_loc_idxs,
                x_possible_actions=x_possible_actions,
                x_max_seq_len=x_max_seq_len,
            )
        )

    thread_pool.encode_inputs_multi(
        games,
        *[
            [x[key] for x in all_data_fields]
            for key in [
                "x_board_state",
                "x_prev_state",
                "x_prev_orders",
                "x_season",
                "x_in_adj_phase",
                "x_build_numbers",
                "x_loc_idxs",
                "x_possible_actions",
                "x_max_seq_len",
            ]
        ],
    )

    for d in all_data_fields:
        # d["x_possible_actions"] = d["x_possible_actions"][:, :, : d["x_max_seq_len"][0], :]
        del d["x_max_seq_len"]
        for k, v in d.items():
            if type(v) == np.ndarray:
                d[k] = torch.from_numpy(v)

    return all_data_fields


# TODO: deprecated, migrate callers to encode_batch_inputs
def encode_inputs(game, *, all_possible_orders=None, game_state=None):
    """Return a 6-tuple of tensors

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_season: shape=(1, 3)
    x_in_adj_phase: shape=(1,), dtype=bool
    build_numbers: shape=(7,)
    loc_idxs: shape=[1, 7, 81], dtype=long, -1, -2, or between 0 and 17
    valid_orders: shape=[1, 7, S, 469] dtype=long, 0 < S <= 17
    """
    if game_state is None:
        game_state = encode_state(game)

    valid_orders_lst, loc_idxs_lst = [], []
    max_seq_len = 0
    for power in POWERS:
        valid_orders, loc_idxs, seq_len = get_valid_orders(
            game,
            power,
            all_possible_orders=all_possible_orders,
            all_orderable_locations=game.get_orderable_locations(),
        )
        valid_orders_lst.append(valid_orders)
        loc_idxs_lst.append(loc_idxs)
        max_seq_len = max(seq_len, max_seq_len)

    game_state["x_loc_idxs"] = torch.stack(loc_idxs_lst, dim=1)
    game_state["x_possible_actions"] = torch.stack(valid_orders_lst, dim=1)[:, :, :max_seq_len]
    return game_state


def get_valid_orders(game, power, *, all_possible_orders=None, all_orderable_locations=None):
    """Return indices of valid orders

    Returns:
    - a [1, 17, 469] int tensor of valid move indexes (padded with -1)
    - a [1, 81] int8 tensor of orderable locs, described below
    - the actual length of the sequence == the number of orders to submit, <= 17
    """
    if all_possible_orders is None:
        all_possible_orders = game.get_all_possible_orders()
    if all_orderable_locations is None:
        all_orderable_locations = game.get_orderable_locations()

    return get_valid_orders_impl(
        power, all_possible_orders, all_orderable_locations, game.get_state()
    )


if __name__ == "__main__":
    game = pydipcc.Game()
    print(
        DipnetAgent(
            model_path="/checkpoint/alerer/fairdiplomacy/sl_fbdata_all/checkpoint.pth.best",
            temperature=1.0,
        ).get_orders(game, "RUSSIA")
    )
