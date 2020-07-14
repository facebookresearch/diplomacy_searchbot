import diplomacy
import torch
import numpy as np
from typing import List

from fairdiplomacy.game import Game, sort_phase_key
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import get_valid_orders_impl, encode_state, DataFields
from fairdiplomacy.models.consts import SEASONS, POWERS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary, EOS_IDX

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_path, temperature, top_p=1.0, device="cuda"):
        self.model = load_dipnet_model(model_path, map_location=device, eval=True)
        self.temperature = temperature
        self.device = device
        self.top_p = top_p

    def get_orders(self, game, power, *, temperature=None, top_p=None):
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return []

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        inputs = encode_inputs(game)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            order_idxs, cand_idxs, logits, final_scores = self.model(
                **inputs, temperature=temperature, top_p=top_p
            )

        resample_duplicate_disbands_inplace(
            order_idxs, cand_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
        )

        return decode_order_idxs(order_idxs[0, POWERS.index(power), :])


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
    mask = (sampled_idxs[:, :, 1] != -1) & x_in_adj_phase.bool().unsqueeze(1)
    if not mask.any():
        return

    new_sampled_idxs = torch.multinomial(
        logits[mask][:, 0].exp() + 1e-7, logits.shape[2], replacement=False
    )

    filler = torch.empty(
        new_sampled_idxs.shape[0],
        sampled_idxs.shape[2] - new_sampled_idxs.shape[1],
        dtype=new_sampled_idxs.dtype,
        device=new_sampled_idxs.device,
    ).fill_(-1)

    sampled_idxs[mask] = torch.cat([new_sampled_idxs, filler], dim=1)
    order_idxs[mask] = torch.cat(
        [x_possible_actions[mask][:, 0].long().gather(1, new_sampled_idxs), filler], dim=1
    )

    # copy step-0 logits to other steps, since they were the logits used above
    # to sample all steps
    logits[mask] = logits[mask][:, 0].unsqueeze(1)


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


DEFAULT_INPUTS = encode_inputs(Game())


def zero_inputs():
    """Return empty input encodings"""
    r = {k: torch.zeros_like(v) for k, v in DEFAULT_INPUTS.items()}
    r["x_loc_idxs"].fill_(-1)
    return r


if __name__ == "__main__":
    game = Game()
    print(
        DipnetAgent(
            model_path="/checkpoint/jsgray/diplomacy/sl_candemb_no13k_ep85.pth", device="cpu"
        ).get_orders(game, "RUSSIA")
    )
