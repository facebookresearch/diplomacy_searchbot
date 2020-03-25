import diplomacy
import torch
from typing import List

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import get_valid_orders_impl
from fairdiplomacy.models.consts import SEASONS, POWERS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary, EOS_IDX

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_path="/checkpoint/jsgray/diplomacy/dipnet.pth", temperature=0.1):
        self.model = load_dipnet_model(model_path, map_location="cuda", eval=True)
        self.temperature = temperature

    def get_orders(self, game, power, *, temperature=None, batch_size=1):
        if len(game.get_orderable_locations(power)) == 0:
            return []

        temperature = temperature if temperature is not None else self.temperature
        inputs = encode_inputs(game)
        inputs = [x.to("cuda") for x in inputs]

        if batch_size > 1:
            # fake a big batch
            inputs = [x.expand(batch_size, *x.shape[1:]).contiguous() for x in inputs]

        with torch.no_grad():
            order_idxs, order_scores, final_scores = self.model(*inputs, temperature=temperature)

        return decode_order_idxs(order_idxs[0, POWERS.index(power), :])


def decode_order_idxs(order_idxs) -> List[str]:
    orders = []
    for idx in order_idxs:
        if idx == EOS_IDX:
            continue
        orders.extend(ORDER_VOCABULARY[idx].split(";"))
    return orders


def encode_inputs(game, *, all_possible_orders=None, game_state=None):
    """Return a 6-tuple of tensors

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_season: shape=(1, 3)
    x_in_adj_phase: shape=(1,), dtype=bool
    loc_idxs: shape=[1, 7, 81], dtype=long, -1, -2, or between 0 and 17
    valid_orders: shape=[1, 7, S, 469] dtype=long, 0 < S <= 17
    """
    if game_state is None:
        game_state = encode_state(game)
    x_board_state, x_prev_orders, x_season, x_in_adj_phase = game_state

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

    return (
        x_board_state,
        x_prev_orders,
        x_season,
        x_in_adj_phase,
        torch.stack(loc_idxs_lst, dim=1),
        torch.stack(valid_orders_lst, dim=1)[:, :, :max_seq_len],
    )


def encode_state(game):
    """Returns a 3-tuple of tensors:

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_season: shape=(1, 3)
    x_in_adj_phase: shape=(1,), dtype=bool
    """
    state = game.get_state()

    x_board_state = torch.from_numpy(board_state_to_np(state)).unsqueeze(0)

    try:
        last_move_phase = [
            phase for phase in game.get_phase_history() if str(phase.name).endswith("M")
        ][-1]
        x_prev_orders = torch.from_numpy(prev_orders_to_np(last_move_phase)).unsqueeze(0)
    except IndexError:
        x_prev_orders = torch.zeros(1, 81, 40)

    x_season = torch.zeros(1, 3)
    x_season[0, SEASONS.index(game.phase.split()[0])] = 1

    x_in_adj_phase = torch.zeros(1, dtype=torch.bool).fill_(state["name"][-1] == "A")

    return x_board_state, x_prev_orders, x_season, x_in_adj_phase


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


DEFAULT_INPUTS = encode_inputs(diplomacy.Game())


def zero_inputs():
    """Return empty input encodings"""
    r = [torch.zeros_like(x) for x in DEFAULT_INPUTS]
    r[-2].fill_(-1)
    return r


if __name__ == "__main__":
    game = diplomacy.Game()
    print(DipnetAgent().get_orders(game, "RUSSIA"))
