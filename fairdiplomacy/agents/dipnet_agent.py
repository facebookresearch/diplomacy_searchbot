import diplomacy
import logging
import torch

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import smarter_order_index
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
)

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_pth):
        self.model = load_dipnet_model(model_pth, map_location="cpu", eval=True)
        self.model.cuda()

    def get_orders(self, game, power, temperature=0.1, debug_probs=False, batch_size=1):
        if len(game.get_orderable_locations(power)) == 0:
            return []
        inputs = encode_inputs(game, power)
        inputs = [x.to("cuda") for x in inputs]
        if batch_size > 1:
            # fake a big batch
            inputs = [x.expand(batch_size, *x.shape[1:]).contiguous() for x in inputs]
        with torch.no_grad():
            order_idxs, order_scores = self.model(*inputs, temperature=temperature)
        if debug_probs:
            self.debug_probs(
                order_idxs,
                order_scores,
                self.model.valid_order_idxs_to_mask(inputs[-1], inputs[-2]),
                temperature,
            )
        return [ORDER_VOCABULARY[idx] for idx in order_idxs[0, :]]

    def debug_probs(self, order_idxs, order_scores, mask, temperature):

        order_scores[~mask] = float("-inf")
        probs = torch.nn.functional.softmax(order_scores / temperature, dim=2)
        for step in range(len(order_idxs[0])):
            order_idx = order_idxs[0, step]
            logging.debug(
                "order {} p={}".format(ORDER_VOCABULARY[order_idx], probs[0, step, order_idx])
            )


def encode_inputs(game, power, all_possible_orders=None, game_state=None):
    """Return a 6-tuple of tensors

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_power: shape=(1, 7)
    x_season: shape=(1, 3)
    x_in_adj_phase: shape=(1,), dtype=bool
    loc_idxs: shape=[1, S], dtype=long, 0 < S <= 17
    x_order_mask: shape=[1, S, 13k], dtype=bool, 0 < S <= 17
    """
    if game_state is None:
        game_state = encode_state(game)
    x_board_state, x_prev_orders, x_season, x_in_adj_phase = game_state
    x_power = torch.zeros(1, 7)
    x_power[0, POWERS.index(power)] = 1
    order_mask, loc_idxs, seq_len = get_order_mask(game, power, all_possible_orders)
    return (
        x_board_state,
        x_prev_orders,
        x_power,
        x_season,
        x_in_adj_phase,
        loc_idxs[:, :seq_len],
        order_mask[:, :seq_len, :],
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


def get_order_mask(game, power, all_possible_orders=None):
    """Return a boolean mask of valid orders

    Returns:
    - a [1, 17, 468] int tensor of valid move indexes (padded with -1)
    - a [1, 17] long tensor of loc_idxs, 0 <= idx < 81
    - the actual lengh of the sequence == the number of orders to submit, <= 17
    """
    orderable_locs = sorted(game.get_orderable_locations(power), key=LOCS.index)
    if all_possible_orders is None:
        all_possible_orders = game.get_all_possible_orders()
    power_possible_orders = [x for loc in orderable_locs for x in all_possible_orders[loc]]
    n_builds = game.get_state()["builds"][power]["count"]
    all_order_idxs = torch.zeros(
        1, MAX_SEQ_LEN, get_order_vocabulary_idxs_len(), dtype=torch.int32
    )
    loc_idxs = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)

    if n_builds > 0:
        # build phase: all possible build orders, up to the number of allowed builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        assert order_idxs.ndimension() == 1
        all_order_idxs[0, :n_builds, : len(order_idxs)] = order_idxs.unsqueeze(0)
        return all_order_idxs, loc_idxs, n_builds

    if n_builds < 0:
        # disbands: all possible disband orders, up to the number of required disbands
        n_disbands = -n_builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        all_order_idxs[0, :n_builds, : len(order_idxs)] = order_idxs.unsqueeze(0)
        return all_order_idxs, loc_idxs, n_disbands

    # move phase: iterate through orderable_locs in topo order
    for i, loc in enumerate(orderable_locs):
        orders, order_idxs = filter_orders_in_vocab(all_possible_orders[loc])
        all_order_idxs[0, i, : len(order_idxs)] = order_idxs
        loc_idxs[0, i] = LOCS.index(loc)

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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="/checkpoint/jsgray/dipnet.pth")
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    args = parser.parse_args()

    agent = DipnetAgent(args.model_path)
    game = diplomacy.Game()
    orders = agent.get_orders(
        game, "ITALY", temperature=args.temperature, debug_probs=True
    )  # , batch_size=1000)
    logging.info("Submit orders: {}".format(orders))

    b, N = 1000, 100
    import time

    tic = time.time()
    for i in range(N):
        orders = agent.get_orders(game, "ITALY", temperature=args.temperature, batch_size=b)
    delta = time.time() - tic
    print(f"batch {b} N {N} : forward'd {b*N} in {delta} s. {b*N/delta} forwards/s")
