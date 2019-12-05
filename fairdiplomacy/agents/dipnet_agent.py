import os
import diplomacy
import logging
import torch

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import smarter_order_index
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary
from fairdiplomacy.models.dipnet.train_sl import new_model

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_pth):
        self.model = load_dipnet_model(model_pth, map_location="cpu", eval=True)

    def get_orders(self, game, power, temperature=0.1, debug_probs=False):
        if len(game.get_orderable_locations(power)) == 0:
            return []
        inputs = encode_inputs(game, power)
        with torch.no_grad():
            order_idxs, order_scores = self.model(*inputs, temperature=temperature)
        if debug_probs:
            self.debug_probs(order_idxs, order_scores, inputs[-1], temperature)
        return [ORDER_VOCABULARY[idx] for idx in order_idxs[0, :]]

    def debug_probs(self, order_idxs, order_scores, mask, temperature):
        order_scores[~mask] = float("-inf")
        probs = torch.nn.functional.softmax(order_scores / temperature, dim=2)
        for step in range(len(order_idxs[0])):
            order_idx = order_idxs[0, step]
            logging.debug(
                "order {} p={}".format(ORDER_VOCABULARY[order_idx], probs[0, step, order_idx])
            )


def encode_inputs(game, power, all_possible_orders=None):
    """Return a 5-tuple of tensors

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_season: shape=(1, 3)
    x_power: shape=(1, 7)
    x_order_mask: shape=[1, S, 13k], dtype=bool, 0 < S <= 17
    """
    x_board_state, x_prev_orders, x_season = encode_state(game)
    x_power = torch.zeros(1, 7)
    x_power[0, POWERS.index(power)] = 1
    order_mask, seq_len = get_order_mask(game, power, all_possible_orders)
    return (x_board_state, x_prev_orders, x_season, x_power, order_mask[:, :seq_len, :])


def encode_state(game):
    """Returns a 3-tuple of tensors:

    x_board_state: shape=(1, 81, 35)
    x_prev_orders: shape=(1, 81, 40)
    x_season: shape=(1, 3)
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

    return x_board_state, x_prev_orders, x_season


def get_order_mask(game, power, all_possible_orders=None):
    """Return a boolean mask of valid orders

    Returns:
    - a [1, 17, 13k] bool tensor
    - the actual lengh of the sequence == the number of orders to submit, <= 17
    """
    orderable_locs = sorted(game.get_orderable_locations(power), key=LOCS.index)
    if all_possible_orders is None:
        all_possible_orders = game.get_all_possible_orders()
    power_possible_orders = [x for loc in orderable_locs for x in all_possible_orders[loc]]
    n_builds = game.get_state()["builds"][power]["count"]
    order_mask = torch.zeros(1, MAX_SEQ_LEN, len(ORDER_VOCABULARY), dtype=torch.bool)

    if n_builds > 0:
        # build phase: all possible build orders, up to the number of allowed builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        order_mask[0, :n_builds, order_idxs] = 1
        return order_mask, n_builds

    if n_builds < 0:
        # disbands: all possible disband orders, up to the number of required disbands
        n_disbands = -n_builds
        _, order_idxs = filter_orders_in_vocab(power_possible_orders)
        order_mask[0, :n_disbands, order_idxs] = 1
        return order_mask, n_disbands

    # move phase: iterate through orderable_locs in topo order
    for i, loc in enumerate(orderable_locs):
        orders, order_idxs = filter_orders_in_vocab(all_possible_orders[loc])
        order_mask[0, i, order_idxs] = 1

    return order_mask, len(orderable_locs)


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
    return ret, idxs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='?', default="/checkpoint/jsgray/dipnet.pth")
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    args = parser.parse_args()

    agent = DipnetAgent(args.model_path)
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY", temperature=args.temperature, debug_probs=True)
    logging.info("Submit orders: {}".format(orders))
