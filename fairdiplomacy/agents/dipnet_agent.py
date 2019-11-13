import diplomacy
import logging
import torch

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import smarter_order_index
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary
from fairdiplomacy.models.dipnet.train_sl import new_model
from fairdiplomacy.models.dipnet.standard_topo_locs import STANDARD_TOPO_LOCS

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_pth):
        self.model = new_model()
        self.model.load_state_dict(
            torch.load(model_pth, map_location=torch.device("cpu"))["model"]
        )
        self.model.eval()

    def get_orders(self, game, power):
        # Get network inputs
        x_board_state, x_prev_orders, x_season = self.encode_state(game)
        x_power = torch.zeros(1, 7)
        x_power[0, POWERS.index(power)] = 1
        order_mask, seq_len = self.get_order_mask(game, power)

        # Forward pass
        order_idxs, _ = self.model(x_board_state, x_prev_orders, x_power, x_season, order_mask)
        return [ORDER_VOCABULARY[idx] for idx in order_idxs[:seq_len]]

    def encode_state(self, game):
        """Returns a 3-tuple of tensors:

        x_board_state: shape=(1, 81, 35)
        x_prev_orders: shape=(1, 81, 40)
        x_season: shape=(1, 3)
        """
        state = game.get_state()
        x_board_state = torch.from_numpy(board_state_to_np(state)).unsqueeze(0)

        try:
            last_move_phase, last_move_orders = [
                (phase, orders)
                for phase, orders in game.order_history.items()
                if str(phase).endswith("M")
            ][-1]
            last_move_state = game.state_history[last_move_phase]
            x_prev_orders = torch.from_numpy(
                prev_orders_to_np(last_move_state, last_move_orders)
            ).unsqueeze(0)
        except IndexError:
            x_prev_orders = torch.zeros(1, 81, 40)

        x_season = torch.zeros(1, 3)
        x_season[0, SEASONS.index(game.phase.split()[0])] = 1

        return x_board_state, x_prev_orders, x_season

    def get_order_mask(self, game, power):
        """Return a boolean mask of valid orders

        Returns:
        - a [17, 1, 13k] bool tensor
        - the actual lengh of the sequence == the number of orders to submit, <= 17
        """
        orderable_locs = sorted(game.get_orderable_locations(power), key=STANDARD_TOPO_LOCS.index)
        all_possible_orders = game.get_all_possible_orders()
        power_possible_orders = [x for loc in orderable_locs for x in all_possible_orders[loc]]
        n_builds = game.get_state()["builds"][power]["count"]
        order_mask = torch.zeros(MAX_SEQ_LEN, 1, len(ORDER_VOCABULARY), dtype=torch.bool)

        if n_builds > 0:
            # build phase: all possible build orders, up to the number of allowed builds
            _, order_idxs = filter_orders_in_vocab(power_possible_orders)
            order_mask[:n_builds, 0, order_idxs] = 1
            return order_mask, n_builds

        if n_builds < 0:
            # disbands: all possible disband orders, up to the number of required disbands
            n_disbands = -n_builds
            _, order_idxs = filter_orders_in_vocab(power_possible_orders)
            order_mask[:n_disbands, 0, order_idxs] = 1
            return order_mask, n_disbands

        # move phase: iterate through orderable_locs in topo order
        for i, loc in enumerate(orderable_locs):
            orders, order_idxs = filter_orders_in_vocab(all_possible_orders[loc])
            order_mask[i, 0, order_idxs] = 1
        return order_mask, len(orderable_locs)


def filter_orders_in_vocab(orders):
    """Return the subset of orders that are found in the vocab, and their idxs"""
    ret, idxs = [], []
    for order in orders:
        try:
            idx = smarter_order_index(order)
            ret.append(order)
            idxs.append(idx)
        except ValueError:
            continue
    return ret, idxs


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    agent = DipnetAgent("/checkpoint/jsgray/dipnet.pth")
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY")
    logging.info("Submit orders: {}".format(orders))
