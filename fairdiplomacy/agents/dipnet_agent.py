import diplomacy
import logging
import os
import random
import torch

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.dataset import smarter_order_index
from fairdiplomacy.models.consts import SEASONS, POWERS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary
from fairdiplomacy.models.dipnet.train_sl import new_model

ORDER_VOCABULARY = get_order_vocabulary()


class DipnetAgent(BaseAgent):
    def __init__(self, model_pth):
        self.model = new_model()
        self.model.load_state_dict(torch.load(model_pth, map_location=torch.device("cpu")))
        self.model.eval()

    def get_orders(self, game, power):
        # Get network inputs
        x_board_state, x_prev_orders, x_season = self.encode_state(game)
        x_power = torch.zeros(1, 7)
        x_power[0, POWERS.index(power)] = 1

        # Score all actions
        scores = self.model(x_board_state, x_prev_orders, x_power, x_season)

        orders = []
        all_possible_orders = game.get_all_possible_orders()
        for loc in game.get_orderable_locations(power):
            loc_possible_orders = all_possible_orders[loc]

            # FIXME: ensure WAIVE is in vocab, and remove this
            loc_possible_orders = [x for x in loc_possible_orders if x != "WAIVE"]

            idxs = [smarter_order_index(order) for order in loc_possible_orders]
            order_scores = torch.Tensor([scores[0, idx] for idx in idxs])
            order_probs = torch.nn.functional.softmax(order_scores).numpy()
            for order, score, prob in sorted(
                zip(loc_possible_orders, order_scores, order_probs), key=lambda z: -z[2]
            ):
                logging.debug('"{}": {} -> {}'.format(order, score, prob))

            order = random.choices(loc_possible_orders, order_probs)[0]
            orders.append(order)

        return orders

    def encode_state(self, game):
        """Returns a 3-tuple of tensors:

        x_board_state: shape=(1, 81, 35)
        x_prev_orders: shape=(1, 81, 40)
        x_season: shape=(1, 3)
        """
        state = game.get_state()
        x_board_state = torch.from_numpy(board_state_to_np(state)).unsqueeze(0)

        try:
            last_move_orders = [
                orders for phase, orders in game.order_history.items() if str(phase).endswith("M")
            ][-1]
            x_prev_orders = torch.from_numpy(prev_orders_to_np(state, last_move_orders)).unsqueeze(
                0
            )
        except IndexError:
            x_prev_orders = torch.zeros(1, 81, 40)

        x_season = torch.zeros(1, 3)
        x_season[0, SEASONS.index(game.phase.split()[0])] = 1

        return x_board_state, x_prev_orders, x_season


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    agent = DipnetAgent(
        os.path.join(os.path.dirname(__file__), "../models/dipnet/dipnet_state.pth")
    )
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY")
    logging.info("Submit orders: {}".format(orders))
