from typing import Dict, List, Sequence, Tuple
import json
import logging

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import encode_inputs, decode_order_idxs
from fairdiplomacy.models.consts import POWERS
from parlai_diplomacy.utils.wrapper import wrapper_agent


class ParlAIAgent(BaseAgent):
    def __init__(self, *, model_path, predict_all_orders):
        if predict_all_orders:
            self._model = wrapper_agent.ParlAIAllOrderWrapper(model_path)
        else:
            self._model = wrapper_agent.ParlAISingleOrderWrapper(model_path)

    def get_orders(self, game, power) -> List[str]:
        return self.get_orders_many_powers(game, [power])[power]

    def get_orders_many_powers(self, game, powers: Sequence[str]) -> Dict[str, List[str]]:
        inputs = encode_inputs(game)
        # Shape: power X max_locations X max_orders.
        possible_order_ids = inputs["x_possible_actions"].squeeze(0)

        power_orders = {}
        game_state = game.get_state()
        game_json = json.loads(game.to_json())
        for power in powers:
            possible_orders = decode_order_idxs(possible_order_ids[POWERS.index(power)].view(-1))
            n_builds = game_state["builds"][power]["count"]
            max_orders = abs(n_builds) if n_builds != 0 else None

            power_orders[power] = self._model.produce_order(
                game_json=game_json,
                power=power,
                possible_orders=possible_orders,
                max_orders=max_orders,
            )
            power_orders[power], bad_orders = _filter_orders(power_orders[power], possible_orders)
            if bad_orders:
                logging.info("Removing bad orders for %s: %s", power, bad_orders)

        return power_orders


def _filter_orders(
    all_orders: Sequence[str], allowed_orders: Sequence[str]
) -> Tuple[List[str], List[str]]:
    allowed_orders = frozenset(allowed_orders)
    good_orders, bad_orders = [], []
    for order in all_orders:
        (bad_orders, good_orders)[order in allowed_orders].append(order)
    return good_orders, bad_orders
