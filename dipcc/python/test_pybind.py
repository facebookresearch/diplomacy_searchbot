from fairdiplomacy import pydipcc
from pprint import pprint
import json
import fairdiplomacy.game
from fairdiplomacy.models.dipnet.encoding import board_state_to_np


def setify_orders(all_possible_orders):
    return {k: frozenset(v) for k, v in all_possible_orders.items()}


def orders_diff(a, b):
    a = {(loc, order) for loc, orders in a.items() for order in orders}
    b = {(loc, order) for loc, orders in b.items() for order in orders}
    return a - b


game = pydipcc.Game()
encoded = pydipcc.encode_board_state(game)
# pprint(game.get_state())
# pprint(game.game_id)
# pprint(game.is_game_done)


game_2 = pydipcc.Game.from_json(game.to_json())
# pprint(json.loads(game.to_json()))
# pprint(json.loads(game_2.to_json()))
assert json.loads(game_2.to_json()) == json.loads(game.to_json())


py_game = fairdiplomacy.game.Game()

py_encoded = board_state_to_np(py_game.get_state())
assert (encoded == py_encoded).all()

# pprint(py_game.get_orderable_locations())
# pprint(game.get_orderable_locations())
# assert set(py_game.get_orderable_locations()) == set(game.get_orderable_locations())
#
# for orders in [["F STP/SC - BOT", "F SEV - BLA"], ["F BOT - SWE", "F BLA - RUM"]]:
#     for g in [game, py_game]:
#         g.set_orders("RUSSIA", orders)
#         g.process()
#
#     print("\n" + g.phase)
#
#     # locations
#     print(game.get_orderable_locations())
#     print("EXTRA:", orders_diff(game.get_orderable_locations(), py_game.get_orderable_locations()))
#     print(
#         "MISSING:", orders_diff(py_game.get_orderable_locations(), game.get_orderable_locations())
#     )
#
#     # orders
#     py = setify_orders(py_game.get_all_possible_orders())
#     cc = setify_orders(game.get_all_possible_orders())
#     print("MISSING:", orders_diff(py, cc))
#     print("EXTRA:", orders_diff(cc, py))
