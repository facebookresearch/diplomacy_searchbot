import glob
import json
import time
import torch

import pydipcc
from fairdiplomacy.data.dataset import get_valid_orders_impl
from fairdiplomacy.game import Game
from fairdiplomacy.models.dipnet.encoding import (
    board_state_to_np,
    prev_orders_to_np,
    mila_board_state_to_np,
)
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
)
from fairdiplomacy.models.consts import LOCS, POWERS
from pprint import pprint


ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}
cc_valid_orders_encoder = pydipcc.ValidOrdersEncoder(
    ORDER_VOCABULARY_TO_IDX, get_order_vocabulary_idxs_len()
)

t_cc = 0
t_py = 0

for game_json in glob.glob("/checkpoint/jsgray/diplomacy/6hgames/*.json"):
    print("Testing", game_json)

    with open(game_json) as f:
        j = json.load(f)

    py_game = Game()
    cc_game = pydipcc.Game()

    for phase, phase_orders in j["order_history"].items():
        if (
            py_game.current_short_phase != cc_game.current_short_phase
            and py_game.current_short_phase[-1] == "R"
        ):
            print("Skipping", py_game.current_short_phase)
            py_game.process()
            continue

        assert (
            py_game.current_short_phase == cc_game.current_short_phase == phase
        ), "{} != {} != {}".format(py_game.current_short_phase, cc_game.current_short_phase, phase)

        # TEST

        # t = time.time()
        # cc_bs_enc = pydipcc.encode_board_state_from_game(cc_game)
        # t_cc += time.time() - t

        # t = time.time()
        # py_bs_enc = mila_board_state_to_np(py_game.get_state())
        # t_py += time.time() - t

        # if not (cc_bs_enc == py_bs_enc).all():
        #     print("BS ENCODING DIFFERS:", (cc_bs_enc != py_bs_enc).any(axis=0).nonzero())

        # py_last_m_phase = [p for p in py_game.get_phase_history() if p.name[-1] == "M"]
        # if len(py_last_m_phase) > 0:
        #     py_last_m_phase = py_last_m_phase[-1]
        #     cc_last_m_phase = [p for p in cc_game.get_phase_history() if p.name[-1] == "M"][-1]
        #     cc_po_enc = pydipcc.encode_prev_orders(cc_last_m_phase)
        #     py_po_enc = prev_orders_to_np(py_last_m_phase)

        #     if not (cc_po_enc == py_po_enc).all():
        #         loc_diff = set((cc_po_enc != py_po_enc).any(axis=1).nonzero()[0].tolist())
        #         idx_diff = set((cc_po_enc != py_po_enc).any(axis=0).nonzero()[0].tolist())
        #         order_by_loc = {
        #             order.split()[1]: order
        #             for k, v in py_last_m_phase.orders.items()
        #             for order in v
        #         }

        #         # VIA move encodings are broken in py encoding
        #         if not all([order_by_loc[LOCS[i]].endswith("VIA") for i in loc_diff]):
        #             print("PO ENCODING DIFFERS idxs={} locs={}".format(idx_diff, loc_diff))
        #             import ipdb

        #             ipdb.set_trace()

        for power in POWERS:
            cc_valid_orders = cc_valid_orders_encoder.encode_valid_orders_from_game(power, cc_game)
            py_valid_orders = get_valid_orders_impl(
                power,
                py_game.get_all_possible_orders(),
                py_game.get_orderable_locations(),
                py_game.get_state(),
            )
            if not (torch.from_numpy(cc_valid_orders[0]) == py_valid_orders[0]).all():
                cc = torch.from_numpy(cc_valid_orders[0])
                py = py_valid_orders[0]
                for a, b in (cc != py).any(dim=-1).nonzero():
                    print(
                        "VALID ORDERS DIFFERS:",
                        [
                            get_order_vocabulary()[x]
                            for x in (set(cc[a, b].tolist()) ^ set(py[a, b].tolist()))
                        ],
                    )
                    if "A BER B" in [
                        get_order_vocabulary()[x]
                        for x in (set(cc[a, b].tolist()) ^ set(py[a, b].tolist()))
                    ]:
                        import ipdb
                        ipdb.set_trace()
            elif not (torch.from_numpy(cc_valid_orders[1]) == py_valid_orders[1]).all():
                print("VALID LOCS DIFFERS locs={}".format((cc != py).any(dim=-1).nonzero()))
                cc = torch.from_numpy(cc_valid_orders[0])
                py = py_valid_orders[0]
            # assert (torch.from_numpy(cc_valid_orders[0]) == py_valid_orders[0]).all()
            # assert (torch.from_numpy(cc_valid_orders[1]) == py_valid_orders[1]).all()
            elif not (cc_valid_orders[2] == py_valid_orders[2]):
                print("#LOCS DIFFERS", cc_valid_orders[2], "!=", py_valid_orders[2])

        # !TEST

        for game in [py_game, cc_game]:
            for power, orders in phase_orders.items():
                game.set_orders(power, orders)
            game.process()

print("t_py =", t_py)
print("t_cc =", t_cc)
print("t_py / t_cc =", t_py / t_cc)
