from fairdiplomacy import pydipcc
import time
import numpy as np
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
)

ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}

N_THREADS = 50
N_GAMES = 150

OPENING = {
    "AUSTRIA": ["A BUD - SER", "F TRI - ALB", "A VIE - TRI"],
    "ENGLAND": ["F LON - ENG", "A LVP - YOR", "F EDI - NTH"],
    "FRANCE": ["A MAR - SPA", "A PAR - BUR", "F BRE - MAO"],
    "GERMANY": ["A MUN - RUH", "A BER - KIE", "F KIE - DEN"],
    "ITALY": ["A VEN H", "F NAP - ION", "A ROM - APU"],
    "RUSSIA": ["F SEV - BLA", "A WAR - GAL", "F STP/SC - BOT", "A MOS - UKR"],
    "TURKEY": ["A CON - BUL", "F ANK - BLA", "A SMY - ARM"],
}

# ORIGINAL

games = [pydipcc.Game() for _ in range(N_GAMES)]
for game in games:
    for power, orders in OPENING.items():
        game.set_orders(power, orders)

t = time.time()
for game in games:
    game.process()
print("ORIGINAL", time.time() - t)

for game in games:
    assert game.current_short_phase == "F1901M"


# WITH THREADPOOL

games = [pydipcc.Game() for _ in range(N_GAMES)]
pool = pydipcc.ThreadPool(N_THREADS, ORDER_VOCABULARY_TO_IDX, get_order_vocabulary_idxs_len())

for game in games:
    for power, orders in OPENING.items():
        game.set_orders(power, orders)


t = time.time()
# pool.process_multi(games)
# print("THREADPOOL", time.time() - t)
#
# for game in games:
#     assert game.current_short_phase == "F1901M"


# TEST ENCODE_INPUTS_MULTI
all_data_fields = []
for game in games:
    all_data_fields.append(
        dict(
            x_board_state=np.empty((1, 81, 35), dtype=np.float32),
            x_prev_state=np.empty((1, 81, 35), dtype=np.float32),
            x_prev_orders=np.empty((1, 2, 100), dtype=np.long),
            x_season=np.empty((1, 3), dtype=np.float32),
            x_in_adj_phase=np.empty((1,), dtype=np.float32),
            x_build_numbers=np.empty((1, 7), dtype=np.float32),
            x_loc_idxs=np.empty((1, 7, 81), dtype=np.int8),
            x_possible_actions=np.empty((1, 7, 17, 469), dtype=np.int32),
            x_max_seq_len=np.empty((1,), dtype=np.int32),
        )
    )

pool.encode_inputs_multi(
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
    ]
)

# pool.encode_inputs_multi(
#     [games[0]],
#     [x_board_state],
#     [x_prev_state],
#     [x_prev_orders],
#     [x_season],
#     [x_in_adj_phase],
#     [x_build_numbers],
#     [x_loc_idxs],
#     [x_possible_actions],
#     [x_max_seq_len],
# )

print(all_data_fields[1]["x_board_state"])
print(all_data_fields[1]["x_prev_orders"])
print(all_data_fields[1]["x_max_seq_len"])
