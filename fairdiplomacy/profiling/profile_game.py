from collections import defaultdict
import random
import time

import diplomacy

from fairdiplomacy.agents.random_agent import RandomAgent


if __name__ == "__main__":
    game = diplomacy.Game()

    n_turns = 0
    timings = defaultdict(float)

    print("START")
    t_start = time.time()

    while not game.is_game_done:
        t = time.time()
        all_possible_orders = game.get_all_possible_orders()
        timings["all_possible_orders"] += time.time() - t

        for power in game.powers:
            t = time.time()
            orderable_locs = game.get_orderable_locations(power)
            timings["orderable_locs"] += time.time() - t

            t = time.time()
            orders = [random.choice(all_possible_orders[loc]) for loc in orderable_locs]
            timings["random.choice"] += time.time() - t

            t = time.time()
            game.set_orders(power, orders)
            timings["set_orders"] += time.time() - t

        t = time.time()
        game.process()
        timings["process"] += time.time() - t

        n_turns += 1

    timings['total'] = time.time() - t_start
    print("END")

    for k, v in timings.items():
        print("{}:\t{} / {} = {}".format(k, v, n_turns, v / n_turns))
