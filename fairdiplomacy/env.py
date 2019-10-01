import logging
import random
from concurrent.futures import ThreadPoolExecutor

import diplomacy

from agents.random_agent import RandomAgent

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.INFO)


class Env:
    def __init__(self, agents):
        """
        agents must be one of:
            1. a list of 7 Agent objects, which will be randomly assigned to powers
            2. a dict of power name -> Agent object,
                e.g.  {"AUSTRIA": <Agent>, "FRANCE": <Agent>, ...}
        """
        self.game = diplomacy.Game()
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.game.powers))

        # Save self.agents as a map of power name -> Agent
        if all(power in agents for power in self.game.powers.keys()):
            self.agents = agents
        else:
            expected_len_powers = len(self.game.powers.keys())
            if len(agents) != expected_len_powers:
                raise ValueError(
                    "Bad value for arg: agents, must be list or dict of {} Agents".format(
                        expected_len_powers
                    )
                )
            random.shuffle(agents)
            self.agents = dict(zip(self.game.powers.keys(), agents))

    def process_turn(self, timeout=10):
        logging.info("process_turn {}".format(self.game.phase))

        all_possible_orders = self.game.get_all_possible_orders()

        # TODO: this code is faster but harder to debug, leave it for now
        #
        # self.thread_pool_submit(agent.get_orders, self.game, possible_orders)
        # futures = {
        #     power: self.thread_pool.submit(agent.get_orders, self.game, all_possible_orders)
        #     for power, agent in self.agents.items()
        # }

        for power, agent in self.agents.items():
            possible_orders = {
                loc: orders
                for loc, orders in all_possible_orders.items()
                if loc in self.game.get_orderable_locations(power)
            }

            orders = agent.get_orders(self.game.get_state(), possible_orders)
            self.game.set_orders(power, orders)
            logging.info("Set orders {} {}".format(power, orders))

        self.game.process()

    def process_all_turns(self):
        """Process all turns until game is over

        Returns a dict mapping power -> supply count
        """
        while not self.game.is_game_done:
            self.process_turn()

        return {k: len(v) for k, v in self.game.get_state()["centers"].items()}

    def save(self, output_path):
        logging.info("Saving to {}".format(output_path))
        diplomacy.utils.export.to_saved_game_format(self.game, output_path)


if __name__ == "__main__":
    env = Env([RandomAgent() for _ in range(7)])
    results = env.process_all_turns()
    logging.info("Game over! Results: {}".format(results))
    env.save("game.json")
