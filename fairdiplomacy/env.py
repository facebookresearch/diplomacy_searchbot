import logging
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

import diplomacy

from fairdiplomacy.agents.random_agent import RandomAgent
from fairdiplomacy.agents.mila_sl_agent import MilaSLAgent
from fairdiplomacy.agents.dipnet_agent import DipnetAgent

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.INFO)


class Env:
    def __init__(self, agents, seed=0, cf_agent=None):
        """
        agents must be one of:
            1. a list of 7 Agent objects, which will be randomly assigned to powers
            2. a dict of power name -> Agent object,
                e.g.  {"AUSTRIA": <Agent>, "FRANCE": <Agent>, ...}
        """
        self.game = diplomacy.Game()
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.game.powers))

        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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

        self.cf_agent = cf_agent
    def process_turn(self, timeout=10):
        logging.info("process_turn {}".format(self.game.phase))

        for power, agent in self.agents.items():
            if not self.game.get_orderable_locations(power):
                logging.info(f"Skipping orders for {power}")
                continue
            orders = agent.get_orders(self.game, power)
            logging.info("Set orders {} {}".format(power, orders))
            if self.cf_agent:
                cf_orders = self.cf_agent.get_orders(self.game, power)
                logging.info("CF orders {} {}".format(power, cf_orders))
            self.game.set_orders(power, orders)

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
    # mila_sl_agent = MilaSLAgent()
    dipnet_agent = DipnetAgent("/checkpoint/jsgray/dipnet.pth")
    env = Env(
        {
            "ITALY": dipnet_agent,
            "ENGLAND": dipnet_agent,
            "FRANCE": dipnet_agent,
            "GERMANY": dipnet_agent,
            "AUSTRIA": dipnet_agent,
            "RUSSIA": dipnet_agent,
            "TURKEY": dipnet_agent,
        }
    )
    results = env.process_all_turns()
    logging.info("Game over! Results: {}".format(results))
    env.save("game.json")
