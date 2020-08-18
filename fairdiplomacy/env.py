import logging
import numpy as np
import random
import time
import torch
from concurrent.futures import ThreadPoolExecutor

from fairdiplomacy.game import Game

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.INFO)


class Env:
    def __init__(self, agents, seed=0, cf_agent=None, max_year=1935):
        """
        agents must be one of:
            1. a list of 7 Agent objects, which will be randomly assigned to powers
            2. a dict of power name -> Agent object,
                e.g.  {"AUSTRIA": <Agent>, "FRANCE": <Agent>, ...}
        """
        self.game = Game(max_year=max_year)
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
        logging.debug("Starting turn {}".format(self.game.phase))

        for power, agent in self.agents.items():
            if not self.game.get_orderable_locations().get(power):
                logging.debug(f"Skipping orders for {power}")
                continue
            t = time.time()
            orders = agent.get_orders(self.game, power)
            logging.info(
                "Set orders {} {} {} in {}s".format(
                    self.game.current_short_phase, power, orders, time.time() - t
                )
            )
            if self.cf_agent:
                cf_orders = self.cf_agent.get_orders(self.game, power)
                logging.debug(
                    "CF  orders {} {} {}".format(self.game.current_short_phase, power, cf_orders)
                )
            self.game.set_orders(power, orders)

        self.game.process()

    def process_all_turns(self, max_turns=0):
        """Process all turns until game is over

        Returns a dict mapping power -> supply count
        """
        turn_id = 0
        while not self.game.is_game_done:
            if max_turns and turn_id >= max_turns:
                break
            self.process_turn()
            turn_id += 1

        return {k: len(v) for k, v in self.game.get_state()["centers"].items()}

    def save(self, output_path):
        logging.info("Saving to {}".format(output_path))
        self.game.to_saved_game_format(output_path)
