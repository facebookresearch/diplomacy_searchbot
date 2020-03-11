import diplomacy
import logging
import os
import time

from fairdiplomacy.agents import BRSearchAgent

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)
    logging.info("PID: {}".format(os.getpid()))

    game = diplomacy.Game()

    agent = BRSearchAgent()
    logging.info("Constructed agent")
    logging.info("Warmup: {}".format(agent.get_orders(game, "ITALY")))

    tic = time.time()
    N = 4
    for _ in range(N):
        logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))

    logging.info(f"Performed all rollouts for {N} searches in {time.time() - tic} s")
