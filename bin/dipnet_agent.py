#!/usr/bin/env python
import diplomacy
import logging

from fairdiplomacy.agents.dipnet_agent import DipnetAgent

if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="/checkpoint/jsgray/dipnet.pth")
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    args = parser.parse_args()

    agent = DipnetAgent(args.model_path)
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY", temperature=args.temperature, debug_probs=True)
    logging.info("Submit orders: {}".format(orders))
