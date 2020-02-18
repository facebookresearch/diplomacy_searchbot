#!/usr/bin/env python
import argparse
import logging
import os
from ast import literal_eval
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

from fairdiplomacy.env import Env
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents import (
    DipnetAgent,
    MilaSLAgent,
    SimpleSearchDipnetAgent,
    CFR1PAgent,
)


def run_1v6_trial(agent_one, agent_six, agent_one_power, save_path=None, seed=0, cf_agent=None):
    """Run a trial of 1x agent_one vs. 6x agent_six

    Arguments:
    - agent_one/six: fairdiplomacy.agents.BaseAgent inheritor objects
    - agent_one_power: the power to assign agent_one (the other 6 will be agent_six)
    - save_path: if specified, save game.json to this path
    - seed: random seed
    - cf_agent: print out the orders for each power assuming that this agent was in charge

    Returns "one" if agent_one wins, or "six" if one of the agent_six powers wins, or "draw"
    """
    env = Env(
        {power: agent_one if power == agent_one_power else agent_six for power in POWERS},
        seed=seed,
        cf_agent=cf_agent,
    )

    scores = env.process_all_turns()

    if save_path is not None:
        env.save(save_path)

    if all(s < 18 for s in scores.values()):
        if scores[agent_one_power] > 0:
            # agent 1 is still alive and nobody has won
            return "draw"
        else:
            # agent 1 is dead, one of the agent 6 agents has won
            return "six"

    winning_power = max(scores, key=scores.get)
    logging.info(
        f"Scores: {scores} ; Winner: {winning_power} ; agent_one_power= {agent_one_power}"
    )
    return "one" if winning_power == agent_one_power else "six"


def parse_agent_class(s):
    return {
        "mila": MilaSLAgent,
        "search": SimpleSearchDipnetAgent,
        "diptorch": DipnetAgent,
        "cfr1p": CFR1PAgent,
        None: None,
    }[s]


def parse_kwargs(args):
    return {k: literal_eval(v) for k, v in (a.split("=", 1) for a in args)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_one", help="Either 'mila', 'search', or a path to a .pth file")
    parser.add_argument("agent_six", help="Either 'mila', 'search', or a path to a .pth file")
    parser.add_argument(
        "power_one",
        choices=["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"],
    )
    parser.add_argument("--out", "-o", help="Path to write game.json file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cf_agent", help="Either 'mila', 'search', or a path to a .pth file")
    parser.add_argument("--kwargs-one", nargs="+", default=[], help="kwargs to pass to agent one")
    parser.add_argument("--kwargs-six", nargs="+", default=[], help="kwargs to pass to agent six")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)
    logging.info(f"Args: {args}")

    agent_one = parse_agent_class(args.agent_one)(**parse_kwargs(args.kwargs_one))
    agent_six = parse_agent_class(args.agent_six)(**parse_kwargs(args.kwargs_six))
    cf_agent_cls = parse_agent_class(args.cf_agent)
    cf_agent = cf_agent_cls() if cf_agent_cls else None

    result = run_1v6_trial(
        agent_one,
        agent_six,
        args.power_one,
        save_path=args.out if args.out else None,
        seed=args.seed,
        cf_agent=cf_agent,
    )
    logging.warning("Result: {}".format(result))
