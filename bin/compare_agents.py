#!/usr/bin/env python
import argparse
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

from fairdiplomacy.env import Env
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import DipnetAgent
from fairdiplomacy.agents.mila_sl_agent import MilaSLAgent
from fairdiplomacy.agents.simple_search_dipnet_agent import SimpleSearchDipnetAgent


def make_agent(agent_arg):
    if agent_arg is None:
        return None
    elif isinstance(agent_arg, BaseAgent):
        return agent_arg
    elif type(agent_arg) == type:
        return agent_arg()
    else:
        cls, args = agent_arg
        return cls(*args)


def run_1v6_trial(
    agent_one_arg, agent_six_arg, agent_one_power, save_path=None, seed=0, cf_agent=None
):
    """Run a trial of 1x agent_one vs. 6x agent_six

    Arguments:
    - agent_one/six_arg: fairdiplomacy.agents.BaseAgent inheritor class, or (class, [args])
    - agent_one_power: the power to assign agent_one (the other 6 will be agent_six)
    - save_path: if specified, save game.json to this path
    - seed: random seed
    - cf_agent: print out the orders for each power assuming that this agent was in charge

    Returns "one" if agent_one wins, or "six" if one of the agent_six powers wins, or "draw"
    """
    agent_one = make_agent(agent_one_arg)
    agent_six = make_agent(agent_six_arg)
    cf_agent = make_agent(cf_agent)

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
    print(f"Scores: {scores} ; Winner: {winning_power} ; agent_one_power= {agent_one_power}")
    return "one" if winning_power == agent_one_power else "six"


def parse_agent_cmdline(s):
    if s is None:
        return None

    if s == "mila":
        return MilaSLAgent
    elif s.startswith("mila:"):
        temp = float(s.split(":")[1])
        return (MilaSLAgent, (temp,))
    elif s == "search":
        return (SimpleSearchDipnetAgent, ("/checkpoint/jsgray/diplomacy/dipnet.pth",))
    else:
        return (DipnetAgent, (s,))


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
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    agent_one = parse_agent_cmdline(args.agent_one)
    agent_six = parse_agent_cmdline(args.agent_six)
    cf_agent = parse_agent_cmdline(args.cf_agent)

    result = run_1v6_trial(
        agent_one,
        agent_six,
        args.power_one,
        save_path=args.out if args.out else None,
        seed=args.seed,
        cf_agent=cf_agent,
    )
    logging.warning("Result: {}".format(result))
