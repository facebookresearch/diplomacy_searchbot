import argparse
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

from fairdiplomacy.env import Env
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents.dipnet_agent import DipnetAgent
from fairdiplomacy.agents.mila_sl_agent import MilaSLAgent


def run_1v6_trial(agent_one_arg, agent_six_arg, agent_one_power):
    """Run a trial of 1x agent_one vs. 6x agent_six

    Arguments:
    - agent_one/six_arg: fairdiplomacy.agents.BaseAgent inheritor class, or (class, [args])
    - agent_one_power: the power to assign agent_one (the other 6 will be agent_six)

    Returns "one" if agent_one wins, or "six" if one of the agent_six powers wins, or "draw"
    """
    if type(agent_one_arg) == type:
        agent_one = agent_one_arg()
    else:
        cls, args = agent_one_arg
        agent_one = cls(*args)

    if type(agent_six_arg) == type:
        agent_six = agent_six_arg()
    else:
        cls, args = agent_six_arg
        agent_six = cls(*args)

    env = Env({power: agent_one if power == agent_one else agent_six for power in POWERS})
    scores = env.process_all_turns()

    if all(s < 18 for s in scores.values()):
        return "draw"

    winning_power = max(scores, key=scores.get)
    return "one" if winning_power == agent_one_power else "six"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a", help="Either 'mila' or a path to a .pth file")
    parser.add_argument("agent_b", help="Either 'mila' or a path to a .pth file")
    args = parser.parse_args()

    if args.agent_a.lower() == "mila":
        agent_a = MilaSLAgent
    else:
        agent_a = (DipnetAgent, args.agent_a)

    if args.agent_b.lower() == "mila":
        agent_b = MilaSLAgent
    else:
        agent_b = (DipnetAgent, args.agent_b)

    pool = ProcessPoolExecutor(14)

    # run 1A vs. 6B
    results_1a6b = {
        agent_a_power: pool.submit(run_1v6_trial, agent_a, agent_b, agent_a_power)
        for agent_a_power in POWERS
    }

    # run 6A vs. 1B
    results_6a1b = {
        agent_b_power: pool.submit(run_1v6_trial, agent_b, agent_a, agent_b_power)
        for agent_b_power in POWERS
    }

    # wait for results
    results_1a6b = {k: v.result() for k, v in results_1a6b.items()}
    results_6a1b = {k: v.result() for k, v in results_6a1b.items()}

    # pretty table with results
    winner_1a6b = []
    winner_6a1b = []
    for p in POWERS:
        if results_1a6b[p] == "draw":
            winner_1a6b.append("DRAW")
        else:
            winner_1a6b.append("A" if results_1a6b[p] == "one" else "B")

        if results_6a1b[p] == "draw":
            winner_6a1b.append("DRAW")
        else:
            winner_6a1b.append("B" if results_6a1b[p] == "one" else "A")

    print("AGENT A = {}".format(args.agent_a))
    print("AGENT B = {}".format(args.agent_b))
    print(tabulate(zip(POWERS, winner_1a6b, winner_6a1b), headers=["", "1A vs. 6B", "6A vs. 1B"]))
