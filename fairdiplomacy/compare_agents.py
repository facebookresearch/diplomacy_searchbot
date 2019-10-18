import argparse
import joblib

from fairdiplomacy.env import Env
from fairdiplomacy.models.consts import POWERS

from fairdiplomacy.agents.dipnet_agent import DipnetAgent
from fairdiplomacy.agents.mila_sl_agent import MilaSLAgent


def run_1v6_trial(agent_a, agent_b, agent_a_power):
    """Run a trial of 1x agent_a vs. 6x agent_b

    Arguments:
    - agent_a/b: fairdiplomacy.agents.BaseAgent inheritors
    - agent_a_power: the power to assign agent_a (the other 6 will be agent_b)

    Returns "a" if agent_a wins, or "b" if one of the agent_b powers wins, or "draw"
    """
    env = Env(
        {power: agent_a if power == agent_a_power else agent_b for power in POWERS}
    )
    scores = env.process_all_turns()

    if all(s < 18 for s in scores.values()):
        return "draw"

    winning_power = max(scores, key=scores.get)
    return "a" if winning_power == agent_a_power else "b"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a", help="Either 'mila' or a path to a .pth file")
    parser.add_argument("agent_b", help="Either 'mila' or a path to a .pth file")
    args = parser.parse_args()

    if args.agent_a.lower() == "mila":
        agent_a = MilaSLAgent()
    else:
        agent_a = DipnetAgent(args.agent_a)

    if args.agent_b.lower() == "mila":
        agent_b = MilaSLAgent()
    else:
        agent_b = DipnetAgent(args.agent_b)

    # Run trials (parallel)
    # results = joblib.Parallel(n_jobs=len(POWERS))(
    #     joblib.delayed(run_1v6_trial)(agent_a, agent_b, agent_a_power)
    #     for agent_a_power in POWERS
    # )

    results = {
        agent_a_power: run_1v6_trial(agent_a, agent_b, agent_a_power)
        for agent_a_power in POWERS
    }

    from pprint import pprint

    pprint(results)
