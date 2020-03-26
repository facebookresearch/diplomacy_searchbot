import logging

from fairdiplomacy.env import Env
from fairdiplomacy.models.consts import POWERS


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