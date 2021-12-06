# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import os
import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.env import Env, OneSixPolicyProfile, SharedPolicyProfile

from conf import conf_cfgs


def run_1v6_trial(
    agent_one, agent_six, agent_one_power: str, cfg: conf_cfgs.CompareAgentsTask, cf_agent=None
):
    """Run a trial of 1x agent_one vs. 6x agent_six

    Arguments:
    - agent_one/six: fairdiplomacy.agents.BaseAgent inheritor objects
    - agent_one_power: the power to assign agent_one (the other 6 will be agent_six)
    - cfg: see conf.proto

    Returns "one" if agent_one wins, or "six" if one of the agent_six powers wins, or "draw"
    """
    torch.set_num_threads(1)

    if cfg.use_shared_agent:
        del agent_six  # Unused.
        policy_profile = SharedPolicyProfile(agent_one)
    else:
        policy_profile = OneSixPolicyProfile(
            agent_one=agent_one,
            agent_six=agent_six,
            agent_one_power=agent_one_power,
            seed=cfg.seed,
        )

    if cfg.start_game:
        with open(cfg.start_game) as stream:
            game_obj = pydipcc.Game.from_json(stream.read())
        if cfg.start_phase:
            game_obj = game_obj.rolled_back_to_phase_start(cfg.start_phase)

    else:
        game_obj = pydipcc.Game()

    if cfg.draw_on_stalemate_years > 0:
        game_obj.set_draw_on_stalemate_years(cfg.draw_on_stalemate_years)

    env = Env(
        policy_profile=policy_profile,
        seed=cfg.seed,
        cf_agent=cf_agent,
        max_year=cfg.max_year,
        game_obj=game_obj,
        capture_logs=cfg.capture_logs,
    )

    scores = env.process_all_turns(max_turns=cfg.max_turns)

    if cfg.out:
        env.save(cfg.out)
    if all(s < 18 for s in scores.values()):
        if scores[agent_one_power] > 0:
            # agent 1 is still alive and nobody has won
            result = "draw"
        else:
            # agent 1 is dead, one of the agent 6 agents has won
            result = "six"
        winning_power = "NONE"
    else:
        winning_power = max(scores, key=scores.get)
        result = "one" if winning_power == agent_one_power else "six"

    logging.info(
        f"Scores: {scores} ; Winner: {winning_power} ; agent_one_power= {agent_one_power}"
    )
    return result


def call_with_args(args):
    args[0](*args[1:])


def run_1v6_trial_multiprocess(
    agent_one, agent_six, agent_one_power, cfg: conf_cfgs.CompareAgentsTask, cf_agent=None
):
    torch.set_num_threads(1)
    save_base, save_ext = os.path.splitext(cfg)
    os.makedirs(save_base, exist_ok=True)
    pool = mp.get_context("spawn").Pool(cfg.num_processes)
    BIG_PRIME = 377011
    pool.map(
        call_with_args,
        [
            (
                run_1v6_trial,
                agent_one,
                agent_six,
                agent_one_power,
                f"{save_base}/output_{job_id}{save_ext}",
                cfg.seed + job_id * BIG_PRIME,
                cf_agent,
            )
            for job_id in range(cfg.num_trials)
        ],
    )
    logging.info("TERMINATING")
    pool.terminate()
    logging.info("FINISHED")
    return ""
