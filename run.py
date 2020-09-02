import logging
import os
import json
import torch
import numpy as np

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.compare_agents import run_1v6_trial, run_1v6_trial_multiprocess
from fairdiplomacy.launch_bot import run_with_cfg as launch_bot_run_with_cfg
from fairdiplomacy.models.dipnet import train_sl
from fairdiplomacy.press.models.parldipnet import train_press_sl
from fairdiplomacy.situation_check import run_situation_check
from fairdiplomacy.webdip_api import play_webdip as play_webdip_impl

import heyhi
import conf.conf_pb2


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def compare_agents(cfg):

    # NEED TO SET THIS BEFORE CREATING THE AGENT!
    if cfg.seed >= 0:
        logging.info(f"Set seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    agent_one = build_agent_from_cfg(cfg.agent_one)
    agent_six = build_agent_from_cfg(cfg.agent_six)
    if cfg.cf_agent.WhichOneof("agent") is not None:
        cf_agent = build_agent_from_cfg(cfg.cf_agent)
    else:
        cf_agent = None

    def _power_to_string(power_id):
        power_enum = conf.conf_pb2.CompareAgentsTask.Power
        return dict(zip(power_enum.values(), power_enum.keys()))[power_id]

    power_string = _power_to_string(cfg.power_one)

    if cfg.num_processes > 0:
        assert cfg.num_trials > 0
        result = run_1v6_trial_multiprocess(
            agent_one,
            agent_six,
            power_string,
            save_path=cfg.out if cfg.out else None,
            seed=cfg.seed,
            cf_agent=cf_agent,
            num_processes=cfg.num_processes,
            num_trials=cfg.num_trials,
            max_turns=cfg.max_turns,
            max_year=cfg.max_year,
            use_shared_agent=cfg.use_shared_agent,
        )
    else:
        result = run_1v6_trial(
            agent_one,
            agent_six,
            power_string,
            save_path=cfg.out if cfg.out else None,
            seed=cfg.seed,
            cf_agent=cf_agent,
            max_turns=cfg.max_turns,
            max_year=cfg.max_year,
            use_shared_agent=cfg.use_shared_agent,
        )
        logging.warning("Result: {}".format(result))


@_register
def train(cfg):
    train_sl.run_with_cfg(cfg)


@_register
def press_train(cfg):
    train_press_sl.run_with_cfg(cfg)


@_register
def launch_bot(cfg):
    if getattr(cfg, "requeue", False):
        heyhi.maybe_init_requeue_handler()
    launch_bot_run_with_cfg(cfg)


@_register
def exploit(cfg):
    # Do not load RL stuff by default.
    import fairdiplomacy.selfplay.exploit

    fairdiplomacy.selfplay.exploit.task(cfg)


@_register
def build_db_cache(cfg):
    from fairdiplomacy.data.build_db_cache import build_db_cache_from_cfg

    build_db_cache_from_cfg(cfg)


@_register
def build_press_db_cache(cfg):
    from fairdiplomacy.press.data.dataset import build_press_db_cache_from_cfg

    build_press_db_cache_from_cfg(cfg)


@_register
def situation_check(cfg):

    # NEED TO SET THIS BEFORE CREATING THE AGENT!
    if cfg.seed >= 0:
        logging.info(f"Set seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    agent = build_agent_from_cfg(cfg.agent)

    # If not absolute path, assume relative to project root.
    with open(heyhi.PROJ_ROOT / cfg.situation_json) as f:
        meta = json.load(f)

    selection = None
    if cfg.selection != "":
        selection = cfg.selection.split(",")
        meta = {k: v for k, v in meta.items() if k in selection}

    run_situation_check(meta, agent)


@_register
def benchmark_agent(cfg):
    import fairdiplomacy.benchmark_agent

    fairdiplomacy.benchmark_agent.run(cfg)


@_register
def play_webdip(cfg):

    agent = build_agent_from_cfg(cfg.agent)

    play_webdip_impl(
        webdip_url=cfg.webdip_url,
        api_keys=cfg.api_key.split(","),
        game_id=cfg.game_id,
        agent=agent,
        check_phase=cfg.check_phase,
        json_out=cfg.json_out,
        force=cfg.force,
        force_power=cfg.force_power,
    )


@_register
def compute_xpower_statistics(cfg):
    from fairdiplomacy.get_xpower_supports import compute_xpower_statistics, get_game_paths

    paths = get_game_paths(
        cfg.game_dir,
        metadata_path=cfg.metadata_path,
        metadata_filter=cfg.metadata_filter,
        dataset_for_eval=cfg.dataset_for_eval,
        max_games=cfg.max_games,
    )

    if cfg.cf_agent.WhichOneof("agent") is not None:
        cf_agent = build_agent_from_cfg(cfg.cf_agent)
    else:
        cf_agent = None

    compute_xpower_statistics(paths, max_year=cfg.max_year, cf_agent=cf_agent)


@_register
def profile_model(cfg):
    from fairdiplomacy.profile_model import profile_model

    profile_model(cfg.model_path)


@heyhi.save_result_in_cwd
def main(task, cfg):
    heyhi.setup_logging()
    logging.info("Cwd: %s", os.getcwd())
    logging.info("Task: %s", task)
    logging.info("Cfg:\n%s", cfg)
    heyhi.log_git_status()
    logging.info("Is on slurm: %s", heyhi.is_on_slurm())
    if heyhi.is_on_slurm():
        logging.info("Slurm job id: %s", heyhi.get_slurm_job_id())
    logging.info("Is master: %s", heyhi.is_master())

    if task not in TASKS:
        raise ValueError("Unknown task: %s. Known tasks: %s" % (task, sorted(TASKS)))
    return TASKS[task](cfg)


if __name__ == "__main__":
    heyhi.parse_args_and_maybe_launch(main)
