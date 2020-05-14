import logging
import os
import subprocess
import json


from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.compare_agents import run_1v6_trial, run_1v6_trial_multiprocess
from fairdiplomacy.game import Game
from fairdiplomacy.launch_bot import run_with_cfg as launch_bot_run_with_cfg
from fairdiplomacy.models.dipnet import train_sl
import heyhi
import conf.conf_pb2


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def compare_agents(cfg):
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
        )
    else:
        result = run_1v6_trial(
            agent_one,
            agent_six,
            power_string,
            save_path=cfg.out if cfg.out else None,
            seed=cfg.seed,
            cf_agent=cf_agent,
        )
        logging.warning("Result: {}".format(result))


@_register
def train(cfg):
    train_sl.run_with_cfg(cfg)


@_register
def launch_bot(cfg):
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
def situation_check(cfg):
    from fairdiplomacy.models.consts import POWERS

    agent = build_agent_from_cfg(cfg.agent)

    selection = None
    if cfg.selection != "":
        selection = cfg.selection.split(",")

    with open(cfg.situation_json) as f:
        meta = json.load(f)

    for name, config in meta.items():
        if selection is not None and name not in selection:
            print(f"Skipping {name} because it's not in {selection}")
            continue
        print("=" * 80)
        print(f"{name}: {config['comment']} ({config['phase']})")
        print(f"path: {config['game_path']}")
        with open(config["game_path"]) as f:
            j = json.load(f)
        game = Game.from_saved_game_format(j)
        game = Game.clone_from(game, up_to_phase=config["phase"])

        prob_distributions = agent.get_all_power_prob_distributions(game)  # FIXME: early exit
        print("CFR Average strategy:")
        for power in POWERS:  # config["powers"]:
            pd = prob_distributions[power]
            pdl = sorted(list(pd.items()), key=lambda x: -x[1])
            print(f"    {power}")
            for order, prob in pdl:
                if prob < 0.05:
                    break
                print(f"        {prob:5.2f} {order}")


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
