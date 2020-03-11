import logging
import os
import subprocess

import bin.compare_agents
from fairdiplomacy.models.dipnet import train_sl

if "SLURM_PROCID" not in os.environ:
    subprocess.check_call(
        "protoc conf/conf.proto --python_out ./".split(),
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

import heyhi
import conf.conf_pb2


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def compare_agents(cfg):
    agent_one = bin.compare_agents.build_agent_from_cfg(cfg.agent_one)
    agent_six = bin.compare_agents.build_agent_from_cfg(cfg.agent_six)
    cf_agent = bin.compare_agents.build_agent_from_cfg(cfg.cf_agent)

    def _power_to_string(power_id):
        power_enum = conf.conf_pb2.CompareAgentsTask.Power
        return dict(zip(power_enum.values(), power_enum.keys()))[power_id]

    power_string = _power_to_string(cfg.power_one)

    result = bin.compare_agents.run_1v6_trial(
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
