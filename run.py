import logging
import os

import bin.compare_agents
import bin.train_sl
import heyhi
import conf.conf_pb2


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


def _build_agent(agent_stanza: conf.conf_pb2.Agent) -> bin.compare_agents.BaseAgent:
    agent_name = agent_stanza.WhichOneof("agent")
    if agent_name == "mila_agent":
        return bin.compare_agents.MilaSLAgent()
    else:
        raise RuntimeError(f"Unknown agent: {agent_name}")


@_register
def compare_agents(cfg):
    agent_one = _build_agent(cfg.agent_one)
    agent_six = _build_agent(cfg.agent_six)
    cf_agent = _build_agent(cfg.cf_agent)

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
    bin.train_sl.run_with_cfg(cfg)


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
