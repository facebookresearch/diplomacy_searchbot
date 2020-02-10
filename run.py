import logging
import os

import fairdiplomacy.bin.compare_agents
import heyhi


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def compare_agents(cfg):
    def _instantiate(agent_stanza):
        builder = getattr(fairdiplomacy.bin.compare_agents, agent_stanza.classname)
        kwargs = agent_stanza.kwargs or {}
        return builder(**kwargs)

    agent_one = _instantiate(cfg.agent_one)
    agent_six = _instantiate(cfg.agent_six)
    cf_agent = _instantiate(cfg.cf_agent)

    result = fairdiplomacy.bin.compare_agents.run_1v6_trial(
        agent_one,
        agent_six,
        cfg.power_one,
        save_path=cfg.out if cfg.out else None,
        seed=cfg.seed,
        cf_agent=cf_agent,
    )
    logging.warning("Result: {}".format(result))


@heyhi.save_result_in_cwd
def main(cfg):
    heyhi.setup_logging()
    logging.info("CWD: %s", os.getcwd())
    logging.info("cfg:\n%s", cfg.pretty())
    heyhi.log_git_status()
    logging.info("Is on slurm: %s", heyhi.is_on_slurm())
    if heyhi.is_on_slurm():
        logging.info("Slurm job id: %s", heyhi.get_slurm_job_id())
    logging.info("Is master: %s", heyhi.is_master())

    if cfg.task is None:
        raise ValueError('Config has to define "task" field')
    if cfg.task not in TASKS:
        raise ValueError("Unknown task: %s. Known tasks: %s" % (cfg.task, sorted(TASKS)))
    return TASKS[cfg.task](cfg)


if __name__ == "__main__":
    heyhi.parse_args_and_maybe_launch(main)
