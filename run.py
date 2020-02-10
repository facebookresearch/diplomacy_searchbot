import logging

import heyhi


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

    task_dict = {}

    if cfg.task is None:
        raise ValueError('Config has to define "task" field')
    if cfg.task not in task_dict:
        raise ValueError("Unknown task: %s. Known tasks: %s" % (cfg.task, sorted(task_dict)))
    task = getattr(task_dict, cfg.task)
    return task(cfg)


if __name__ == "__main__":
    heyhi.parse_args_and_maybe_launch(main)
