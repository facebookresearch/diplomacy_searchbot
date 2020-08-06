"""HeyHi is here to configure, launch, and get results of jobs.

HeyHi is in no-way a framework or "best solution" for anything. Instead, it's
a skeleton of arguebly useful things that yolo been carrying from project to
project and its API is only stable within a project.

HeyHi has three main concepts:
    * Config - a set of parameters for a run.
    * Job - a single run.
    * Task - user's code that take a config and doing something.

Each job is uniquely identified by a folder it's "stored" in. On job start
heyhi will change cwd to this folder and user's code is expected to write all
files in this folder.

Each job is launched with a materialized config. The config is build using a
base config and dict of overrides after rendering of all the includes.

Currenly, hydra package is used for config rendering. Therefore, base config
is yaml file and overrides is a flat dict where keys are dot-separated paths
in the config and values are either scalars or lists of scalars.

To ilustrate how includes and overrides work lets consider an example. Let
assume that one has config like below in `conf/c01_supervised/yolo.yaml` and
passes the following overrides in the command line: `optimizer=sgd seed=1
model=resnet`.

Config:
   defaults:
     - optimizer: adam
     - supervised_data: equity
   seed: 0
   cfr_eval:
     cfr_binary: /checkpoint/yolo/shared/cfvpy/cfr_7017f7
     args:
       read_file: subgames/canonical_subgame1.txt
       eval_iters: 128
       num_threads: 52


`defaults` section lists default includes. Includes are small chunks of
config. E.g, there must be file `conf/common/optimizer/adam.yaml` or
`conf/c01_supervised/optimizer/adam.yaml`.

The following will happen:
  * All overrides are split into include-overrides and value-overrides.
    Includes include files. For instance, of there is file
    `conf/common/resnet.yaml` or `conf/c01_supervised/model/resnet/yaml`,
    then `model=resnet` is a include-override. Otherwise, it's a value-override.
  * If any include-overrides keys are in defaults section, then this include
    will be updated.
  * Now all default-includes and override includes happen. They are merged
    according to hydra rules that not completely clear to yolo. So it's
    better not to have conflicts among includes or between the main config
    and an include.
    In the example above optimizer/sgd.yaml, supervised_data/equity.yaml and
    model/resnet.yaml will be merged.
  * Finally, value-overrides are apply, i.e., seed will be set to 1.

A fully rendered config (that is also a yaml file) will be stored as
`config.yaml` in the job's folder.

HeyHi applies limitations on how configs should be stored. All configs are
explected to be in conf/ folder such that heyhi/ and conf/ folders are on the
same level. The conf folder is expected to have `common` folder with general
includes. It may also contain folders for groups of expetiments. The
recommended way to name that is to enumerate and tag, e.g., `c01_supervised`
and `c02_selfplay` to keep track of progress. Practice shows, that succinct
is better than verbose.

Having said this, HeyHi can run a config from whereever. It may be useful if
you want to re-run a fully rendered config of another job again for debugging
purposes.

Finally, what does "to run" mean and what is a task.
A task is a function that takes config and does some job. It can access the
config as a dict (cfg["optimizer"]["lr"]) or as an object (cfg.optimizer.lr).
There are many way to "run". The default one is use `run.py` in root of the
project. It calls many method of heyhi is right order, but not part of heyhi
as it's part of the userland.

Once it's done, one can start a run like this:

    python run.py --cfg conf/c01_supervised.main.yaml --adhoc optimizer=sgd
                        ^^^ the base config                   ^^^ override

The experiment folder name is a function of config name (but not content!)
and overrides. I.e., if you run the same experiment twice, heyhy will not run
the job again (unless you pass --mode=restart or delete the experiment folder).
With --adhoc a timestamp will be added to experiment folder and so it will be
always unique.

By default, the job is launched locally. There is a special `launcher`
section in the config that specifies where to launch a job. You can find some
default launchers in `conf/common/launcher/`. E.g., adding
`launcher=slurm_8gpus` override to command line, will start the task on
volta. If you want more, you can apply value-overrides on include-overrides :)
E.g,`launcher=slurm_8gpus launcher.num_gpus=80` will launch your job on 10
machines. [Is up to user to setup distrubuted stuff within her task.
Currently there is only `heyhi.is_master()` helper function, but fell free to
add more useful general stuff.]

Caveats:
  * There is no schema for the config. All overrides are valid, and so typos
    in overrides are not catched. On python side, if config value does not
    exist, it's None.
  * The jobs is executed from the current folder. All changes in current folder
    may affect your running jobs. In practice, it usually fine to edit python
    file after job is _scheduled_. Changing/recompiling c++ will end up
    badly.
  * Configs must contain a `task` field that matches some task in task_dict.

Things in the experiment folder:
  * Experiment folder path is printer on job start. If you started a job
    without `--adhoc`, you can run the same command again and find the
    experiment folder.
  * Experiment folder will contain slurm logs in `slurm/` folder.
  * If there was a diff on experiment launch, then `workdir.diff` is stored.
  * Current git revision is printed in the log. (see *err in slurm/)

TODO(yolo): explain non-adhoc runs and --mode.
TODO(yolo): explain sweeps/group runs.
TODO(yolo): explain export to google-sheet.
"""
from typing import Callable, Sequence, Optional
import argparse
import logging
import os
import pathlib
import pprint

import torch

from . import util

# This hardcoding is terrible. But as HeyHi is not a framework, it's fine. The
# reason for hardcoding, it that sweeps call `maybe_launch` here, while adhoc
# runs use `parse_args_and_maybe_launch` in user's run.py. And it's trickier to
# extract project name from the userland then from here.
# This constant is only used to define where to store logs.
PROJECT_NAME = "fairdiplomacy"


def get_exp_dir(project_name) -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("HH_EXP_DIR", f"/checkpoint/{os.environ['USER']}/{project_name}/heyhi")
    )


def get_default_exp_dir():
    return get_exp_dir(PROJECT_NAME)


def parse_args_and_maybe_launch(main: Callable) -> None:
    """Main entrypoint to HeyHi.

    It does eveything HeyHi is for:
        * Finds a config.
        * Applies local and global includes and overridefs to the config.
        * Determindes a folder to put the experiment.
        * Detects the status of the experiment (if already running).
        * Depending on the status and mode, maybe start/restarts experiment
        locally or remotely.

    Args:
        task: dictionary of callables one of which will be called based on
            cfg.task.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cfg", required=True, type=pathlib.Path)
    parser.add_argument("--adhoc", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Do not ask confirmation in restart mode"
    )
    parser.add_argument(
        "--mode",
        choices=util.MODES,
        default="gentle_start",
        help="See heyhi/util.py for mode definitions.",
    )
    parser.add_argument(
        "--exp_id_pattern_override",
        default=util.EXP_ID_PATTERN,
        help="A pattern to construct exp_id. Job's data is stored in <exp_root>/<exp_id>",
    )
    args, overrides = parser.parse_known_args()

    overrides = [x.lstrip("-") for x in overrides]

    maybe_launch(main, exp_root=get_exp_dir(PROJECT_NAME), overrides=overrides, **vars(args))


def maybe_launch(
    main: Callable,
    *,
    exp_root: Optional[pathlib.Path],
    overrides: Sequence[str],
    cfg: pathlib.Path,
    mode: util.ModeType,
    adhoc: bool = False,
    exp_id_pattern_override=None,
    force: bool = False,
) -> util.ExperimentDir:
    """Computes the task locally or remotely if neeeded in the mode.

    This function itself is always executed locally.

    The function checks the exp_handle first to detect whether the experiment
    is running, dead, or dead. Depending on that and the mode the function
    may kill the job, wipe the exp_handle, start a computation or do none of
    this.

    See handle_dst() for how the modes and force are handled.

    The computation may run locally or on the cluster depending on the
    launcher config section. In both ways main(cfg) with me executed with the
    final config with all overrides and substitutions.
    """
    util.setup_logging()
    logging.info("Config: %s", cfg)
    logging.info("Overrides: %s", overrides)

    if exp_root is None:
        exp_root = get_exp_dir(PROJECT_NAME)

    exp_id = util.get_exp_id(cfg, overrides, adhoc, exp_id_pattern=exp_id_pattern_override)
    exp_handle = util.ExperimentDir(exp_root / exp_id, exp_id=exp_id)
    need_run = util.handle_dst(exp_handle, mode, force=force)
    logging.info("Exp dir: %s", exp_handle.exp_path)
    logging.info("Job status [before run]: %s", exp_handle.get_status())
    if need_run:
        util.run_with_config(main, exp_handle, cfg, overrides)
    if exp_handle.is_done():
        result = torch.load(exp_handle.result_path)
        if result is not None:
            simple_result = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
            pprint.pprint(simple_result, indent=2)
    return exp_handle
