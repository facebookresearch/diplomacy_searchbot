#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Parameter sweep utilities.

Contains basic helper functions for running a parameter sweep on the FAIR cluster using
the SLURM scheduler.
"""

from typing import Optional, Dict, Any, Set

from collections import namedtuple
import json
import os
import subprocess
import sys
import random
import hashlib

import parlai

DEFAULT_PARLAI_PATH = os.path.dirname(os.path.dirname(parlai.__file__))
BASH_IF_CLAUSE = """
if [[ "$SLURM_ARRAY_TASK_ID" == "{index}" ]]; then
    # -K kills all subtasks if one particular task crashes. This is necessary for
    # distributed training
    srun -K1 bash {SAVE}/run.sh >> {SAVE}/stdout.${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} 2>> {SAVE}/stderr.${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
fi
"""
SLRM_JOB_ARRAY_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={SWEEP_NAME}
#SBATCH --output={SAVE_ROOT}/slurm_logs/slrm_stdout.%j
#SBATCH --error={SAVE_ROOT}/slurm_logs/slrm_stderr.%j
#SBATCH --partition={partition}
## make sure we don't clobber log files if jobs get restarted
#SBATCH --open-mode=append
#SBATCH --nodes={nodes}
#SBATCH --time={jobtime}
## make sure we are told about preempts, and jobs running out of time, 5 min beforehand
#SBATCH --signal=USR1@60
## number of cpus *per task*. Highly recommend this to be 10.
#SBATCH --cpus-per-task={cpus}
## srun forks ntasks_per_node times on each node
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={mem_gb}G
{SBATCH_EXTRAS}

echo
nvidia-smi
echo "# -------- BEGIN CALL TO run.sh --------"
{JOB_LAUNCHER}
echo
nvidia-smi
"""

SH_TEMPLATE = """
#!/bin/bash
set -e

# stores the child process
CHILD=""

# handles a TERM signal
term_handler () {{
    # catch and ignore TERM. we get multiple terms during shutdown, so best
    # to just do nothing
    # but still keep going with the python process
    wait "$CHILD"
}}

# handles an interrupt (aka ctrl-C)
int_handler () {{
    # propagate a ctrl-C to the python process
    kill -s INT "$CHILD"
    wait "$CHILD"
}}

# handles a USR1, which signals preemption or job time is up
usr1_handler () {{
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)."
    kill -s INT "$CHILD"  # send ctrl-c to python
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then
        echo "Resubmitting ${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}..."
        scontrol requeue ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
        echo "Successfully resubmitted."
    fi
    wait "$CHILD"
    exit 1
}}

trap 'int_handler' INT
trap 'usr1_handler' USR1
trap 'term_handler' TERM

echo "Running $0 on $(hostname)"

if [ "$SLURM_PROCID" -eq "0" ]; then
    env
    echo ----------------------------------------------------------------------------
fi

# Uncommenting these two lines can help with identifying hangs
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# setting this this can also help with hangs
# NCCL_LL_THRESHOLD=0

# if in distributed, make sure we're using the actual network
export NCCL_SOCKET_IFNAME=^docker0,lo
cd {NEW_PARLAI_PATH}
export PYTHONPATH={SAVE_ROOT}/ParlAI:$PYTHONPATH
export PARLAI_DATAPATH={PARLAI_DATAPATH}
export PARLAI_DOWNPATH={PARLAI_DOWNPATH}
{python_cmd} &
CHILD="$!"
wait "$CHILD"
RETVAL=$?
sleep 30
exit $RETVAL
"""


def sha1(string):
    """
    Compute the sha1 hexdigest of the string.
    """
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def run_grid(
    grid: Dict[str, Any],
    name_keys: Set[str],
    sweep_name: str,
    user: str = os.environ['USER'],
    prefix: str = 'python -u scripts/train.py',  # NOTE: I changed this here from parlai_internal
    gpus: int = 1,
    cpus: int = 10,
    nodes: int = 1,
    node_exclude=None,
    partition: str = 'learnfair',
    PARLAI_PATH: str = DEFAULT_PARLAI_PATH,
    jobtime: str = '01:59:59',
    include_job_id: bool = False,
    hide_keys: bool = None,
    create_model_file: bool = True,
    hashname: bool = False,
    fixedname: bool = False,
    saveroot: Optional[str] = None,
    mem_gb: int = 64,
    requeue: bool = False,
    one_job_per_node: bool = False,
    comment: Optional[str] = None,
    volta: bool = False,
    volta32: bool = False,
    copy_env: bool = True,
    copy_dirs=('tests', 'examples', 'projects',),
    max_num_jobs=None,
    data_parallel: Optional[bool] = None,
):
    """
    Generates full commands from a grid.

    :param grid:
        keys are hyperparam strings (e.g. --learningrate or -lr), values are
        lists of parameter options (e.g. [0.5, 0.05, 0.005]).  You can tie
        options together in a limited fashion (e.g.  '--opt': ['sgd -lr 0.5',
        'adam -lr 0.005']), but we don't support nesting dicts/lists yet.
    :param name_keys:
        (set) contains any params to always include in the model
        filename (e.g. {'-hs'} will make sure that the filename includes
        _hs=X_). By default, any key with more than one value will also be
        included in the model filename.
    :param sweep_name:
        name of the sweep
    :param user:
        user name to use for save directory (default $USER)
    :param prefix:
        base command to run. defaults to running the standard train_model
        script in ParlAI/examples/.
    :param hashname:
        if True, uses a hash of the parameters as the folder. Sometimes
        necessary for long commands (default False).
    :param one_job_per_node:
        set to True if running with nn.DataParallel or with model parallel
    :param volta:
        set to True to request a volta machine
    :param volta32:
        set to True to request a 32gb volta machine
    :param comment:
        you need to add a text comment to use priority partition
    :param copy_env:
        if True, copies local ParlAI directory components to the save root, and
        uses this to run the jobs
    :param copy_dirs:
        list of additional directories (besides parlai and parlai_internal) to
        copy when copying the ParlAI dir
    :param max_num_jobs:
        maximum number of jobs
    :param data_parallel:
        alias for one_job_per_node. for backwards compatiblity.
    """

    # for backwards compatibility, respect data_parallel as true to mean
    # same as one_job_per_node
    one_job_per_node = one_job_per_node or data_parallel

    # default values that aren't mutable
    if hide_keys is None:
        hide_keys = {}

    if not hasattr(grid, 'items'):
        raise TypeError('Grid should be a dict.')

    if not one_job_per_node and (gpus > 1 or nodes > 1) and ('train_model' in prefix):
        raise ValueError(
            "Looks like you should be using distributed training. Change "
            "prefix to 'python -u -m parlai.scripts.distributed_train'."
        )

    if saveroot is None:
        SAVE_ROOT = save_root(sweep_name, user)
    else:
        SAVE_ROOT = saveroot
    Job = namedtuple('Job', ['cmd', 'name'])
    all_jobs = [Job(cmd=prefix, name='')]

    for key, args in grid.items():
        new_jobs = []
        # save_name
        save_key = key
        while save_key.startswith('-'):
            save_key = save_key[1:]
        save_key = save_key.replace('_', '')

        for job in all_jobs:
            for a in args:
                new_cmd = ' '.join((job.cmd, str(key), str(a)))
                new_name = job.name
                if (len(args) > 1 or key in name_keys) and key not in hide_keys:
                    if type(a) == str:
                        a = a.replace('_', '')
                        if ' ' in a:
                            a = a.replace(' --', '_').replace(' -', '_')
                            a = a.replace(' ', '=')
                        if save_key in {"t", "task", "et", "evaltask"}:
                            a = a.replace("parlaiinternal.projects.", "")

                    new_name += '_{}={}'.format(save_key, a)
                if hashname:
                    new_name = sha1(new_name)
                new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    # Sample grid points
    if isinstance(max_num_jobs, int) and max_num_jobs < len(all_jobs):
        all_jobs = random.sample(all_jobs, max_num_jobs)

    # shorten names if possible
    if hashname:
        # keep the names from getting too long
        full_names = [name for _, name in all_jobs]
        cutoff = i = 4
        while i < 40:
            if len(set([n[1:i] for n in full_names])) == len(full_names):
                cutoff = i
                break
            i += 1
    else:
        cutoff = None

    final_jobs = []
    job_id = 1
    for job in all_jobs:
        new_name = job.name[1:cutoff] if cutoff else job.name[1:]
        if include_job_id:
            if fixedname:
                new_name = fixedname
            new_name += '_jobid=' + str(job_id)
        if create_model_file:
            new_cmd = '{} --model-file {}/{}/model'.format(job.cmd, SAVE_ROOT, new_name)
        else:
            new_cmd = '{} '.format(job.cmd)
        final_jobs.append(Job(cmd=new_cmd, name=new_name))
        job_id += 1

    print('Example of first job:\n{}\n'.format(final_jobs[0].cmd))
    print('Your jobs will run for {}.'.format(jobtime))
    ans = input(
        'About to launch {} jobs for a total of {} GPUs. Continue? (Y/y to proceed) '.format(
            len(final_jobs), nodes * gpus * len(final_jobs)
        )
    )
    if ans.strip().lower() != 'y':
        print('Aborting...')
        sys.exit(-1)

    PARLAI_DATAPATH = os.environ.get('PARLAI_DATAPATH')
    if '-dp' in grid:
        PARLAI_DATAPATH = grid['-dp'][0]
    elif '--datapath' in grid:
        PARLAI_DATAPATH = grid['--datapath'][0]
    elif not PARLAI_DATAPATH:
        PARLAI_DATAPATH = os.path.join(PARLAI_PATH, 'data')

    PARLAI_DOWNPATH = os.environ.get('PARLAI_DOWNPATH')
    if '--download-path' in grid:
        PARLAI_DOWNPATH = grid['--download-path'][0]
    elif not PARLAI_DOWNPATH:
        PARLAI_DOWNPATH = os.path.join(PARLAI_PATH, 'downloads')

    # NOTE: we are avoiding the copy env call here
    # if copy_env:
    if False:
        print('[ Copying env... make take up to a minute ]')
        # Make ParlAI Dir
        bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'ParlAI'))
        folders = ['parlai', 'parlai_internal', 'parlai_fb']
        for folder in folders:
            bash(
                f'rsync -av {PARLAI_PATH}/{folder} '
                f'{SAVE_ROOT}/ParlAI --exclude mturk --exclude .git --exclude "*.ipynb" '
                f'--exclude messenger --exclude data --exclude heroku-cli-*" '
            )
        to_copy = ['setup.py', 'README.md', 'LICENSE', 'requirements.txt']
        to_copy += copy_dirs
        for c in to_copy:
            c_head, c_tail = os.path.split(c)
            if len(c_head) > 1:
                bash('mkdir {SAVE_ROOT}/ParlAI/{c_head}'.format(**locals()))
            bash('cp -r {PARLAI_PATH}/{c} {SAVE_ROOT}/ParlAI/{c}'.format(**locals()))
        NEW_PARLAI_PATH = '{SAVE_ROOT}/ParlAI'.format(**locals())
    else:
        NEW_PARLAI_PATH = PARLAI_PATH

    # Dump grid to grid file
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    with open(os.path.join(SAVE_ROOT, 'grid.json'), 'w') as f:
        json.dump(grid, f)

    # shuffle jobs so we're not systematically doing them in any order
    random.shuffle(final_jobs)
    # remove job array list if it already existed
    jobs_path = []
    for job in final_jobs:
        jobs_path.append(
            create_job_files(
                sweep_name,
                SAVE_ROOT,
                job.name,
                job.cmd,
                gpus=gpus,
                nodes=nodes,
                one_job_per_node=one_job_per_node,
                requeue=requeue,
                PARLAI_DATAPATH=PARLAI_DATAPATH,
                PARLAI_DOWNPATH=PARLAI_DOWNPATH,
                NEW_PARLAI_PATH=NEW_PARLAI_PATH,
            )
        )
    submit_array_jobs(
        SWEEP_NAME=sweep_name,
        SAVE_ROOT=SAVE_ROOT,
        gpus=gpus,
        cpus=cpus,
        nodes=nodes,
        node_exclude=node_exclude,
        partition=partition,
        jobtime=jobtime,
        PARLAI_PATH=PARLAI_PATH,
        mem_gb=mem_gb,
        requeue=requeue,
        one_job_per_node=one_job_per_node,
        comment=comment,
        volta=volta,
        volta32=volta32,
        PARLAI_DATAPATH=PARLAI_DATAPATH,
        PARLAI_DOWNPATH=PARLAI_DOWNPATH,
        NEW_PARLAI_PATH=NEW_PARLAI_PATH,
        jobs_path=jobs_path,
    )


def bash(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip('b').strip('\'').strip('"')
    return output


def save_root(SWEEP_NAME, unixname):
    """
    Return root folder for saving model files, stdout, stderr, etc.
    """
    DATE = bash('date +"%Y%m%d"')
    SAVE_ROOT = os.path.join('/checkpoint/', str(unixname), DATE, SWEEP_NAME)
    return SAVE_ROOT


def create_job_files(
    SWEEP_NAME,
    SAVE_ROOT,
    param_name,
    python_cmd,
    gpus=1,
    nodes=1,
    one_job_per_node=False,
    requeue=False,
    PARLAI_DATAPATH="~/ParlAI/data",
    PARLAI_DOWNPATH="~/ParlAI/downloads",
    NEW_PARLAI_PATH="~/ParlAI",
):
    """
    Creates job folders and scripts.
    """
    SHOULD_REQUEUE = str(requeue).lower()
    SAVE = os.path.join(SAVE_ROOT, param_name)
    bash('mkdir -p ' + SAVE)
    SCRIPTFILE = os.path.join(SAVE, 'run.sh')
    ARRAYJOBFILE = os.path.join(SAVE_ROOT, 'array_jobs')

    if one_job_per_node or not gpus:
        ntasks_per_node = 1
    else:
        ntasks_per_node = gpus
    if nodes > 1 or (gpus > 1 and not one_job_per_node):
        python_cmd += " --distributed-world-size " + str(nodes * ntasks_per_node)
    with open(SCRIPTFILE, 'w') as fw:
        fw.write(SH_TEMPLATE.format(**locals()).lstrip())
    return SAVE


def submit_array_jobs(
    SWEEP_NAME,
    SAVE_ROOT,
    gpus=1,
    cpus=1,
    nodes=1,
    node_exclude=None,
    partition='learnfair',
    jobtime='01:59:59',
    PARLAI_PATH="~/ParlAI/",
    mem_gb=64,
    requeue=False,
    comment=None,
    volta=False,
    volta32=False,
    PARLAI_DATAPATH="~/ParlAI/data",
    PARLAI_DOWNPATH="~/ParlAI/downloads",
    NEW_PARLAI_PATH="~/ParlAI",
    jobs_path=None,
    one_job_per_node=None,
):
    assert jobs_path is not None

    SLURMFILE = os.path.join(SAVE_ROOT, 'run.slrm')
    if one_job_per_node or not gpus:
        ntasks_per_node = 1
    else:
        ntasks_per_node = gpus
    SBATCH_EXTRAS = []
    if node_exclude is not None:
        # If any nodes are down, exclude them here
        SBATCH_EXTRAS.append('#SBATCH --exclude ' + str(node_exclude))

    constraints = []

    total_num_jobs = len(jobs_path) - 1

    # Request the number of GPUs (defaults to 1)
    if gpus > 0:
        gpustr = f'#SBATCH --gpus-per-node={gpus}'
        if volta:
            constraints.append('volta')
        if volta32:
            constraints.append('volta32gb')
        SBATCH_EXTRAS.append(gpustr)

    if constraints:
        SBATCH_EXTRAS.append("#SBATCH -C '{}'".format('&'.join(constraints)))

    if comment:
        SBATCH_EXTRAS.append('#SBATCH --comment="{}"'.format(comment))

    # make sure sbatch extras are a string
    SBATCH_EXTRAS = "\n".join(SBATCH_EXTRAS)
    JOB_LAUNCHER = []
    for idx, each_path in enumerate(jobs_path):
        JOB_LAUNCHER.append(BASH_IF_CLAUSE.format(index=idx, SAVE=each_path))
    JOB_LAUNCHER = "\n".join(JOB_LAUNCHER)
    bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'slurm_logs'))
    with open(SLURMFILE, 'w') as fw:
        fw.write(SLRM_JOB_ARRAY_TEMPLATE.format(**locals()).lstrip())

    print(bash('sbatch --test-only --array=0-{} {}'.format(total_num_jobs, SLURMFILE)))
    print(bash('sbatch --array=0-{} {}'.format(total_num_jobs, SLURMFILE)))
