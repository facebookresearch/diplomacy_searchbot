#!/usr/bin/env python
"""
This script is indended to be used to launch bots on the AWS cluster
"""
import argparse
import subprocess

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

BASE_EXP_DIR = "/home/jsgray/exp"
REPO = "/home/jsgray/code/fairdiplomacy"
# MODEL_PATH = "/home/jsgray/cfr2_cfr_data_400.pth"
MODEL_PATH = "/home/jsgray/sl_candidx_B2.5k_vclip1e-7.pth"
HOST = "10.100.34.208"

SBATCH_SCRIPT = """#!/bin/bash

mkdir -p {exp_dir}

srun --output {exp_dir}/out.log --error {exp_dir}/out.log -- \
    python {REPO}/run.py \
        --cfg {REPO}/conf/c03_launch_bot/launch_bot.prototxt \
        --exp_id_pattern_override={exp_dir} \
        host={HOST} \
        I.agent=agents/cfr1p \
        agent.cfr1p.model_path={MODEL_PATH} \
        agent.cfr1p.max_rollout_length=5 \
        agent.cfr1p.mix_square_ratio_scoring=0.1 \
        agent.cfr1p.postman_sync_batches=False \
        agent.cfr1p.n_plausible_orders=24 \
        agent.cfr1p.max_actions_units_ratio=2.5 \
        agent.cfr1p.average_n_rollouts=3 \
        agent.cfr1p.rollout_temperature=0.5 \
        agent.cfr1p.n_rollout_procs=168 \
        agent.cfr1p.rollout_top_p=0.95 \
        game_id={game_id} \
        power={power}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id")
    parser.add_argument("--power")
    parser.add_argument("--powers-except")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()
    assert (args.power is None) ^ (args.powers_except is None), args
    assert (args.power and args.power in POWERS) or (
        args.powers_except and args.powers_except in POWERS
    ), args

    powers = [args.power] if args.power else [p for p in POWERS if p != args.powers_except]

    for power in powers:
        job_name = args.game_id + (f"_{args.suffix}" if args.suffix else "") + f"_{power[:3]}"
        exp_dir = f"{BASE_EXP_DIR}/{job_name}"

        sbatch_script = SBATCH_SCRIPT.format(
            REPO=REPO,
            MODEL_PATH=MODEL_PATH,
            HOST=HOST,
            exp_dir=exp_dir,
            game_id=args.game_id,
            power=power,
        )
        p = subprocess.run(
            [
                "sbatch",
                "--job-name",
                job_name,
                "--cpus-per-task=32",
                "--gpus=1",
                "--mem=0",
                "--time=1440",
                # f"--chdir={exp_dir}",
            ],
            check=True,
            input=sbatch_script,
            text=True,
        )
