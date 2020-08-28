#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to aggregate the validation outputs to one single json, 
run this script only after after running distributed_eval.py (or other evaluation sweeps) with world logging,
only need to be run once after every evaluation
Example usage 
```
python aggregate_valid_reports.py --sweep_name save_diplomacy_validation_bart --teacher state_order_chunk --date_of_sweep 20200804
```
args:
--sweep_name: validation sweep name, 
--teacher: validation teacher name
--date_of_sweep: date of the validation sweep
"""
import json
from tqdm import tqdm
from glob import glob
import re
import os
import sys
import time
import numpy as np
from copy import deepcopy
from datetime import date
import argparse

import parlai.utils.logging as logging
import parlai_diplomacy.utils.game_loading as game_loading
import parlai_diplomacy.tasks.common_task_utils as utls


# valid_json save path
VALID_SAVE_ROOT = "/checkpoint/fairdiplomacy/press_diplomacy/validation/validation_report/"


def get_valid_report_path(sweep_name, data_type, teacher, date_of_sweep=None):
    """
    return report related paths
    """
    # save path
    USER = os.environ["USER"]
    DATE = date_of_sweep if date_of_sweep else str(date.today()).replace("-", "")
    # report paths
    VALID_REPORT_SAVE_PATH = os.path.join(
        VALID_SAVE_ROOT, USER, DATE, sweep_name, teacher, "valid_report"
    )
    # jsonl paths
    VALID_REPORT_JSONL_PATH = os.path.join(
        VALID_SAVE_ROOT, USER, DATE, sweep_name, teacher, "valid_report", "*_replies.jsonl"
    )
    # combined json path
    VALID_REPORT_SAVE_JSON_PATH = os.path.join(
        VALID_SAVE_ROOT,
        USER,
        DATE,
        sweep_name,
        teacher,
        "combined_parlai_valid_set_prediction.json",
    )

    return VALID_REPORT_SAVE_PATH, VALID_REPORT_JSONL_PATH, VALID_REPORT_SAVE_JSON_PATH


def convert_test_results(sweep_name, teacher, date_of_sweep=None, predict_all_order=False):
    """
    gather valid reports --> output one json for evaluation and comparison
    """
    # get validation report path
    _, VALID_REPORT_JSONL_PATH, VALID_REPORT_SAVE_JSON_PATH = get_valid_report_path(
        sweep_name, teacher, date_of_sweep
    )

    # read jsonl
    paths = glob(VALID_REPORT_JSONL_PATH)
    final_data = {}
    raw_data = []
    for i, path in enumerate(paths):
        print(f"[Loading ...{i+1}/{len(paths)}: {path} ...]")
        raw_data.extend(load_jsonl(path))

    # initialize some stats
    total = 0
    total_without_key_error = 0
    correct = 0
    key_error = 0
    for data in raw_data:
        total += 1
        try:
            total_without_key_error += 1
            ground_truth = data["dialog"][0][0]
            game_id, phase_id, speaker_id, order = (
                ground_truth["game_id"],
                ground_truth["phase_id"],
                int(ground_truth["player_id"]),
                ground_truth["eval_labels"][0],
            )

            if predict_all_order:
                order = [
                    s.replace(ground_truth["player"] + ": ", "")
                    for s in order.split("\n")
                    if s.startswith(ground_truth["player"])
                ][0]
                predicted_order = [
                    s.replace(ground_truth["player"] + ": ", "")
                    for s in data["dialog"][0][-1]["text"].split("\n")
                    if s.startswith(ground_truth["player"])
                ][0]
            else:
                predicted_order = data["dialog"][0][-1]["text"]
            order = order.replace("[EO_O]", "").strip()
            predicted_order = predicted_order.replace("[EO_O]", "").strip()

            if order == predicted_order:
                correct += 1

            if game_id in final_data:
                if phase_id in final_data[game_id]:
                    if speaker_id in final_data[game_id][phase_id]:
                        assert ValueError("repeated speaker_id")
                    else:
                        final_data[game_id][phase_id][
                            utls.COUNTRY_ID_TO_POWER[speaker_id]
                        ] = predicted_order
                else:
                    final_data[game_id][phase_id] = {
                        utls.COUNTRY_ID_TO_POWER[speaker_id]: predicted_order
                    }
            else:
                final_data[game_id] = {
                    phase_id: {utls.COUNTRY_ID_TO_POWER[speaker_id]: predicted_order}
                }
        except KeyError as e:
            # some has key error, possibly due to partial games
            total_without_key_error -= 1
            print(f"I got a KeyError - reason {e}")
            key_error += 1

    # just for reference, these metrics cannot be directly compared against fairdip
    print(f"order acc: {correct/total_without_key_error} ({correct}/{total_without_key_error})")
    print(f"key error: {key_error/total} ({key_error}/{total})")

    with open(VALID_REPORT_SAVE_JSON_PATH, "w") as fh:
        json.dump(final_data, fh)

    print(f"aggregated validation json saved to {VALID_REPORT_SAVE_JSON_PATH}")


def load_jsonl(path):
    """
    load jsonl files
    """
    with open(path, "r") as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        result = json.loads(json_str)
        results.append(result)
    return results


def main(sweep_name, teacher, date_of_sweep=None, predict_all_order=False):
    convert_test_results(sweep_name, teacher, date_of_sweep, predict_all_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_name", type=str, help="validation sweep name",
    )
    parser.add_argument(
        "--teacher", type=str, help="validation teacher name",
    )
    parser.add_argument(
        "--date_of_sweep", type=str, help="date of the validation sweep",
    )
    parser.add_argument(
        "--predict_all_order", action="store_true", help="if the report is for all orders",
    )

    args = parser.parse_args()

    # save to json
    main(args.sweep_name, args.teacher, args.date_of_sweep, args.predict_all_order)
