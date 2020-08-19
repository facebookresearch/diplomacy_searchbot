#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to analyze a ParlAI model on a set of test cases.

It outputs a model's predictions on a set of test cases

Example usage:
```
diplom dgop -mf /checkpoint/wyshi/diplomacy/Bart/Bart_diplomacy_lower_lr/18e/model --datapath /private/home/wyshi/ParlAI/data --game-format viz2dip
```
"""
from parlai.core.agents import create_agent
from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.display_model import setup_args as setup_args_dm
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.utils.game_loading as game_load
import parlai_diplomacy.utils.game_to_sequence_formatting as formatter
import parlai.utils.logging as logging

import pathlib
import json

load.register_all_agents()
load.register_all_tasks()


TESTCASE_PATH = str(pathlib.Path(__file__).parent.parent.parent / "test_situations.json")


def setup_args():
    parser = setup_args_dm()
    parser.add_argument(
        "--test-situation-path",
        type=str,
        default=TESTCASE_PATH,
        help="Path to test case, see default at fairdiplomacy/test_situations.json",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="state_order",
        help="Format that the data is expected to be in; for example, state_order takes state as input and returns an order",
    )
    parser.add_argument(
        "--game-format",
        type=str,
        default="viz2dipcc",
        choices={"sql", "viz2dipcc", "viz2dip"},
        help="Format that the game is loaded in",
    )
    parser.set_params(
        interactive_mode=True, skip_generation=False,
    )
    return parser


def order_prob(order_lst, order, second_order=None):
    """
    Hack since we aren't returning probabilities
    """
    if order in order_lst:
        if second_order is not None:
            if second_order in order_lst:
                return 1.0
            else:
                return 0.0
        return 1.0
    else:
        return 0.0


def load_game(opt, path):
    """
    Load the game from the specified path in the specified format
    """
    if opt["game_format"] == "viz2dip":
        game = game_load.load_viz_to_dip_format(path)
    elif opt["game_format"] == "viz2dipcc":
        game = game_load.load_viz_to_dipcc_format(path)
    else:
        # SQL format
        game = game_load.load_single_sql_game(path)
    game = game_load.organize_game_by_phase(game)
    # convert to format expected by the model
    format_sequences = formatter.SequenceFormatHelper.change_format(game, opt["data_format"])

    return format_sequences


def get_model_pred(agent, input_seq):
    """
    Return the model's prediction for an input sequence
    """
    ex = {
        "text": input_seq,
        "episode_done": True,
    }
    agent.observe(ex)
    response = agent.act()["text"]
    return response


def run_tests(power, response, tests):
    """
    Check tests for a given response from the model
    """
    for test_name, test in tests.items():
        if test_name[:3] != power[:3]:  # not relevant to this power
            continue
        orders = formatter.order_seq_to_fairdip(response)
        func = eval(test)
        success = func(orders)
        if not success:
            logging.error(f"\tFAIL: {test_name}")
        else:
            logging.success(f"\tPASS: {test_name}")


def display_game_order_preds(opt):
    agent = create_agent(opt)  # create model agent

    datapath = opt["test_situation_path"]
    with open(datapath, "r") as f:
        data = json.load(f)

    for situation, info in data.items():
        game_json_path = info["game_path"]
        phase_name = info["phase"]
        logging.warn(f"\n\nTesting situation: {situation} ...")
        game = load_game(opt, game_json_path)
        phase = game[phase_name]

        for power, input_seq in phase.items():
            response = get_model_pred(agent, input_seq)
            logging.info(f"{power}: {response}")
            run_tests(power, response, info["tests"])


@register_script("display_game_order_preds", aliases=["dgop"])
class DisplayGameOrderPreds(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        display_game_order_preds(self.opt)


if __name__ == "__main__":
    DisplayGameOrderPreds.main()
