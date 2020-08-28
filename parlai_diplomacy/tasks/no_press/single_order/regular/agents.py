#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.dict import DictionaryAgent
from parlai.core.teachers import FixedDialogTeacher
from parlai.utils import misc

import parlai_diplomacy.tasks.common_task_utils as utls

import json
import os
import sys
import random
from tqdm import tqdm
import numpy as np


"""
File for board state data -> Order, NOT streaming.

For streaming data, please see stream/
"""


@register_teacher("base_order")
class BaseOrderTeacher(FixedDialogTeacher):
    """
    Plain diplomacy (message-order) teacher.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        # add common arguments
        argparser = utls.add_common_args(argparser)

        # add teacher-specific arguments
        argparser.add_argument(
            "--include-message-from",
            type=str,
            default="all_msg",
            choices={"speaker_msg_only", "partner_msg_only", "all_msg"},
            help="whose messages to include (speaker-only, listener-only, or all messages)",
        )
        argparser.add_argument(
            "--overwrite-joined-json", action="store_true", help="Overwrite the joined json",
        )
        argparser.add_argument(
            "--load-splitted-valid",
            action="store_true",
            help="the validation set is splitted, do we want to load the splitted set or build the validation set from scratch",
        )
        argparser.add_argument(
            "--test-file-id", type=str, help="test file chunk id",
        )
        argparser.add_argument(
            "--with-special-token",
            action="store_true",
            help="with the added special tokens with underscores",
        )
        argparser.add_argument(
            "--train-valid-split-by",
            type=str,
            default="game",
            choices={"game", "data_point_with_msg",},
            help="how to split the train/validation set, by games (used for comparison with fairdip) or \
                data_point_with_msg (cannot be used for fairdip, used only to see if the msg data is helping)",
        )
        argparser.add_argument(
            "--train-val-split-pct",
            type=float,
            default=0.95,
            help="train/validation split percent",
        )
        argparser.add_argument(
            "--debug",
            action="store_true",
            help="debug mode to load fewer games (--debug-game-size)",
        )
        argparser.add_argument(
            "--debug-game-size", type=int, default=500, help="how many games to use in debug mode",
        )
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.dt = opt["datatype"].split(":")[0]
        self.min_turns = opt["min_turns"]
        self.include_message_from = opt["include_message_from"]
        self.overwrite_joined_json = opt["overwrite_joined_json"]
        self.with_special_token = opt["with_special_token"]

        self.valid_json_dir = utls.VALID_JSON_PATH
        self.is_train = utls.is_training(opt["datatype"])

        if shared is None:
            # set map
            if opt["load_splitted_valid"]:
                self.data = self._load_valid_data(opt)
            else:
                self.data = self._setup_data(opt)
            self.num_exs = len(self.data)
        else:
            self.data = shared["data"]
            self.num_exs = shared["num_exs"]

        super().__init__(opt, shared)
        self.reset()

    def _print_example_data(self):
        nl = "\n"
        examples = [
            f"text_{i}: {self._construct_msg(self.data[i], 0)['text']}{nl}labels_{i}: "
            f"{self._construct_msg(self.data[i], 0)['labels']}{nl}game_{i}: {self._construct_msg(self.data[i], 0)['game']}{nl}"
            f"game_{i}: {self._construct_msg(self.data[i], 0)['game_phase']}{nl}{nl}"
            for i in range(5)
        ]
        example_data_text = f"example data{nl} {nl.join(examples)}"
        misc.warn_once(example_data_text)

    def _replace_with_special_token(self, msg):
        # should be used when the saved joined_json is 'A BAL',
        # 'A BAL' --> '__A__ __BAL__'
        # TODO depreciate this once we've saved the joins_json with __
        if self.with_special_token:
            msg["order"] = utls.replace_with_special_token(msg["order"])
            msg["state"] = utls.replace_with_special_token(msg["state"])

        return msg

    def _get_train_val_split_num(self, TRAIN_VAL_SPLIT_PERCENT):
        if self.opt["train_valid_split_by"] == "game":
            print(f"validation set are on game-level, should be used for fairdip comparison!")
            print(f"Total dataset Diplomacy (message+order) games: {self.iterator.num_games}")
            TRAIN_SPLIT_NUM = int(self.iterator.num_games * TRAIN_VAL_SPLIT_PERCENT)
            VALID_SPLIT_NUM = self.iterator.num_games - TRAIN_SPLIT_NUM
            print(f"Total dataset Diplomacy (message+order) games for training: {TRAIN_SPLIT_NUM}")
            print(
                f"Total dataset Diplomacy (message+order) games for validation: {VALID_SPLIT_NUM}"
            )
        elif self.opt["train_valid_split_by"] == "data_point_with_msg":
            print(
                f"validation set contains pairs with non-empty message only! Shouldn't be used for fairdip comparison"
            )
            print(f"Total dataset Diplomacy (message+order) pairs: {self.iterator.num_pairs}")
            TRAIN_SPLIT_NUM = int(self.iterator.num_pairs * TRAIN_VAL_SPLIT_PERCENT)
            VALID_SPLIT_NUM = self.iterator.num_pairs - TRAIN_SPLIT_NUM
            print(f"Total dataset Diplomacy (message+order) pairs for training: {TRAIN_SPLIT_NUM}")
            print(
                f"Total dataset Diplomacy (message+order) pairs for validation: {VALID_SPLIT_NUM}"
            )

        return TRAIN_SPLIT_NUM, VALID_SPLIT_NUM

    def _construct_common_msg_fields(self, msg):
        # speaker token
        game_phase = msg["game_phase"]
        speaker_token = (
            f"{game_phase} {msg['speaker'].capitalize()}:"  # looks like "1901M France:"
        )

        # new_msg
        new_msg = {
            "labels": [msg["order"]],
            "episode_done": True,
        }

        # other fields
        for k, v in msg.items():
            new_msg[k] = v

        return speaker_token, new_msg

    def _construct_msg(self, pair, idx):
        # individual teacher should implement this
        raise NotImplementedError("Please implement _construct_msg for individual teachers!")

    def _setup_data(self, opt):
        # TODO: set in to train/test/valid split (confirm with group)
        # Load data iterator
        self.iterator = utls.MessageOrderDataIterator(opt)

        # get train/valid split number
        TRAIN_VAL_SPLIT_PERCENT = opt["train_val_split_pct"]
        TRAIN_SPLIT_NUM, VALID_SPLIT_NUM = self._get_train_val_split_num(TRAIN_VAL_SPLIT_PERCENT)

        # initialize game and phase stats
        tot_phases, tot_pairs, tot_valid_pairs = 0, 0, 0
        game_ids, game_phase_ids = [], []
        # Run through all phases to get a list of (message-order) pairs
        pairs = []
        for i, phase in enumerate(tqdm(self.iterator)):
            if opt["train_valid_split_by"] == "game":
                if "train" in opt["datatype"]:
                    if self.iterator.game_idx >= TRAIN_SPLIT_NUM:
                        break
                else:
                    # test/valid are the same for now
                    # the last games are valid
                    # TODO: change
                    if self.iterator.game_idx < TRAIN_SPLIT_NUM:
                        continue
            elif opt["train_valid_split_by"] == "data_point_with_msg":
                if "train" not in opt["datatype"]:
                    # the first games with msg are valid
                    if tot_valid_pairs >= VALID_SPLIT_NUM:
                        break

            # book keeping
            tot_phases += 1
            game_ids.append(self.iterator.game_idx)
            game_phase_ids.append(f"{self.iterator.game_idx}-{self.iterator.phase_idx}")
            for pair in phase:
                # train/valid split condition, the validation dataset contains datapoints with msg only
                # the first games with msg are valid
                if opt["train_valid_split_by"] == "data_point_with_msg":
                    if "train" in opt["datatype"]:
                        # the top VALID_SPLIT_NUM "Game_Phase_Msg" will be validation
                        if (
                            pair["data_status"] == "Game_Phase_Msg"
                            and tot_valid_pairs < VALID_SPLIT_NUM
                        ):
                            tot_valid_pairs += 1
                            continue
                        else:
                            pass
                    else:
                        if (
                            pair["data_status"] == "Game_Phase_Msg"
                            and tot_valid_pairs < VALID_SPLIT_NUM
                        ):
                            tot_valid_pairs += 1
                            pass
                        else:
                            if pair["data_status"] != "Game_Phase_Msg":
                                continue
                            elif tot_valid_pairs >= VALID_SPLIT_NUM:
                                break

                # include this pair in this datatype
                pairs.append(pair)

                # book keeping
                tot_pairs += 1

        if self.is_train:
            random.shuffle(pairs)

        return pairs

    def _load_valid_data(self, opt):
        print(f"load valid_split_{opt['test_file_id']}.json")
        with open(
            f"{utls.TEST_SPLIT_JSON_PATH}/valid_split_{opt['test_file_id']}.json", "r"
        ) as fh:
            pairs = json.load(fh)
        return pairs

    def get(self, episode_idx, entry_idx=0):
        ex = self._construct_msg(self.data[episode_idx], entry_idx)
        return Message(ex)

    def num_examples(self):
        # fix this
        return self.num_exs

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared["data"] = self.data
        shared["num_exs"] = self.num_exs
        return shared


@register_teacher("state_order")
class StateOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "state_order"
        new_msg["text"] = f"{new_msg['state']} {speaker_token}"

        return new_msg


@register_teacher("speaker_token_order")
class SpeakerTokenOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "speaker_token_order"
        new_msg["text"] = f"{speaker_token}"

        return new_msg


@register_teacher("dummy_token_order")
class DummyTokenOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "dummy_token_order"
        new_msg["text"] = f"UNK"

        return new_msg
