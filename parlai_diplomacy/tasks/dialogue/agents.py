#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import copy
from glob import glob
from typing import List, Tuple

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher, ChunkTeacher
from parlai.utils import logging

import parlai_diplomacy.tasks.common_task_utils as utls
import parlai_diplomacy.utils.datapath_constants as constants
from parlai_diplomacy.tasks.no_press.single_order.stream.agents import BaseOrderChunkTeacher

"""
File for all dialogue teachers
"""

TRAIN_SPLIT = 800_000  # (total game turns is 868_046)


@register_teacher("dialogue")
class DialogueTeacher(FixedDialogTeacher):
    """
    Plain diplomacy dialogue teacher.

    Does not use any game moves.

    Example use:
    ```
    python parlai_diplomacy/scripts/test_train_script.py -mf /tmp/test_stuff -m transformer/generator -t diplomacy_test
    ```
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = utls.add_common_dialogue_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.min_turns = opt["min_turns"]
        self.is_train = utls.is_training(opt["datatype"])

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            self.num_exs = sum(len(conv) for conv in self.data)
        else:
            self.data = shared["data"]
            self.num_exs = shared["num_exs"]
        super().__init__(opt, shared)
        self.reset()

    def _construct_msg(self, conv, idx):
        msg = conv[idx]
        new_msg = {
            "text": msg["input"],
            "labels": [msg["response"]],
            "episode_done": idx >= (len(conv) - 1),
        }
        for k, v in msg.items():
            if k not in ["input", "response"]:
                new_msg[k] = v

        return new_msg

    def _setup_data(self, opt):
        # TODO: set in to train/test/valid split (confirm with group)
        # Load data iterator
        self.iterator = utls.DataIterator()
        # Run through all turns to get a list of conversations
        convs = []
        print(f"Total dataset Diplomacy turns: {len(self.iterator)}")
        tot_turns = 0
        for i, turn in enumerate(self.iterator):
            if "train" in opt["datatype"]:
                if i >= TRAIN_SPLIT:
                    break
            else:
                # test/valid are the same for now
                # TODO: change
                if i < TRAIN_SPLIT:
                    continue

            tot_turns += 1
            for conv in turn:
                if len(conv) >= self.min_turns:
                    convs.append(conv)

        dt = opt["datatype"].split(":")[0]
        print(f"Loaded {tot_turns} Diplomacy turns for datasplit {dt}")

        if self.is_train:
            random.shuffle(convs)

        return convs

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


@register_teacher("dialogue_chunk")
class DialogueChunkTeacher(ChunkTeacher):
    """
    Dialogue teacher but split into chunks for faster loading

    Example usage:
    ```
    parlai display_data -t internal:diplomacy:dialogue_chunk -dt train:stream
    ```
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = utls.add_common_dialogue_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        if shared is None:
            # set map
            self.opt = opt
            self._set_chunk_idx_to_file()
        else:
            self.chunk_idx_to_file = shared["chunk_idx_to_file"]
        self.min_turns = opt["min_turns"]
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return constants.CHUNK_DIALOGUE_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt["datatype"]
        if "train" in datatype:
            if self.min_turns == 1:
                return 5140030, 9836290
            elif self.min_turns == 2:
                return 2244518, 6940778
            elif self.min_turns == 3:
                return 1031766, 4515274
            elif self.min_turns == 4:
                return 520942, 2982802
        if "valid" in datatype:
            if self.min_turns == 1:
                return 433562, 830248
            elif self.min_turns == 2:
                return 189932, 586618
            elif self.min_turns == 3:
                return 88058, 382870
            elif self.min_turns == 4:
                return 44384, 251848

        raise RuntimeError(
            f"Min turns {self.min_turns} for datatype {datatype} currently not supported"
        )

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        file_lst = sorted(glob(os.path.join(folder, "games_*")))
        self.chunk_idx_to_file = {i: x for i, x in enumerate(file_lst)}

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt["datatype"]
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if "train" in datatype:
            return all_chunk_idxs[:-20]
        elif "valid" in datatype:
            return all_chunk_idxs[-20:-10]
        else:
            return all_chunk_idxs[-10:]

    def load_from_chunk(self, chunk_idx: int) -> List[Tuple[str, str]]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])

        with open(chunk_path, "r") as f:
            data = json.load(f)

        iterator = utls.DataIterator(data)

        convs = []
        for _, turn in enumerate(iterator):
            for conv in turn:
                if len(conv) >= self.min_turns:
                    convs.append(conv)

        return convs

    def create_message(self, queue_output: Tuple[str, ...], entry_idx=0) -> "Message":
        """
        Given the tuple output of the queue, return an act.
        """
        conv = queue_output
        msg = conv[entry_idx]
        new_msg = {
            "text": msg["input"],
            "labels": [msg["response"]],
            "episode_done": entry_idx >= (len(conv) - 1),
        }
        for k, v in msg.items():
            if k not in ["input", "response"]:
                new_msg[k] = v

        return new_msg

    def share(self):
        shared = super().share()
        shared["chunk_idx_to_file"] = self.chunk_idx_to_file
        return shared


class DefaultTeacher(DialogueTeacher):
    pass


@register_teacher("base_dialogue_chunk")
class BaseDialogueChunkTeacher(BaseOrderChunkTeacher):
    """
    Streaming data base dialogue teacher for messages/orders.

    Label is next message
    """

    @staticmethod
    def add_silence_messages(power, messages, message_history):
        """
        Static method that adds the SILENCE tokens to messages
        :param power:
        :param messages:
        :param message_history:
        :return:
        """
        empty_output = f"{power}: SILENCE"
        updated_msgs = []
        if not message_history:
            updated_msgs.append(empty_output)
            power_cur_speaker = False
        else:
            power_cur_speaker = message_history[-1].startswith(power)

        for msg in messages:
            if msg.startswith(power):
                power_cur_speaker = True
                updated_msgs.append(msg)
            else:
                if not power_cur_speaker:
                    updated_msgs.append(empty_output)

                updated_msgs.append(msg)
                power_cur_speaker = False

        return updated_msgs

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt["datatype"]
        if "train" in datatype:
            return 10343021, 10343021

        if "valid" in datatype:
            return 113013, 113013

    def _generate_example_tuples(self, game_id, phase_id, player_id, data):
        """
        Yields example tuple(s) used in `_create_message`

        :param game_id: Game id
        :param phase_id: Phase ID
        :param player_id: Player/Power ID
        :param data: Chunk data
        :return:
        """
        data["game_id"] = game_id
        data["phase_id"] = phase_id
        data["player_id"] = player_id
        cur_power = utls.COUNTRY_ID_TO_POWER[int(player_id)].capitalize()
        data["player"] = cur_power

        # We only include Game_Phase_Msg
        if data["metadata"]["data_status"] != "Game_Phase_Msg":
            return

        # format orders
        data["order"] = self.format_order(data["order"])
        data["order_history"] = self.format_order_history(data["order_history"])

        # format state
        data["state"] = self.format_state(data["state"])

        # format messages
        data["message"] = self.format_msg(data)
        data["message_history"] = self.format_msg_history(data)

        # We convert a single data example into several different ones by splitting each "message" into
        # a different example and adapting "message_history" accordingly at each phase
        msgs = data["message"].split("\n")  # check

        msg_history_list = (
            data["message_history"].replace(data["message"], "").split("\n")
        )  # Small hack to fix message history

        msgs = self.add_silence_messages(cur_power, msgs, msg_history_list)

        msg_history_buffer_list = msg_history_list.copy()
        pre_msg_history_buffer_list = msg_history_list.copy()

        # We include self-talks, messages to GameMaster and SILENCE
        default_msg_dict = {
            f"{cur_power} -> {v.capitalize()}": [] for _, v in utls.COUNTRY_ID_TO_POWER.items()
        }
        default_msg_dict[f"{cur_power} -> GameMaster"] = []
        default_msg_dict[f"{cur_power}"] = []

        msg_dict = copy.deepcopy(default_msg_dict)
        yield_example = False

        for msg in msgs:
            if msg.startswith(cur_power):
                if not yield_example:
                    pre_msg_history_buffer_list = msg_history_buffer_list.copy()

                if "SILENCE" not in msg:
                    msg_history_buffer_list.append(msg)

                yield_example = True

                key, msg = msg.split(":", 1)
                msg_dict[key].append(msg)

            else:
                if yield_example:
                    yield self._construct_example_dict(
                        data.copy(), msg_dict, pre_msg_history_buffer_list
                    )
                    yield_example = False
                    msg_dict = copy.deepcopy(default_msg_dict)

                msg_history_buffer_list.append(msg)

        if yield_example:
            yield self._construct_example_dict(data.copy(), msg_dict, pre_msg_history_buffer_list)

    @staticmethod
    def _construct_example_dict(data_dict, msg_dict, pre_msg_history_buffer_list):
        """
        Static method that takes the data dict and updates "messages" and "message_history"
        with the msg_dict
        :param data_dict:
        :param msg_dict:
        :param pre_msg_history_buffer_list:
        :return:
        """
        output_message_list = []
        for k, v in msg_dict.items():
            if v:
                joined_msg = " ".join(v)
                output_message_list.append(f"{k}: {joined_msg}")

        data_dict["message"] = "\n".join(output_message_list)
        data_dict["message_history"] = "\n".join(pre_msg_history_buffer_list)

        return data_dict

    def _get_base_msg(self, queue_output):
        base_msg = {
            "episode_done": True,
            "player_id": queue_output["player_id"],
            "player": queue_output["player"],
            "game_id": queue_output["game_id"],
            "phase_id": queue_output["phase_id"],
            "labels": [queue_output["message"]],
        }

        base_msg.update(queue_output)

        return base_msg


@register_teacher("message_history_state_dialogue_chunk")
class MessageHistoryStateDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message_history']} {queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("state_message_history_dialogue_chunk")
class StateMessageHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [State, Message History] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['state']} {queue_output['message_history']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_order_history_dialogue_chunk")
class MessageHistoryOrderHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY, ORDER HISTORY information
    - [Message History, Order History] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['order_history']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_order_history_state_dialogue_chunk")
class MessageHistoryOrderHistoryStateDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY, ORDER HISTORY, STATE information
    - [Message History, Order History, State] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['order_history']} {queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_dialogue_chunk")
class MessageHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY, ORDER HISTORY, STATE information
    - [Message History, Order History, State] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message_history']} {curr_player}"

        return Message(msg)
