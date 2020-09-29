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
import parlai_diplomacy.utils.game_to_sequence_formatting as game_formatting

"""
File for all dialogue teachers that don't load pseudo orders
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
    def add_silence_messages(phase, power, msgs, msg_history):
        """
        Static method that adds the SILENCE tokens to messages
        :param phase:
        :param power:
        :param msgs:
        :param msg_history:
        :return:
        # We should either 
        # 1) focus on the phase-level, i.e. have SILENCE at the start and finish the tail at the end,
        #    even if we've added a SILENCE token at the end of the phase t-1, a new SILENCE is still needed
        #    at phase t at the beginning, search for 'S1903R France' in 
        #    /checkpoint/fairdiplomacy/press_diplomacy/display_data_paste_for_debug/message_history_dialogue_chunk_new_valid_new.log
        #    we have more SILENCE this way 
        # OR
        # 2) focus on the msg-level only, forget about the phase start, only add SILENCE when there are 
        #    consecutive messages whose speakers are not us.
        #    e.g. check the last msg or msg_history, and add SILENCE 
        #    ONLY when msg_history[-1][-1]['speaker']!=cur_power, and msgs[0]['speaker']!=cur_power, 
        #    we have fewer SILENCE this way
        """
        silence_message = {
            "speaker": power,
            "listener": None,
            "message": "SILENCE",
            "time_sent": None,
            "phase": phase,
        }
        updated_msgs = []

        assert isinstance(msg_history, list)

        # we add SILENCE at the beginnig of a phase
        power_cur_speaker = False
        for msg in msgs:
            if msg["speaker"] == power:
                power_cur_speaker = True
                updated_msgs.append(msg)
            else:
                if not power_cur_speaker:
                    updated_msgs.append(silence_message)

                updated_msgs.append(msg)
                power_cur_speaker = False

        # we add SILENCE if the last speaker is not us
        if not power_cur_speaker:
            updated_msgs.append(silence_message)

        return updated_msgs

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt["datatype"]
        n_chunks = opt.get("n_chunks")
        if n_chunks < 0:
            if "train" in datatype:
                return 14834555, 14834555

            if "valid" in datatype:
                return 159498, 159498
        else:
            return self._get_num_samples_for_n_chunks(opt)

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
        data["data_status"] = data["metadata"]["data_status"]
        data_status = data["data_status"]

        cur_power = utls.COUNTRY_ID_TO_POWER[int(player_id)].capitalize()
        cur_phase = phase_id

        data["player"] = cur_power

        # We only include Game_Phase_Msg and Game_Phase_NoMsg
        if data_status not in [
            "Game_Phase_Msg",
            "Game_Phase_NoMsg",
        ]:
            return

        # format orders
        data["order"] = self.format_order(data["order"])
        data["order_history"] = self.format_order_history(data["order_history"])
        data["all_orders"] = self.format_all_orders(data["all_orders"], player_id)

        # format state
        data["short_state"] = self.format_short_state(data["state"])
        data["short_state_history"] = self.format_short_state_history(data["state_history"])
        data["state"] = self.format_state(data["state"])
        data["state_history"] = self.format_state_history(data["state_history"])

        # save original formatted message for debugging purposes
        data["formatted_original_message"] = self.format_msg(data["message"])
        data["formatted_original_message_history"] = self.format_msg_history(
            data["message_history"]
        )
        del data["message_processed"]
        del data["message_history_processed"]

        # We convert a single data example into several different ones by splitting each "message" into
        # a different example and adapting "message_history" accordingly at each phase
        msgs = data["message"]
        msg_history = data["message_history"]

        msgs = self.add_silence_messages(cur_phase, cur_power, msgs, msg_history)

        msg_history.append([])
        msg_history_buffer_list = copy.deepcopy(msg_history)
        pre_msg_history_buffer_list = copy.deepcopy(msg_history)

        # We include self-talks, and SILENCE
        # key for the dict is a tuple (cur_power, to_power)
        default_msg_dict = {
            (cur_power, v.capitalize()): [] for _, v in utls.COUNTRY_ID_TO_POWER.items()
        }
        default_msg_dict[(cur_power, None)] = []
        default_msg_dict[(cur_power, "GameMaster")] = []

        msg_dict = copy.deepcopy(default_msg_dict)
        yield_example = False
        n_examples = 0

        for msg in msgs:
            if msg["speaker"] == cur_power:
                if not yield_example:
                    pre_msg_history_buffer_list = copy.deepcopy(msg_history_buffer_list)

                if msg["message"] != "SILENCE":
                    msg_history_buffer_list[-1].append(msg)

                yield_example = True

                key = (msg["speaker"], msg["listener"])
                msg_dict[key].append(msg)

            else:
                if yield_example:
                    n_examples += 1
                    yield self._construct_example_dict(
                        data.copy(), msg_dict, pre_msg_history_buffer_list, n_examples
                    )
                    yield_example = False
                    msg_dict = copy.deepcopy(default_msg_dict)

                msg_history_buffer_list[-1].append(msg)

        if yield_example:
            n_examples += 1
            yield self._construct_example_dict(
                data.copy(), msg_dict, pre_msg_history_buffer_list, n_examples
            )

    def _construct_example_dict(
        self, data_dict, msg_dict, pre_msg_history_buffer_list, n_examples
    ):
        """
        Static method that takes the data dict and updates "messages" and "message_history"
        with the msg_dict
        :param data_dict:
        :param msg_dict:
        :param pre_msg_history_buffer_list:
        :param n_examples: n_examples to build the special id used in pseudo-order joining, the id looks like 44-S1901M-5-4 (game_id-phase_id-player_id-n_examples)
        :return:
        """
        cur_phase = data_dict["phase_id"]
        output_message_list = []
        for k, v in msg_dict.items():
            if v:
                output_message_list.extend(v)

        data_dict["message"] = self.format_msg(output_message_list)
        data_dict["message_history"] = self.format_msg_history(pre_msg_history_buffer_list)

        # add a special id to join pseudo-order later,
        # if doesn't hurt to leave it here when not loading pseudo orders, and could also helps us debug
        data_dict[
            "example_id"
        ] = f"{data_dict['game_id']}-{data_dict['phase_id']}-{data_dict['player_id']}-{n_examples}"

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


@register_teacher("message_history_allorder_dialogue_chunk")
class MessageHistoryAllorderHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY, All future order information
    - [Message History, All future orders at the end of the turn] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['all_orders']} {curr_player}"

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
    Text field (input) contains MESSAGE HISTORY information
    - [Message History] -> [Message]
    Label is the next dialogue
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message_history']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_shortstate_dialogue_chunk")
class MessageHistoryShortstateDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE then SHORT_STATE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['short_state']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_shortstate_allorder_dialogue_chunk")
class MessageHistoryShortstateAllorderDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE then ALL_ORDER information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['short_state']} {queue_output['all_orders']} {curr_player}"

        return Message(msg)


@register_teacher("allorder_message_history_dialogue_chunk")
class AllorderMessageHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains ALL_ORDER, MESSAGE_HISTORY information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['all_orders']} {queue_output['message_history']} {curr_player}"

        return Message(msg)
