#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import ChunkTeacher
from parlai.utils import logging

import parlai_diplomacy.tasks.common_task_utils as utls
import parlai_diplomacy.utils.datapath_constants as constants
import parlai_diplomacy.utils.game_to_sequence_formatting as game_formatting
import parlai_diplomacy.utils.game_loading as game_loading

from abc import ABC
from glob import glob
import os
from typing import List, Tuple

"""
File that takes board state data to predict orders. (streaming)
"""
TRAIN_VAL_SPLIT = 990  # 99% of 1000 to mimic fairdip NOTE: this changed recently!


@register_teacher("base_order_chunk")
class BaseOrderChunkTeacher(ChunkTeacher, ABC):
    """
    Streaming data base teacher for messages/orders.

    Label is the order given by the player
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = utls.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        if shared is None:
            # set map
            self.opt = opt
            self._set_chunk_idx_to_file()
        else:
            self.chunk_idx_to_file = shared["chunk_idx_to_file"]
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return constants.CHUNK_MESSAGE_ORDER_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt["datatype"]
        # TODO: Emily update me
        if "train" in datatype:
            return 14211400, 14211400

        if "valid" in datatype:
            return 141624, 141624

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        file_lst = sorted(glob(folder))
        self.chunk_idx_to_file = {i: x for i, x in enumerate(file_lst)}

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt["datatype"]
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if "test" in datatype:
            logging.warn("Test set does not exist, switching to valid")
            datatype = datatype.replace("test", "valid")

        if "train" in datatype:
            return all_chunk_idxs[:TRAIN_VAL_SPLIT]
        elif "valid" in datatype:
            return all_chunk_idxs[TRAIN_VAL_SPLIT:]

    def load_from_chunk(self, chunk_idx: int) -> List[Tuple[str, str]]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])

        game_data = game_loading.load_from_gz(chunk_path)

        lst = []
        for game_id, game in game_data.items():
            for phase_id, phase in game.items():
                for player_id, data in phase.items():
                    lst.extend(self._generate_example_tuples(game_id, phase_id, player_id, data))

        logging.info(f"Loaded {len(lst)} examples from chunk {chunk_idx}.")

        return lst

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
        data["player"] = utls.COUNTRY_ID_TO_POWER[int(player_id)].capitalize()

        # format orders
        data["order"] = self.format_order(data["order"])
        data["order_history"] = self.format_order_history(data["order_history"])

        # format state
        data["state"] = self.format_state(data["state"])

        # format messages
        data["message"] = self.format_msg(data)
        data["message_history"] = self.format_msg_history(data)

        yield data

    def format_order(self, order_lst):
        """
        Format order

        Left easily overridable to change formatting
        """
        return game_formatting.flatten_orders(order_lst)

    def format_order_history(self, order_history_dct):
        """
        Format order

        Left easily overridable to change formatting
        """
        flat_orders = []
        for phase, data in order_history_dct.items():
            for speaker, order in data.items():
                flat_order = game_formatting.flatten_orders(order)
                flat_order = f"{phase} {speaker.capitalize()}: {flat_order}"
                flat_orders.append(flat_order)

        return "\n".join(flat_orders)

    def format_state(self, state_dct):
        """
        Format state

        Left easily overridable to change formatting
        """
        return game_formatting.flatten_state(state_dct)

    def format_msg(self, data):
        """
        Format messages

        Left easily overridable to change formatting
        """
        msg = data["message_processed"]
        del data["message_processed"]
        return msg

    def format_msg_history(self, data):
        """
        Format message history

        Left easily overridable to change formatting
        """
        msg_hist = data["message_history_processed"]
        del data["message_history_processed"]
        return msg_hist

    def create_message(self, queue_output, entry_idx=0) -> "Message":
        """
        Given the tuple output of the queue, return an act.
        """
        raise RuntimeError("Must implement this for your base class")

    def share(self):
        shared = super().share()
        shared["chunk_idx_to_file"] = self.chunk_idx_to_file
        return shared

    def _get_base_msg(self, queue_output):
        base_msg = {
            "episode_done": True,
            "player_id": queue_output["player_id"],
            "player": queue_output["player"],
            "game_id": queue_output["game_id"],
            "phase_id": queue_output["phase_id"],
            "labels": [queue_output["order"]],
        }

        base_msg.update(queue_output)

        return base_msg

    def _get_player_prompt_token(self, queue_output):
        player_prompt_token = f"{queue_output['phase_id']} {queue_output['player']}:"
        return player_prompt_token


@register_teacher("state_order_chunk")
class StateOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("order_history_order_chunk")
class OrderHistoryOrdeChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains ORDER HISTORY information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['order_history']} {curr_player}"

        return Message(msg)


@register_teacher("speaker_token_order_chunk")
class SpeakerTokenOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains player information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{curr_player}"

        return Message(msg)


@register_teacher("dummy_token_order_chunk")
class DummyTokenOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains only UNK.

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        msg["text"] = "UNK"

        return Message(msg)
