#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import ChunkTeacher
from parlai.utils import logging

import parlai_diplomacy.tasks.common_task_utils as utls

from abc import ABC
from glob import glob
import json
import os
from typing import List, Tuple

"""
File streaming messages and board state data to predict orders.
"""
TRAIN_VAL_SPLIT = 475  # 95% of 500 to mimic what Weiyan is doing


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
        return utls.CHUNK_ORDER_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        # TODO: get actual counts here
        datatype = opt["datatype"]
        if "train" in datatype:
            return 5140030, 9836290

        if "valid" in datatype:
            return 433562, 830248

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

        with open(chunk_path, "r") as f:
            data = json.load(f)

        lst = []
        for game_id, game in data.items():
            for phase_id, phase in game.items():
                for player_id, data in phase.items():
                    data["game_id"] = game_id
                    data["phase_id"] = phase_id
                    data["player_id"] = player_id
                    lst.append(data)

        logging.info(f"Loaded {len(lst)} examples from chunk {chunk_idx}.")

        return lst

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
        return {
            "episode_done": True,
            "player_id": queue_output["player_id"],
            "game_id": queue_output["game_id"],
            "phase_id": queue_output["phase_id"],
            "labels": [queue_output["order"]],
        }


@register_teacher("state_message_order_chunk")
class StateMessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains STATE then MESSAGE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['state']} {queue_output['message']} {curr_player}"

        return Message(msg)


@register_teacher("message_state_order_chunk")
class MessageStateOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['message']} {queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("order_history_message_order_chunk")
class OrderHistoryMessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains ORDER HISTORY then MESSAGE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['order_history']} {queue_output['message']} {curr_player}"

        return Message(msg)


@register_teacher("message_order_history_order_chunk")
class MessageOrderHistoryOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE then ORDER HISTORY information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['message']} {queue_output['order_history']} {curr_player}"

        return Message(msg)


@register_teacher("state_order_chunk")
class StateOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("message_order_chunk")
class MessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
        msg["text"] = f"{queue_output['message']} {curr_player}"

        return Message(msg)


@register_teacher("order_history_order_chunk")
class OrderHistoryOrdeChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains ORDER HISTORY information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
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
        curr_player = utls.COUNTRY_ID_TO_POWER[int(queue_output["player_id"])].capitalize()
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
