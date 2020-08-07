#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import List, Tuple

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.utils import logging

import parlai_diplomacy.tasks.common_task_utils as utls
from parlai_diplomacy.tasks.no_press.stream.agents import BaseOrderChunkTeacher


@register_teacher("press_chunk")
class PressChunkTeacher(BaseOrderChunkTeacher):
    """
    Dialogue/Order teacher for messages/orders.

    Label is next message or order
    """

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        # TODO: get actual counts here
        datatype = opt["datatype"]
        if "train" in datatype:
            return 25092949, 25092949

        if "valid" in datatype:
            return 1327701, 1327701

    @staticmethod
    def _generate_example_tuples(game_id, phase_id, player_id, data):
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

        # We convert a single data example into several different ones by splitting each "message" into
        # a different example and adapting message_history accordingly at each phase
        msgs = data["message"].split("\n")  # check
        msg_history = (
            data["message_history"].replace(data["message"], "").split("\n")
        )  # Small hack to fix message history
        msg_history_buffer = msg_history.copy()
        for msg in msgs:
            if msg.startswith(data["player"]):
                data_dict = data.copy()
                data_dict["message"] = msg
                data_dict["message_history"] = "\n".join(msg_history_buffer)
                yield data_dict
            else:
                msg_history_buffer.append(msg)

        # Add orders from the model in messages
        data_dict = data.copy()
        data_dict["message"] = data["order"]
        data_dict["message_history"] = "\n".join(msg_history_buffer)
        yield data_dict

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

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['state']} {queue_output['message_history']} {curr_player}"

        return Message(msg)
