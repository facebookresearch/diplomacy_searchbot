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
from parlai_diplomacy.tasks.no_press.single_order.stream.agents import BaseOrderChunkTeacher

from abc import ABC
from glob import glob
import os
from typing import List, Tuple

"""
File that takes board state data to predict orders for ALL players. (streaming)
"""


@register_teacher("base_allorder_chunk")
class BaseAllorderChunkTeacher(BaseOrderChunkTeacher):
    """
    Streaming data base teacher for messages/orders.

    Label is all orders given by ALL players
    """

    def _get_base_msg(self, queue_output):
        """
        use all_orders for the labels
        """
        base_msg = {
            "episode_done": True,
            "player_id": queue_output["player_id"],
            "player": queue_output["player"],
            "game_id": queue_output["game_id"],
            "phase_id": queue_output["phase_id"],
            # use all_orders instead of order as in BaseOrderChunkTeacher
            "labels": [queue_output["all_orders"]],
        }

        base_msg.update(queue_output)

        return base_msg


@register_teacher("state_allorder_chunk")
class StateAllorderChunkTeacher(BaseAllorderChunkTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("shortstate_allorder_chunk")
class ShortstateAllorderChunkTeacher(BaseAllorderChunkTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['short_state']} {curr_player}"

        return Message(msg)


@register_teacher("order_history_allorder_chunk")
class OrderHistoryAllorderChunkTeacher(BaseAllorderChunkTeacher):
    """
    Text field (input) contains ORDER HISTORY information only

    Label is all orders given by ALL players
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['order_history']} {curr_player}"

        return Message(msg)


@register_teacher("speaker_token_allorder_chunk")
class SpeakerTokenAllorderChunkTeacher(BaseAllorderChunkTeacher):
    """
    Text field (input) contains player information only

    Label is all orders given by ALL players
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{curr_player}"

        return Message(msg)


@register_teacher("dummy_token_allorder_chunk")
class DummyTokenAllorderChunkTeacher(BaseAllorderChunkTeacher):
    """
    Text field (input) contains only UNK.

    Label is all orders given by ALL players
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        msg["text"] = "UNK"

        return Message(msg)
