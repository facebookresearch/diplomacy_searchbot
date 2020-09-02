#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai_diplomacy.tasks.no_press.single_order.stream.agents import BaseOrderChunkTeacher

"""
File streaming messages and board state data to predict orders.
"""


@register_teacher("state_message_order_chunk")
class StateMessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains STATE then MESSAGE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
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
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message']} {queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("allmessage_state_order_chunk")
class AllmessageStateOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        if queue_output["message"] == "":
            # current phase has no msg
            all_message = queue_output["message_history"]
        else:
            all_message = queue_output["message_history"] + "\n" + queue_output["message"]
        msg["text"] = f"{all_message} {queue_output['state']} {curr_player}"

        return Message(msg)


@register_teacher("allmessage_shortstate_order_chunk")
class AllmessageStateOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        if queue_output["message"] == "":
            # current phase has no msg
            all_message = queue_output["message_history"]
        else:
            all_message = queue_output["message_history"] + "\n" + queue_output["message"]
        msg["text"] = f"{all_message} {queue_output['short_state']} {curr_player}"

        return Message(msg)


@register_teacher("order_history_message_order_chunk")
class OrderHistoryMessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains ORDER HISTORY then MESSAGE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
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
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message']} {queue_output['order_history']} {curr_player}"

        return Message(msg)


@register_teacher("message_order_chunk")
class MessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE information only

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg["text"] = f"{queue_output['message']} {curr_player}"

        return Message(msg)


@register_teacher("allmessage_order_chunk")
class MessageOrderChunkTeacher(BaseOrderChunkTeacher):
    """
    Text field (input) contains MESSAGE information only
    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        all_message = queue_output["message_history"] + "\n" + queue_output["message"]
        msg["text"] = f"{all_message} {curr_player}"

        return Message(msg)
