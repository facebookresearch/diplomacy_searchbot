#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai_diplomacy.tasks.no_press.regular.agents import BaseOrderTeacher

"""
File for message and board state data, NOT streaming.

For streaming data, please see dialogue_and_state_stream
"""


@register_teacher("state_message_order")
class StateMessageOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "state_message_order"
        new_msg["text"] = f"{new_msg['state']} {new_msg['message']} {speaker_token}"

        return new_msg


@register_teacher("message_state_order")
class MessageStateOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "message_state_order"
        new_msg["text"] = f"{new_msg['message']} {new_msg['state']} {speaker_token}"

        return new_msg


@register_teacher("message_history_state_order")
class MessageStateOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "message_history_state_order"
        new_msg["text"] = f"{new_msg['message_history']} {new_msg['state']} {speaker_token}"

        return new_msg


@register_teacher("order_history_message_order")
class OrderHistoryMessageOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "order_history_message_order"
        new_msg["text"] = f"{new_msg['order_history']} {new_msg['message']} {speaker_token}"

        return new_msg


@register_teacher("message_order_history_order")
class MessageOrderHistoryOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "message_order_history_order"
        new_msg["text"] = f"{new_msg['message']} {new_msg['order_history']} {speaker_token}"

        return new_msg


@register_teacher("message_order")
class MessageOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "message_order"
        new_msg["text"] = f"{new_msg['message']} {speaker_token}"

        return new_msg
