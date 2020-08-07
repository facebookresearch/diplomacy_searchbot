#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for converting game JSONs into string sequences for model input
"""
import parlai_diplomacy.tasks.common_task_utils as task_utils


class SequenceFormatHelper:
    """
    Helper class to convert a game JSON to a sequence format for a ParlAI model.

    TODO: support all game formats here, once we decide on a definitive message format
    """

    @classmethod
    def change_format(cls, json, fmt="state_order"):
        if fmt == "state_order":
            return cls.state_order_format(json)

        raise RuntimeError(f"Format {fmt} not currently supported")

    @staticmethod
    def state_order_format(json):
        seqs = {}
        for phase in json:
            seqs[phase] = {}
            # TODO: make this work with special tokens
            state = task_utils.flatten_state(json[phase]["state"], None, None)
            for _, speaker in task_utils.COUNTRY_ID_TO_POWER.items():
                # set up speaker
                player_prompt_token = f"{phase} {speaker.capitalize()}:"
                input_seq = f"{state} {player_prompt_token}"
                seqs[phase][speaker] = input_seq

        return seqs


def order_seq_to_fairdip(order_sequence):
    """
    Convert order sequence output by a ParlAI models to a list of orders
    expected by the no-press setting.

    # TODO: check, is this correct?
    """
    order = order_sequence.replace(" [EO_O]", "")
    order_lst = order.split("; ")
    return order_lst
