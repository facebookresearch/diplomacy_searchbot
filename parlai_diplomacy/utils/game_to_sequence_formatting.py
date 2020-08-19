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


def map_special_tokens(token, special_tokens_map):
    """
    Convert token to a special token if it exists
    """
    if token in special_tokens_map:
        return special_tokens_map[token]
    else:
        return token


def add_end_token(text, end_token, with_special_token=False, special_tokens_map=None):
    """
    Add end and possibly special tokens
    """
    if with_special_token:
        converted_text = f"{text} {map_special_tokens(end_token, special_tokens_map)}"
    else:
        converted_text = f"{text} {end_token}"
    return converted_text


def flatten_orders(order_list, special_tokens_map=None, with_special_token=False):
    """
    Flatten the order in game*.json
    """

    if type(order_list) is not list:
        order_list = [str(order_list)]
    # sort the orders
    order_list = sorted(order_list)
    # convert to special tokens
    if with_special_token:
        order_list = [
            " ".join([map_special_tokens(token, special_tokens_map) for token in order.split()])
            for order in order_list
        ]
    # join the orders
    flat_order_str = "; ".join(order_list)
    # add end_of_order token: [EO_O]
    flat_order_str = add_end_token(
        flat_order_str, "[EO_O]", with_special_token, special_tokens_map
    )

    return flat_order_str


def flatten_state(state, special_tokens_map=None, with_special_token=False):
    """
    Flatten the state in game*.json
    """

    def flatten_country_status(key, country_status):
        status_list = []
        for country, status in country_status.items():
            if type(status) is not list:
                status = [str(status)]
            status = sorted(status)
            if with_special_token:
                status = [
                    " ".join(
                        [map_special_tokens(token, special_tokens_map) for token in sta.split()]
                    )
                    for sta in status
                ]
            status_list.append(f'{country.capitalize()}: {", ".join(status)}')
        final_status = f'{key}: {"; ".join(status_list)}'

        return final_status

    keys = [
        "units",
        "retreats",
        "centers",
        "homes",
        "influence",
        "civil_disorder",
        "builds",
    ]
    state_list = [flatten_country_status(key, state[key]) for key in keys if key in state]
    final_state_str = "\n".join(state_list)
    # add end_or_state_token: [EO_STATE]
    final_state_str = add_end_token(
        final_state_str, "[EO_STATE]", with_special_token, special_tokens_map
    )

    return final_state_str
