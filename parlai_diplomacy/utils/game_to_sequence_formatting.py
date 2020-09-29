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
        if fmt == "shortstate_order":
            return cls.shortstate_order_format(json)

        raise RuntimeError(f"Format {fmt} not currently supported")

    @staticmethod
    def state_order_format(json):
        seqs = {}
        for phase in json:
            seqs[phase] = {}
            # TODO: make this work with special tokens
            state = flatten_state(json[phase]["state"], None, None)
            for _, speaker in task_utils.COUNTRY_ID_TO_POWER.items():
                # set up speaker
                player_prompt_token = f"{phase} {speaker.capitalize()}:"
                input_seq = f"{state} {player_prompt_token}"
                seqs[phase][speaker] = input_seq

        return seqs

    @staticmethod
    def shortstate_order_format(json):
        seqs = {}
        for phase in json:
            seqs[phase] = {}
            # TODO: make this work with special tokens
            state = flatten_state(json[phase]["state"], None, None, short_version=True)
            for _, speaker in task_utils.COUNTRY_ID_TO_POWER.items():
                # set up speaker
                player_prompt_token = f"{phase} {speaker.capitalize()}:"
                input_seq = f"{state} {player_prompt_token}"
                seqs[phase][speaker] = input_seq

        return seqs


def order_seq_to_fairdip(order_sequence, special_tokens_map=None, with_special_token=False):
    """
    Convert order sequence output by a ParlAI models to a list of orders
    expected by the no-press setting.

    # TODO: check, is this correct?
    """
    # remove [EO_O]
    if with_special_token:
        order = order_sequence.replace(special_tokens_map["[EO_O]"], "").strip()
    else:
        order = order_sequence.replace("[EO_O]", "").strip()

    if not order:
        # empty order
        # TODO: check in no-press, empty_order is [] or [''] ?
        return []

    order_lst = order.split("; ")
    return order_lst


def order_is_empty(order_sequence, special_tokens_map=None, with_special_token=False):
    """
    Check if an order sequence is empty
    """
    if with_special_token:
        order = order_sequence.replace(special_tokens_map["[EO_O]"], "").strip()
    else:
        order = order_sequence.replace("[EO_O]", "").strip()
    if not order:
        return True

    return False


def all_orders_seq_to_dct(
    all_orders_sequence,
    special_tokens_map=None,
    with_special_token=False,
    power_name_as_key_and_return_as_set=True,
):
    """
    Convert a sequence of a prediction for all orders to a dict containing
    the orders for all dict.
    
    Example all orders: England: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]\nFrance: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]
    :params power_name_as_key_and_return_as_set: if True (should be used for metric evaluation), the returned order_dct will use power_name as the key, and the value of the order_dct will be a set
                                                 if False (should be used in sort_orders), the returned order_dct will use power_id as the key, and the value will be a list
    """
    order_dct = {}
    if not all_orders_sequence:
        return order_dct

    order_split = all_orders_sequence.split("\n")
    for order in order_split:
        power, order_seq = order.split(": ")
        if power_name_as_key_and_return_as_set:
            order_dct[power] = set(
                order_seq_to_fairdip(order_seq, special_tokens_map, with_special_token)
            )
        else:
            order_dct[str(task_utils.POWER_TO_COUNTRY_ID[power.upper()])] = order_seq_to_fairdip(
                order_seq, special_tokens_map, with_special_token
            )

    return order_dct


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


def sort_orders(
    order_str, cur_player_id, special_tokens_map=None, with_special_token=False, all_order=True
):
    """
    given an order string, return the sorted version
    sorted by countries and each country's order is sorted alphabatically
    """
    if not order_str:
        return ""

    if all_order:
        # unflatten the order_str
        all_order_dct = all_orders_seq_to_dct(
            order_str,
            special_tokens_map,
            with_special_token,
            power_name_as_key_and_return_as_set=False,
        )
        # re-sort the order_dct
        sorted_order_str = flatten_all_orders(
            all_order_dct, cur_player_id, special_tokens_map, with_special_token
        )
    else:
        # if the order_str contains order for a single player
        # unflatten the order_str
        order_lst = order_seq_to_fairdip(order_str, special_tokens_map, with_special_token)
        # resort the order_lst
        sorted_order_str = flatten_orders(order_lst, special_tokens_map, with_special_token)

    return sorted_order_str


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


def flatten_all_orders(
    all_order_dct, cur_player_id, special_tokens_map=None, with_special_token=False
):
    """
    flatten orders for all players, and put the current player's order at the end
    return:
        England: [order] [EO_O]
        France: [order] [EO_O]
        ...
        [cur_player]: [order] [EO_O]
    """

    def format_one_order(one_order, player_name):
        """
        format the order for one player
        return:
            England: [flat_order]
        """
        one_flat_order = flatten_orders(one_order, special_tokens_map, with_special_token)
        one_flat_order = f"{player_name.capitalize()}: {one_flat_order}"

        return one_flat_order

    all_order_list = []

    for player_id, player_name in task_utils.COUNTRY_ID_TO_POWER.items():
        if player_id == int(cur_player_id):
            continue
        one_flat_order = format_one_order(all_order_dct[str(player_id)], player_name)
        all_order_list.append(one_flat_order)

    # append the current player's order at the end
    cur_player_order = format_one_order(
        one_order=all_order_dct[str(cur_player_id)],
        player_name=task_utils.COUNTRY_ID_TO_POWER[int(cur_player_id)],
    )
    all_order_list.append(cur_player_order)

    return "\n".join(all_order_list)


def flatten_one_phase_order(
    order_dct, phase_name, special_tokens_map=None, with_special_token=False
):
    flat_orders = []
    for speaker, order in order_dct.items():
        flat_order = flatten_orders(order)
        flat_order = f"{speaker.capitalize()}: {flat_order}"
        flat_orders.append(flat_order)

    # add phase info
    flat_orders = phase_name + "\n" + "\n".join(flat_orders)

    return flat_orders


def flatten_order_history(order_history_dct, special_tokens_map=None, with_special_token=False):
    """
    return:
        S1901M
        France: [order] [EO_O]
        Italy: [order] [EO_O]
        ...
        [phase_name]
        [player1]: [order] [EO_O]
    """
    phase_orders = []
    for phase_name, order_dct in order_history_dct.items():
        phase_order = flatten_one_phase_order(
            order_dct, phase_name, special_tokens_map, with_special_token
        )
        phase_orders.append(phase_order)
    return "\n".join(phase_orders)


def flatten_state(state, special_tokens_map=None, with_special_token=False, short_version=False):
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

    keys = ["units"]
    if not short_version:
        keys.extend(
            ["retreats", "centers", "homes", "influence", "civil_disorder", "builds",]
        )

    state_list = [flatten_country_status(key, state[key]) for key in keys if key in state]
    final_state_str = "\n".join(state_list)
    # add end_or_state_token: [EO_STATE]
    final_state_str = add_end_token(
        final_state_str, "[EO_STATE]", with_special_token, special_tokens_map
    )

    return final_state_str


def flatten_state_history(
    state_history_dct, special_tokens_map=None, with_special_token=False, short_version=False
):
    """
    return:
        S1901M
        [state1] [EO_STATE]
        F1901M
        [state2] [EO_STATE]
        ...
        [phase_name]
        [state] [EO_STATE]
    """
    state_list = []
    for phase in state_history_dct:
        one_state = flatten_state(
            state_history_dct[phase], special_tokens_map, with_special_token, short_version
        )
        one_state = phase + "\n" + one_state
        state_list.append(one_state)

    return "\n".join(state_list)


def concate_msghistory_curmsg(queue_output):
    """
    concate message_history and current_msg to form future msg for the pseudo-order generation
    msg[:t] + msg[t] --> order[t-1]
    :param queue_output: the queue_output with all the data used in the streaming teacher
    """
    if queue_output["phase_id"] in queue_output["message_history"]:
        future_msg = (
            queue_output["message_history"]
            + "\n"
            + "\n".join(queue_output["message"].split("\n")[1:])
        )
    else:
        future_msg = queue_output["message_history"] + "\n" + queue_output["message"]
    future_msg = future_msg.strip("\n")

    return future_msg


def format_one_message(speaker, listener, msg, special_tokens_map=None, with_special_token=False):
    if listener:
        formatted_msg = f"{speaker} -> {listener}: {msg}"
    else:
        formatted_msg = f"{speaker}: {msg}"

    formatted_msg = add_end_token(formatted_msg, "[EO_M]", with_special_token, special_tokens_map)

    return formatted_msg


def flatten_one_message(one_msg_dct, special_tokens_map=None, with_special_token=False):
    speaker = one_msg_dct["speaker"]
    listener = one_msg_dct["listener"]
    # check for GameMaster
    assert speaker != "GameMaster" and listener != "GameMaster"

    msg = one_msg_dct["message"]
    flat_msg = format_one_message(speaker, listener, msg, special_tokens_map, with_special_token)

    return flat_msg


def flatten_one_phase_message(msg_lst, special_tokens_map=None, with_special_token=False):
    """
    flatten messages in one phase, called in self.format_msg in the teacher
    for the current phase msg 
    
    Args:
        msg_lst: list[dict]
    
    Returns:
        if len(msg_lst) == 0: 
            return ""
        else:
            return:
                S1901M  
                England -> Turkey: msg [EO_M] 
                Turkey -> England: msg [EO_M] 
    """
    if len(msg_lst) == 0:
        # empty phase
        return ""

    flat_msgs = []
    phase_name = msg_lst[0]["phase"]
    for msg_dct in msg_lst:
        # just checking if they are from the same phase
        assert phase_name == msg_dct["phase"]
        flat_msg = flatten_one_message(msg_dct, special_tokens_map, with_special_token)
        flat_msgs.append(flat_msg)

    # add phase info
    flat_msgs = phase_name + "\n" + "\n".join(flat_msgs)
    return flat_msgs


def flatten_message_history(msg_his_lst, special_tokens_map=None, with_special_token=False):
    """flatten the message history, 
    message history = all msgs that happen before the prediction
    e.g. 
    1) if predicting an order, message history = all msgs that happen before the end of the phase
    2) if predicting a message, message history = all msgs that happen before the time to generate a new message

    Args:
        msg_his_lst (list[list[dict]]): a list of phase messages, each phase message is a list of messages
        special_tokens_map (dict, optional): [description]. Defaults to None.
        with_special_token (bool, optional): [description]. Defaults to False.

    Returns:
        if len(msg_his_lst) == 0: 
            return ""
        else:
            return:
                S1901M  
                England -> Turkey: msg [EO_M] 
                Turkey -> England: msg [EO_M] 
                F1901M  
                England -> Turkey: msg [EO_M] 
                Turkey -> England: msg [EO_M] 
    """
    phase_msgs = []
    for phase_msg_lst in msg_his_lst:
        phase_msg = flatten_one_phase_message(
            phase_msg_lst, special_tokens_map, with_special_token
        )
        phase_msgs.append(phase_msg)

    # remove empty phase
    phase_msgs = [msg for msg in phase_msgs if msg != ""]
    return "\n".join(phase_msgs)
