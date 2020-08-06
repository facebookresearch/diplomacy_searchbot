#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
from glob import glob
import re
import os
import sys
import time
import numpy as np
from copy import deepcopy

import parlai.utils.logging as logging

###########################################
# CONSTANTS
###########################################

# TODO: better ways to organize the paths
CHUNK_DIALOGUE_PATH = (
    "/checkpoint/fairdiplomacy/press_diplomacy/chat_messages/chat_messages_jsons/"
)
CHUNK_ORDER_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons/dumps_State_OrderHistory_MessageHistory-all-msg-SpecialToken-False_order/*.json"

DATAPATH = "/checkpoint/fairdiplomacy/press_diplomacy/processed_chat_jsons/game_phases/redacted_messages_runthree_*.json"
ORDER_PATH = (
    "/checkpoint/fairdiplomacy/processed_orders_jsons/game_*.json*"  # some are json.partial
)
ORDER_JSON_PATH = "/checkpoint/fairdiplomacy/processed_orders_jsons/combined_json/game.json"
JOINED_JSON_PATH_PREFIX = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons"
VALID_JSON_PATH = (
    "/checkpoint/fairdiplomacy/press_diplomacy/validation/validation_report/valid_data.json"
)
VALID_SPLIT_JSON_PATH = (
    "/checkpoint/fairdiplomacy/press_diplomacy/validation/validation_report/split_jsons"
)

# joined_json will be splitted into
TOTAL_GAMES = 54622
DATA_SPLIT_INTO = 500

# valid_json will be spliited into
TEST_SPLIT_INTO = 64
TEST_RESULT_JSONL_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/validation_report/report4Weiyan_output_split_*_internal:diplomacy:message_order_replies.jsonl"
TEST_RESULT_SAVE_JSON_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/validation_report/valid_prediction_json/parlai_valid_set_prediction.json"
TEST_RESULT_SAVE_JSON_NO_LAST_PHASE_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/validation_report/valid_prediction_json/parlai_valid_set_prediction_no_last_phase.json"
SPECIAL_TOKEN_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/model_utils/special_tokens.txt"
REDACTED_MSG_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/redacted_analysis/redacted_msg.txt"

# for token_stat calculation
BART_OPT_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/model_utils/Bart.opt"
TRANSFORMER_OPT_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/model_utils/Transformer.opt"


# from https://github.com/fairinternal/fairdiplomacy/blob/monster2/fairdiplomacy/data/build_dataset.py
COUNTRY_ID_TO_POWER = {
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}


###########################################
# UTILITY FUNCTIONS
###########################################


def is_training(datatype):
    return "train" in datatype and "evalmode" not in datatype


def analyze_redacted():
    data = load_data()
    msgs = [d["message"] for d in data]
    extracted = []
    all_tokens = []
    msg_lens = []
    num_no_redacted_msg = 0
    num_all_tokens = 0
    num_redacted_tokens = 0
    redacted_msg = []
    for msg in tqdm(msgs):
        cur_extracted = re.findall(r"\[\d+\]", msg)
        extracted.append(cur_extracted)
        all_tokens.extend(msg.split())
        if len(cur_extracted) == 0:
            num_no_redacted_msg += 1
        else:
            redacted_msg.append(msg)
        num_redacted_tokens += len(cur_extracted)
        num_all_tokens += len(msg.split())
        msg_lens.append(len(msg.split()))

    redacted_tokens = []
    for e in extracted:
        redacted_tokens.extend(e)
    num_unique_redacted_tokens = len(set(redacted_tokens))
    num_unique_tokens = len(set(all_tokens))

    print(
        f"{round(len(redacted_msg)/len(msgs), 4)*100}% ({len(redacted_msg)}/{len(msgs)}) messeges are redacted\n"
    )
    redacted_pct_each_msg = [
        len(cur_extracted) / msg_len if msg_len > 0 else -1
        for cur_extracted, msg_len in zip(extracted, msg_lens)
    ]
    redacted_pct_each_msg = np.array(redacted_pct_each_msg)

    avg_redacted_pct = np.mean(redacted_pct_each_msg[redacted_pct_each_msg >= 0])
    print(f"On average, {round(avg_redacted_pct, 4)*100}% tokens in each message is redacted")
    print(
        f"Percentile for the redacted token percentage in each message:\n"
        f"25%: {np.percentile(redacted_pct_each_msg, 25)}\t"
        f"50%: {np.percentile(redacted_pct_each_msg, 50)}\t"
        f"75%: {np.percentile(redacted_pct_each_msg, 75)}\t"
        f"90%: {round(np.percentile(redacted_pct_each_msg, 90)*100, 2)}%\t"
        f"95%: {round(np.percentile(redacted_pct_each_msg, 95)*100, 2)}%\t"
        f"99%: {round(np.percentile(redacted_pct_each_msg, 99)*100, 2)}%\n"
    )

    print(
        f"unique_redacted_tokens/unique_tokens: "
        f"{round(num_unique_redacted_tokens/num_unique_tokens*100, 2)}% "
        f"({num_unique_redacted_tokens}/{num_unique_tokens})\n"
    )
    print(
        f"all_redacted_tokens/all_tokens: "
        f"{round(sum(map(len, extracted))/sum(msg_lens)*100, 2)}% ({sum(map(len, extracted))}/{sum(msg_lens)})\n"
    )
    print(f" [Saving redacted message to {REDACTED_MSG_PATH}...]")
    # if not os.path.exists(REDACTED_MSG_PATH):
    with open(REDACTED_MSG_PATH, "w") as fh:
        fh.writelines([m + "\n" for m in redacted_msg])


def load_special_tokens():
    with open(SPECIAL_TOKEN_PATH, "r") as fh:
        special_tokens = [token.rstrip("\n") for token in fh.readlines()]
        special_tokens_map = {token.lstrip("_").rstrip("_\n"): token for token in special_tokens}
    return special_tokens, special_tokens_map


def map_special_tokens(token, special_tokens_map):
    if token in special_tokens_map:
        return special_tokens_map[token]
    else:
        return token


def add_end_token(text, end_token, with_special_token, special_tokens_map):
    if with_special_token:
        converted_text = f"{text} {map_special_tokens(end_token, special_tokens_map)}"
    else:
        converted_text = f"{text} {end_token}"
    return converted_text


def replace_with_special_token(text):
    """
    replace order and state_str with special tokens
    input: state_str or order without __A__
    output: state_str or order with __
    """

    special_tokens, special_tokens_map = load_special_tokens()

    text_newline = text.split("\n")
    new_text_newline = []
    for t1 in text_newline:
        t_semi = t1.split("; ")
        new_text_semi = []
        for t2 in t_semi:
            t_comma = t2.split(", ")
            new_text_comma = []
            for t3 in t_comma:
                t_space = t3.split(" ")
                new_text_space = []
                for t4 in t_space:
                    if t4 in special_tokens_map:
                        new_text_space.append(special_tokens_map[t4])
                    else:
                        new_text_space.append(t4)
                new_text_space = " ".join(new_text_space)
                new_text_comma.append(new_text_space)
            new_text_comma = ", ".join(new_text_comma)
            new_text_semi.append(new_text_comma)
        new_text_semi = "; ".join(new_text_semi)
        new_text_newline.append(new_text_semi)
    new_text_newline = "\n".join(new_text_newline)
    return new_text_newline


def flatten_orders(order_list, special_tokens_map, with_special_token=True):
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


def flatten_state(state, special_tokens_map, with_special_token=True):
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
    state_list = [flatten_country_status(key, state[key]) for key in keys]
    final_state_str = "\n".join(state_list)
    # add end_or_state_token: [EO_STATE]
    final_state_str = add_end_token(
        final_state_str, "[EO_STATE]", with_special_token, special_tokens_map
    )

    return final_state_str


def split_test_files():
    final_data = {}
    with open(VALID_JSON_DIR, "r") as fh:
        raw_data = json.load(fh)

    for data in raw_data:
        game_id, phase_id, speaker_id, order = (
            data["game"],
            data["game_phase"],
            int(data["speaker_id"]),
            data["order"],
        )
        if game_id in final_data:
            if phase_id in final_data[game_id]:
                if speaker_id in final_data[game_id][phase_id]:
                    assert ValueError("repeated speaker_id")
                else:
                    final_data[game_id][phase_id][speaker_id] = order
            else:
                final_data[game_id][phase_id] = {speaker_id: order}
        else:
            final_data[game_id] = {phase_id: {speaker_id: order}}

    game_ids = list(final_data.keys())
    game_id_splits = np.array_split(np.array(game_ids), TEST_SPLIT_INTO)

    for i, split in enumerate(tqdm(game_id_splits)):
        data_this_split = []
        for data in raw_data:
            game_id = data["game"]
            if game_id in split:
                data_this_split.append(data)
        with open(f"{VALID_SPLIT_JSON_PATH}/valid_split_{i+1}.json", "w") as fh:
            json.dump(data_this_split, fh)


def phase_abbrev_to_phase(phase):
    """
    https://github.com/diplomacy/diplomacy/blob/master/diplomacy/integration/webdiplomacy_net/game.py#L20-L27
    if phase == 'Builds':
        season = 'W'
    {'Diplomacy': 'M', 'Retreats': 'R', 'Builds': 'A'}
    """
    season_map = {"S": "Spring", "F": "Fall", "W": "W"}
    phase_map = {"M": "Diplomacy", "R": "Retreats", "A": "Builds"}
    # season = season_map[phase[0]]
    # year = phase[1:5]
    season_year = phase[:5]
    phase_type = phase_map[phase[5:]]
    phase_converted = f"{season_year} {phase_type}"

    return phase_converted


def convert_test_results(return_data=False):
    paths = glob(TEST_RESULT_JSONL_PATH)
    final_data = {}
    raw_data = []
    for i, path in enumerate(paths):
        print(f"[Loading ...{i+1}/{len(paths)}: {path} ...]")
        raw_data.extend(load_jsonl(path))

    total = 0
    total_without_key_error = 0
    correct = 0
    key_error = 0

    for data in raw_data:
        total += 1
        try:
            total_without_key_error += 1
            ground_truth = data["dialog"][0][0]
            game_id, phase_id, speaker_id, order = (
                ground_truth["game"],
                ground_truth["game_phase"],
                int(ground_truth["speaker_id"]),
                ground_truth["eval_labels"][0],
            )
            predicted_order = data["dialog"][0][-1]["text"]
            order = order.replace("[EO_O]", "").strip()
            predicted_order = predicted_order.replace("[EO_O]", "").strip()

            if order == predicted_order:
                correct += 1

            if game_id in final_data:
                if phase_id in final_data[game_id]:
                    if speaker_id in final_data[game_id][phase_id]:
                        assert ValueError("repeated speaker_id")
                    else:
                        final_data[game_id][phase_id][
                            COUNTRY_ID_TO_POWER[speaker_id]
                        ] = predicted_order
                        # {#'ground_truth': order,
                        # 3  'predicted': predicted_order
                        #  }
                else:
                    final_data[game_id][phase_id] = {
                        COUNTRY_ID_TO_POWER[speaker_id]: predicted_order
                        # {'ground_truth': order,
                        #   'predicted': predicted_order
                        #   }
                    }
            else:
                final_data[game_id] = {
                    phase_id: {
                        COUNTRY_ID_TO_POWER[speaker_id]: predicted_order
                        # {'ground_truth': order,
                        # 'predicted': predicted_order
                        # }
                    }
                }
        except KeyError as e:
            total_without_key_error -= 1
            print(f"I got a KeyError - reason {e}")
            key_error += 1

    final_data_no_last_phase = {}
    for game_id in final_data:
        final_data_no_last_phase[game_id] = {
            phase_id: final_data[game_id][phase_id]
            for phase_id in list(final_data[game_id].keys())[:-1]
        }

    print(f"order acc: {correct/total_without_key_error} ({correct}/{total_without_key_error})")
    print(f"key error: {key_error/total} ({key_error}/{total})")

    with open(TEST_RESULT_SAVE_JSON_PATH, "w") as fh:
        json.dump(final_data, fh)

    with open(TEST_RESULT_SAVE_JSON_NO_LAST_PHASE_PATH, "w") as fh:
        json.dump(final_data_no_last_phase, fh)

    if return_data:
        return final_data


def load_jsonl(path):
    with open(path, "r") as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        result = json.loads(json_str)
        results.append(result)
    return results


def add_common_args(argparser):
    """
    Add common commandline args to the different teachers
    """
    argparser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        choices={1, 2, 3, 4},
        help="Minimum number of dialogue turns per conversation",
    )
    return argparser


def load_data():
    """
    Load data JSONs from DATAPATH
    """
    total_data = []
    tot = len(glob(DATAPATH))
    for i, fle in enumerate(glob(DATAPATH)):
        logging.info(f"[ Loading data from path {i} / {tot}: {fle} ... ]")
        with open(fle, "r") as f:
            database = json.load(f)
            data = database[2]["data"]
            total_data += data

    return total_data


def load_order_data(opt):
    total_data = {}
    tot = len(glob(ORDER_PATH))
    print(f"[ Loading order data from path {ORDER_PATH}, {tot} games in total ... ]")
    if opt["debug"]:
        files = glob(ORDER_PATH)[: opt["debug_game_size"]]
    else:
        files = glob(ORDER_PATH)
    for i, fle in enumerate(tqdm(files)):
        with open(fle, "r") as f:
            game_id = int(re.search("game_(.*).json", fle, re.IGNORECASE).group(1))
            database = json.load(f)
            # some games contain partial order information
            total_data.setdefault(game_id, {"is_partial": "partial" in fle})
            for phase in database["phases"]:
                total_data[game_id].setdefault(phase["name"], phase)
    # TODO: chunk teacher / stream datas

    return total_data


def join_order_and_msg(
    raw_order, raw_msg, include_message_from, save_dir, special_tokens_map, with_special_token, opt
):
    """
    Join order and message information

    Temporarily return (state+msg, orders) only
    """

    def get_msg_from_speaker(conv, speaker_id):
        """
        Return messages from one speaker in one phase,
        temporarily concatenate all messagse from this one speaker,
        and ignore the interactions between speakers.
        """
        speaker_timed_msgs = []
        for from_to in conv:
            if from_to.startswith(str(speaker_id)):
                for entry in conv[from_to]:
                    if include_message_from == "speaker_msg_only":
                        select_this_msg = entry["fromCountryID"] == str(speaker_id)
                    elif include_message_from == "partner_msg_only":
                        select_this_msg = entry["fromCountryID"] != str(speaker_id)
                    elif include_message_from == "all_msg":
                        select_this_msg = True
                    else:
                        raise ValueError(f"Wrong msg_selection_method: {include_message_from}!")

                    if select_this_msg:
                        speaker = (
                            COUNTRY_ID_TO_POWER[int(entry["fromCountryID"])].capitalize()
                            if int(entry["fromCountryID"]) in COUNTRY_ID_TO_POWER
                            else "GameMaster"
                        )
                        listener = (
                            COUNTRY_ID_TO_POWER[int(entry["toCountryID"])].capitalize()
                            if int(entry["toCountryID"]) in COUNTRY_ID_TO_POWER
                            else "GameMaster"
                        )
                        speaker_timed_msg = (
                            entry["timeSent"],
                            f"{speaker} -> {listener}: {entry['message']}",
                        )
                        speaker_timed_msgs.append(speaker_timed_msg)
        # TODO order all messages by time, could be problematic when we want to keep the conversation interaction order
        sorted_speaker_timed_msgs = sorted(speaker_timed_msgs, key=lambda x: int(x[0]))

        # join this speaker's msgs
        speaker_msgs = "\n".join([timed_msg[1] for timed_msg in sorted_speaker_timed_msgs])
        # add end_or_state_token: [EO_M]
        speaker_msgs = add_end_token(
            speaker_msgs, "[EO_M]", with_special_token, special_tokens_map
        )

        no_msgs_selected = len(speaker_timed_msgs) == 0

        return speaker_msgs, no_msgs_selected

    def get_order_history(game_order, cur_phase):
        """
        get all speakers' previous orders
        #TODO under the assumption that phase in raw_order are sorted
        """
        phase_idx = list(game_order.keys())
        flat_orders = []
        for phase in game_order:
            if phase != "is_partial" and phase != "partial":
                if phase == cur_phase:
                    break
                else:
                    for speaker_id, speaker in COUNTRY_ID_TO_POWER.items():
                        order = game_order[phase]["orders"][speaker]
                        flat_order = flatten_orders(order, special_tokens_map, with_special_token,)
                        flat_order = f"{phase} {speaker.capitalize()}: {flat_order}"
                        flat_orders.append(flat_order)
        return "\n".join(flat_orders)

    def check_stat():
        # check game
        print(
            f"{len(raw_order.keys())} games in game.json, {len(raw_msg.keys())} games in message.json"
        )
        print(
            f"{len(set(raw_msg.keys())-set(raw_order.keys()))} in messgage.json but not in game.json"
        )
        print(
            f"{len(set(raw_order.keys())-set(raw_msg.keys()))} in game.json but not in messgage.json"
        )

        in_msg_not_in_order_gameid = list(set(raw_msg.keys()) - set(raw_order.keys()))
        only_order_game_ids = list(set(raw_order.keys()) - set(raw_msg.keys()))

        # check phase
        all_phases_order = []
        for game_id in tqdm(raw_order):
            all_phases_order.extend(raw_order[game_id].keys())

        all_phases_msg = []
        for game_id in tqdm(raw_msg):
            all_phases_msg.extend(raw_msg[game_id].keys())

        all_phases_msg = list(set(all_phases_msg))

        all_phases_order = list(set(all_phases_order))
        for all_phases in (all_phases_order, all_phases_msg):
            for phase in all_phases:
                if (phase.startswith("S") or phase.startswith("F") or phase.startswith("W")) and (
                    phase.endswith("M") or phase.endswith("R") or phase.endswith("A")
                ):
                    continue
                else:
                    print(f"weird phase name: {phase}")
        # season_map = {"S": "Spring", "F": "Fall", "W": "W"}
        # phase_map = {"M": "Diplomacy", "R": "Retreats", "A": "Builds"}

        # check phase match
        num_weird_game = 0
        num_weird_phases = 0
        num_all_phases_msg = 0
        for game_id in tqdm(raw_order):
            if game_id not in raw_msg:
                # inner join by game_id: game_id must be in both raw_order and raw_msg
                continue
            phase_in_msg_not_in_game = set(raw_msg[game_id].keys()) - set(
                raw_order[game_id].keys()
            )
            if len(phase_in_msg_not_in_game) > 0:
                num_weird_game += 1
            num_weird_phases += len(phase_in_msg_not_in_game)
            num_all_phases_msg += len(raw_msg[game_id].keys())
            # print(f"{game_id}: {phase_in_msg_not_in_game}")

        num_total_games_intersection = len(set(raw_order.keys()).intersection(set(raw_msg.keys())))
        print(f"{num_weird_game}/{num_total_games_intersection} games have phase-mismatch")
        print(f"{num_weird_phases}/{num_all_phases_msg} phases have phase-mismatch")

        # check NO-PRESS rule
        all_rules = []
        for game_id in raw_order:
            for phase in raw_order[game_id]:
                if phase != "is_partial" and phase != "partial":
                    all_rules.append(tuple(raw_order[game_id][phase]["state"]["rules"]))
        all_rules = list(set(all_rules))
        print(f"all_rules: {all_rules}")

        # check speaker
        speaker_set = []
        listener_set = []
        for game_id in raw_msg:
            for phase in raw_msg[game_id]:
                for from_to in raw_msg[game_id][phase]:
                    speaker = from_to.split("_")[0]
                    listener = from_to.split("_")[1]
                    speaker_set.append(speaker)
                    listener_set.append(listener)
        speaker_set = list(set(speaker_set))
        listener_set = list(set(listener_set))
        print(f"speaker id: {speaker_set}")
        print(f"listener id: {listener_set}")

    # get basic stat for the joined_json
    check_stat()

    # join data
    joined_data = {}
    num_data_with_msg = 0
    num_data = 0
    logging.info("[ Joining message and order by Phase ID ... ]")
    for game_id in tqdm(raw_order):
        # currently left join on game_id in game.jsons
        joined_data.setdefault(game_id, {})

        prev_msgs = {country: None for country in COUNTRY_ID_TO_POWER.values()}
        for phase in raw_order[game_id]:
            if phase != "is_partial" and phase != "partial":
                # set phase
                joined_data[game_id].setdefault(phase, {})
                state = flatten_state(
                    raw_order[game_id][phase]["state"], special_tokens_map, with_special_token
                )
                order_history = get_order_history(raw_order[game_id], phase)

                if game_id in raw_msg and phase in raw_msg[game_id]:
                    # phase in raw_msg, (state, msg) --> orders
                    for speaker_id, speaker in COUNTRY_ID_TO_POWER.items():
                        speaker_msg, no_msg_selected = get_msg_from_speaker(
                            raw_msg[game_id][phase], speaker_id
                        )
                        speaker_order = flatten_orders(
                            raw_order[game_id][phase]["orders"][speaker],
                            special_tokens_map,
                            with_special_token,
                        )
                        if no_msg_selected:
                            data_status = "Game_Phase_NoMsg"
                        else:
                            data_status = "Game_Phase_Msg"
                            num_data_with_msg += 1
                        cur_msg = (
                            f"{prev_msgs[speaker]} {speaker_msg}"
                            if prev_msgs[speaker] is not None
                            else f"{speaker_msg}"
                        )
                        joined_data[game_id][phase][speaker_id] = {
                            "state": state,
                            "message": speaker_msg,
                            "message_history": cur_msg,
                            "order": speaker_order,
                            "data_status": data_status,
                            "order_history": order_history,
                        }
                        num_data += 1
                        # nl = "\n"
                        # order_history = (
                        #     f"{order_history}{phase} {speaker.capitalize()}:{speaker_order}{nl}"
                        # )
                        prev_msgs[speaker] = cur_msg
                else:
                    # TODO, inner join or left join or right join or?
                    # phase not in raw_msg, state --> orders
                    # a['phases'][0]['state']['rules'] == 'NO_PRESS'
                    for speaker_id, speaker in COUNTRY_ID_TO_POWER.items():
                        speaker_msg = ""
                        # add end_or_state_token: [EO_M]
                        speaker_msg = add_end_token(
                            speaker_msg, "[EO_M]", with_special_token, special_tokens_map
                        )
                        speaker_order = flatten_orders(
                            raw_order[game_id][phase]["orders"][speaker],
                            special_tokens_map,
                            with_special_token,
                        )
                        cur_msg = (
                            f"{prev_msgs[speaker]} {speaker_msg}"
                            if prev_msgs[speaker] is not None
                            else f"{speaker_msg}"
                        )
                        joined_data[game_id][phase][speaker_id] = {
                            "state": state,
                            "message": speaker_msg,
                            "message_history": cur_msg,
                            "order": speaker_order,
                            "order_history": order_history,
                        }
                        # nl = "\n"
                        # order_history = (
                        #     f"{order_history}{phase} {speaker.capitalize()}:{speaker_order}{nl}"
                        # )

                        if game_id not in raw_msg:
                            # no message data for this game
                            joined_data[game_id][phase][speaker_id][
                                "data_status"
                            ] = "NoGame_NoPhase_NoMsg"
                        else:
                            joined_data[game_id][phase][speaker_id][
                                "data_status"
                            ] = "Game_NoPhase_NoMsg"
                        num_data += 1
                        prev_msgs[speaker] = cur_msg

    print(
        f"{round(num_data_with_msg/num_data, 4)*100}% ({num_data_with_msg}/{num_data}) has message"
    )

    # save the joined jsons
    logging.info(f"[ Saving joined_json to {save_dir} ... ]")
    game_ids = list(joined_data.keys())
    game_id_splits = np.array_split(game_ids, DATA_SPLIT_INTO)
    time1 = time.time()
    for i, split in enumerate(tqdm(game_id_splits)):
        cur_dump = {int(k): joined_data[k] for k in split}
        save_json_path = os.path.join(save_dir, f"joined_data_{i}.json")
        with open(save_json_path, "w") as fh:
            json.dump(cur_dump, fh)
        del cur_dump
    time2 = time.time()
    logging.info(f"[ Saved to {save_dir}, took {(time2-time1)/60} mins ... ]")

    return joined_data


def select_by_game_and_turn(data):
    """
    Re-structure data into a dict of dicts of dicts.

    {Game ID --> {Turn ID: {Conversation ID: []}}}
    """
    logging.info("[ Re-ordering data by GAME ID and TURN ID ... ]")
    new_data = {}
    for entry in tqdm(data):
        # select keys
        game_id = int(entry["hashed_gameID"])
        turn_id = int(entry["turn"])
        from_cnt = int(entry["fromCountryID"])
        to_cnt = int(entry["toCountryID"])
        # set game dictionary
        new_data.setdefault(game_id, {})
        # set turn dictionary
        new_data[game_id].setdefault(turn_id, {})
        # isolate conversations between two players
        keys = [f"{from_cnt}_{to_cnt}", f"{to_cnt}_{from_cnt}"]
        for key in keys:
            new_data[game_id][turn_id].setdefault(key, [])
            new_data[game_id][turn_id][key].append(entry)

    return new_data


def select_by_game_and_phase(data):
    # TODO: currently keep conversations btw players to be
    # consistent with select_by_game_and_turn,
    # but could change to single user messages later
    logging.info("[ Re-ordering data by GAME ID and PHASE ID ... ]")
    new_data = {}
    for entry in tqdm(data):
        # select keys
        game_id = int(entry["hashed_gameID"])
        phase_id = str(entry["game_phase"])
        from_cnt = int(entry["fromCountryID"])
        to_cnt = int(entry["toCountryID"])
        # set game dictionary
        new_data.setdefault(game_id, {})
        # set turn dictionary
        new_data[game_id].setdefault(phase_id, {})
        # isolate conversations between two players
        keys = [f"{from_cnt}_{to_cnt}", f"{to_cnt}_{from_cnt}"]
        for key in keys:
            new_data[game_id][phase_id].setdefault(key, [])
            new_data[game_id][phase_id][key].append(entry)

    return new_data


###########################################
# DATA ITERATION OBJECTS
###########################################


class DiplomacyConversation:
    """
    Object representing a Diplomacy conversation.

    Represents a single conversation between two users in a
    single turn in a single game of Diplomacy.
    """

    def __init__(self, key, data, turn, game):
        self.key = key
        self.to_id = int(key.split("_")[0])
        self.from_id = int(key.split("_")[1])
        self.turn = turn
        self.game = game

        self.raw_data = data
        # set up data
        self.data = self._organize_by_turns()

    def _get_msg(self):
        return {
            "game_turn": self.turn,
            "game": self.game,
            "speaker_id": self.from_id,
            "to_id": self.to_id,
            "input": None,
            "response": None,
        }

    def _organize_by_turns(self):
        # first sort the data by time sent
        sorted_data = sorted(self.raw_data, key=lambda y: int(y["timeSent"]))

        # next combine consecutive messages from a single user
        combined_data = []
        prev = {"fromCountryID": None}
        for entry in sorted_data:
            if entry["fromCountryID"] == prev["fromCountryID"]:
                prev["message"] += "\n" + entry["message"]
            else:
                if prev["fromCountryID"] is not None:
                    combined_data.append(prev)
                prev = entry

        if not combined_data or combined_data[-1] != prev:
            if prev["fromCountryID"] is not None:
                # add last message
                combined_data.append(prev)

        # now determine the first speaker and possibly offset the data
        data = []
        first_speaker = combined_data[0]["fromCountryID"]
        if int(first_speaker) != self.from_id:
            first_turn = {
                "fromCountryID": self.from_id,
                "message": "__SILENCE__",
            }
            lst = [first_turn] + combined_data
        else:
            lst = combined_data

        len_conv = len(combined_data) // 2
        for i in range(len_conv):
            first = lst[2 * i]["message"]
            second = lst[2 * i + 1]["message"]
            msg = self._get_msg()
            msg["input"] = first
            msg["response"] = second
            data.append(msg)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiplomacyMessageOrderPair:
    """
    Object representing a Diplomacy (state+message)-order pair.

    Represents (state+message)-order pairs from all speakers
    in one phase in a single game of Diplomacy.
    """

    def __init__(self, data, phase, game):
        self.data = data
        self.phase = phase
        self.game = game

        self.raw_data = data
        # set up data
        self.data = self._organize_by_speakers()

    def _get_msg(self):
        return {
            "game_phase": self.phase,
            "game": self.game,
            "speaker_id": None,
            "speaker": None,
            "state": None,
            "message": None,
            "order": None,
        }

    def _organize_by_speakers(self):
        data = []
        for speaker_id in self.raw_data:
            msg = self._get_msg()
            msg["speaker_id"] = speaker_id
            msg["speaker"] = COUNTRY_ID_TO_POWER[int(speaker_id)].capitalize()
            for k, v in self.raw_data[speaker_id].items():
                msg[k] = v
            data.append(msg)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiplomacyTurn:
    """
    Object representing a Diplomacy turn.
    """

    def __init__(self, data, turn_idx, game_idx):
        self.raw_data = data
        self.key_map = {i: key for i, key in enumerate(data.keys())}
        self.turn = turn_idx
        self.game = game_idx

    def __len__(self):
        return len(self.raw_data.keys())

    def __iter__(self):
        self.iterator_idx = 0
        return self

    def __getitem__(self, idx):
        conv = DiplomacyConversation(
            self.key_map[idx], self.raw_data[self.key_map[idx]], self.turn, self.game,
        )
        return conv

    def __next__(self):
        if self.iterator_idx >= len(self):
            raise StopIteration

        conv = DiplomacyConversation(
            self.key_map[self.iterator_idx],
            self.raw_data[self.key_map[self.iterator_idx]],
            self.turn,
            self.game,
        )
        self.iterator_idx += 1

        return conv


class DataIterator:
    """
    Iterator for Diplomacy message data.

    Iterates through turns among ALL games.
    """

    def __init__(self, raw_data=None):
        if raw_data is None:
            self.raw_data = select_by_game_and_turn(load_data())
        else:
            self.raw_data = raw_data
        self.key_map = {i: key for i, key in enumerate(self.raw_data.keys())}
        self.num_games = len(self.raw_data.keys())
        self.num_turns = sum(len(game.keys()) for game in self.raw_data.values())

    def __len__(self):
        """
        Number of turns out of all games
        """
        return self.num_turns

    def __iter__(self):
        self.game_idx = 0
        self.turn_idx = 0
        return self

    def len_game(self, game):
        return len(game.keys())

    def __next__(self):
        """
        Iterates through games and turns inside games
        """
        if self.game_idx >= self.num_games:
            raise StopIteration

        curr_game = self.raw_data[self.key_map[self.game_idx]]
        if self.turn_idx >= self.len_game(curr_game):
            self.game_idx += 1
            self.turn_idx = 0
            if self.game_idx >= self.num_games:
                raise StopIteration

            curr_game = self.raw_data[self.key_map[self.game_idx]]

        turn_keys = {i: key for i, key in enumerate(curr_game.keys())}
        turn_key = turn_keys[self.turn_idx]

        curr_turn = DiplomacyTurn(curr_game[turn_key], turn_key, self.key_map[self.game_idx])
        self.turn_idx += 1

        return curr_turn


class MessageOrderDataIterator:
    """
    Iterator for Diplomacy (message, order) pair data.

    Iterates through (message, order) pair, phase by phase among ALL games.
    """

    def __init__(
        self, opt, raw_order=None, raw_msg=None,
    ):
        self.opt = deepcopy(opt)
        # listener only, speaker only, both sides
        self.include_message_from = opt["include_message_from"]
        # if the joined json exists, should we overwrite it?
        self.overwrite_joined_json = opt["overwrite_joined_json"]
        # the saved joined json will contain "A" or "__A__"
        self.with_special_token = self.opt["with_special_token"]

        # load special tokens
        self._load_special_tokens()

        # build/load json
        self._build_or_load_joined_data(raw_order=raw_order, raw_msg=raw_msg)

        # key maps
        self.key_map = {i: key for i, key in enumerate(self.joined_data.keys())}
        self.num_games = len(self.joined_data.keys())
        self.num_phases = sum(len(game.keys()) for game in self.joined_data.values())
        self.num_pairs = sum(
            sum([len(phase.keys()) for phase in game.values()])
            for game in self.joined_data.values()
        )

    def _load_special_tokens(self):
        if self.with_special_token:
            self.special_tokens, self.special_tokens_map = load_special_tokens()
        else:
            self.special_tokens, self.special_tokens_map = None, None

    def _build_or_load_joined_data(self, raw_order=None, raw_msg=None):
        # we'll save the joined json
        overwrite = False
        json_dir = f"{JOINED_JSON_PATH_PREFIX}/dumps_State_OrderHistory_MessageHistory-{self.include_message_from.replace('_', '-')}-SpecialToken-{self.with_special_token}_order/"

        if os.path.exists(json_dir) and self.overwrite_joined_json:
            confirm_overwrite = input(
                f"About to overwrite the joined_json in {json_dir}, do you want to do that? [Y/y] for YES, anything else for NO"
            )
            overwrite = confirm_overwrite in ["Y", "y"]
        elif not os.path.exists(json_dir):
            print(f"[{json_dir} doesn't exist, will build and save it ...]")
        else:
            print(f"[{json_dir} exists, will load it ...]")

        if os.path.exists(json_dir) and (not overwrite):
            logging.info(f"[ Loading {json_dir} ...]")
            time1 = time.time()
            self.joined_data = self._load_joined_json(json_dir)
            time2 = time.time()
            logging.info(f"[ Loading finished, took {(time2-time1)/60} mins ...]")
        else:
            if not os.path.exists(json_dir):
                if self.opt["debug"]:
                    confirm_save_debug_data = input(
                        f"It seems you are in debug mode, and trying to build and"
                        f"save the data (you will only save {self.opt['debug_game_size']} games in {json_dir})? "
                        f"Are you sure? [Y/y] for YES, anything else for NO"
                    )
                    confirm_save_debug_data = confirm_save_debug_data in ["Y", "y"]
                    if not confirm_save_debug_data:
                        print("Aborting... Please remove --debug in the next run")
                        sys.exit(-1)
                os.makedirs(json_dir)

            if raw_order is None:
                self.raw_order = load_order_data(self.opt)
                self.raw_msg = select_by_game_and_phase(load_data())
            else:
                self.raw_order = raw_order
                self.raw_msg = raw_msg

            self.joined_data = join_order_and_msg(
                self.raw_order,
                self.raw_msg,
                self.include_message_from,
                save_dir=json_dir,
                special_tokens_map=self.special_tokens_map,
                with_special_token=self.with_special_token,
                opt=self.opt,
            )

    def _load_joined_json(self, dump_path):
        each_dump_contains = TOTAL_GAMES // DATA_SPLIT_INTO
        how_many_dumps_to_load = self.opt["debug_game_size"] // each_dump_contains + 1

        dump_json_path = os.path.join(dump_path, "*.json")
        dump_json_paths = glob(dump_json_path)
        if self.opt["debug"]:
            files = dump_json_paths[:how_many_dumps_to_load]
        else:
            files = dump_json_paths

        games = {}
        for i, fle in enumerate(tqdm(files)):
            with open(fle, "r") as f:
                game = json.load(f)
                games.update(game)

        return games

    def __len__(self):
        """
        Number of phases out of all games
        """
        return self.num_phases

    def __iter__(self):
        self.game_idx = 0
        self.phase_idx = 0
        # self.speaker_idx = 0
        return self

    def len_game(self, game):
        # equals number of phases
        return len(game.keys())

    def __next__(self):
        """
        Iterates through phases inside games
        """
        if self.game_idx >= self.num_games:
            raise StopIteration

        curr_game = self.joined_data[self.key_map[self.game_idx]]
        if self.phase_idx >= self.len_game(curr_game):
            self.game_idx += 1
            self.phase_idx = 0
            if self.game_idx >= self.num_games:
                raise StopIteration

            curr_game = self.joined_data[self.key_map[self.game_idx]]

        phase_keys = {i: key for i, key in enumerate(curr_game.keys())}
        phase_key = phase_keys[self.phase_idx]

        curr_phase = DiplomacyMessageOrderPair(
            curr_game[phase_key], phase_key, self.key_map[self.game_idx]
        )
        self.phase_idx += 1

        return curr_phase
