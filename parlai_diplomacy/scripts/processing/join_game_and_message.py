#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to join and save the data (join game.jsons and message),
only need to be run once unless there is change in the data format
Example usage
```
python join_game_and_message.py --debug --use_fixed_split --save_folder test_folder
```
args:
--debug: to load only 5 games,
--use_fixed_split: to use the fixed train/val split used by fairdip
--save_folder: save folder under /checkpoint/fairdiplomacy/press_diplomacy/joined_json,
e.g. will save to /checkpoint/fairdiplomacy/press_diplomacy/joined_json/test_folder

"""
import json
from tqdm import tqdm
import os
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
import argparse

import parlai.utils.logging as logging
import parlai_diplomacy.utils.game_loading as game_loading
import parlai_diplomacy.utils.game_to_sequence_formatting as game_formatting
import parlai_diplomacy.tasks.common_task_utils as utls


# joined json will be saved to
JOINED_JSON_PATH_PREFIX = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons"

# joined_json will be splitted into
TOTAL_GAMES = 54622
SPLIT_INTO = 1000
TRAIN_SET_SPLIT_PATH = "/private/home/apjacob/data_cache_fb_minrating0.5.pt_train.json"
VALID_SET_SPLIT_PATH = "/private/home/apjacob/data_cache_fb_minrating0.5.pt_valid.json"


def join_order_and_msg(
    raw_order,
    raw_msg,
    include_message_from,
    with_special_token,
    special_tokens_map,
    use_fixed_split=False,
):
    """
    Join order and message information
    args:
        include_message_from: include speaker messgage only, partner msg only, or both sides
        with_special_token: __A__ or A


    The data returned is of the following format
    {
        <Game ID>: {
            <Phase ID>: {
                <Power ID>: {
                    'metadata': <dict of miscellaneous metadata>,
                    'state': <state>,
                    'state_history': <state_history>,
                    'order_history': <order history>,
                    'message_history': <message history>,
                    'message_history_processed': <process the message history to a string>,
                    'message': <list of messages from curr phase>,
                    'message_processed': <process the message to a string>
                    'all_orders': <list of all orders>
                }
            }
        }
    }

    """
    if use_fixed_split:
        # read the fixed train/val split
        with open(TRAIN_SET_SPLIT_PATH, "r") as fh:
            TRAIN_SET_SPLIT = json.load(fh)
        with open(VALID_SET_SPLIT_PATH, "r") as fh:
            VALID_SET_SPLIT = json.load(fh)

    # join data
    joined_data = {}
    num_data_with_msg = 0
    num_data = 0
    logging.info("[ Joining message and order by Phase ID ... ]")
    for game_id in tqdm(raw_order):
        # currently left join on game_id in game.jsons
        joined_data.setdefault(game_id, {})
        # build msg history iteratively, we could also build order/state history iteratively, but
        msg_history_processed = {country: "" for country in utls.COUNTRY_ID_TO_POWER.values()}
        msg_history = {country: [] for country in utls.COUNTRY_ID_TO_POWER.values()}

        for phase in raw_order[game_id]:
            if phase == "is_partial" and phase == "partial":
                continue

            # set phase
            joined_data[game_id][phase] = {}

            # get current phase state_str, the same for all speakers
            state = raw_order[game_id][phase]["state"]

            # get current phase order_history, the same for all speakers
            order_history = get_order_history(
                raw_order[game_id], phase, with_special_token, special_tokens_map
            )

            # get state history
            state_history = get_state_history(raw_order[game_id], phase)

            # get the order for each power
            if "orders" in raw_order[game_id][phase]:
                all_orders = {
                    speaker_id: raw_order[game_id][phase]["orders"].get(speaker, [])
                    for speaker_id, speaker in utls.COUNTRY_ID_TO_POWER.items()
                }
            else:
                all_orders = {speaker_id: [] for speaker_id in utls.COUNTRY_ID_TO_POWER.keys()}

            # start for each speaker
            for speaker_id, speaker in utls.COUNTRY_ID_TO_POWER.items():
                # get current split
                if use_fixed_split:
                    if (TRAIN_SET_SPLIT.get(str(game_id), {}).get(phase, {}) is not None) and (
                        speaker in TRAIN_SET_SPLIT.get(str(game_id), {}).get(phase, {})
                    ):
                        which_split = "train"
                    elif (VALID_SET_SPLIT.get(str(game_id), {}).get(phase, {}) is not None) and (
                        speaker in VALID_SET_SPLIT.get(str(game_id), {}).get(phase, {})
                    ):
                        which_split = "valid"
                    else:
                        which_split = "leftover"
                which_split = "all"

                # get current phase speaker_order
                speaker_order = all_orders[speaker_id]

                # get current phase speaker_msg and data_status
                if game_id in raw_msg and phase in raw_msg[game_id]:
                    speaker_processed_msg, no_msg_selected = get_msg_from_speaker(
                        raw_msg[game_id][phase],
                        speaker_id,
                        include_message_from,
                        with_special_token,
                        special_tokens_map,
                    )
                    speaker_msg = get_msgs_raw_for_speaker(raw_msg[game_id][phase], speaker_id)
                    if no_msg_selected:
                        data_status = "Game_Phase_NoMsg"  # Game in msg.table, phase in msg.table, but no msg in this phase
                    else:
                        data_status = "Game_Phase_Msg"  # Game in msg.table, phase in msg.table, has msg in this phase
                        num_data_with_msg += 1
                else:
                    # current phase speaker_messgae, and add end_or_message_token: [EO_M]
                    speaker_processed_msg = ""
                    speaker_processed_msg = game_formatting.add_end_token(
                        speaker_processed_msg, "[EO_M]", with_special_token, special_tokens_map
                    )
                    speaker_msg = []
                    # data status
                    if game_id not in raw_msg:
                        # no message data for this game
                        data_status = "NoGame_NoPhase_NoMsg"  # Game not in msg.table, phase not in msg.table, no msg in this phase
                    else:
                        data_status = "Game_NoPhase_NoMsg"  # Game in msg.table, phase not in msg.table, no msg in this phase

                joined_data[game_id][phase][speaker_id] = {
                    "metadata": {
                        "game_id": game_id,
                        "phase_id": phase,
                        "speaker_id": speaker_id,
                        "speaker": utls.COUNTRY_ID_TO_POWER[speaker_id],
                        "data_status": data_status,
                        "which_split": which_split,
                    },
                    "state": state,
                    "state_history": state_history,
                    "message": speaker_msg,
                    "message_processed": speaker_processed_msg,
                    "message_history": deepcopy(msg_history[speaker]),
                    "message_history_processed": msg_history_processed[speaker],
                    "order": speaker_order,
                    "order_history": order_history,
                    "all_orders": all_orders,
                }

                # add one more data point
                num_data += 1
                # append to the message_history
                if msg_history_processed[speaker]:
                    msg_history_processed[speaker] += "\n"
                msg_history_processed[speaker] += f"{speaker_processed_msg}"
                msg_history[speaker].append(speaker_msg)

    # how many data have message at the current phase
    print(
        f"{round(num_data_with_msg/num_data, 4)*100}% ({num_data_with_msg}/{num_data}) has message"
    )

    return joined_data


def split_train_val(joined_data, use_fixed_split=True):
    """split train/val
        if not use_fixed_split, all joined_data will be saved to one, and later in teacher we split 95/5
    Args:
        joined_data: the joined json
        use_fixed_split: use_fixed_split or not
    """
    if use_fixed_split:
        # read the fixed train/val split
        with open(TRAIN_SET_SPLIT_PATH, "r") as fh:
            TRAIN_SET_SPLIT = json.load(fh)
        with open(VALID_SET_SPLIT_PATH, "r") as fh:
            VALID_SET_SPLIT = json.load(fh)

    # if we don't want to use the fixed train/valid set, just return all the joined_data,
    # and joined_data will be split into chunks, later 95/5 for train/valid
    if not use_fixed_split:
        joined_data_split = {"all": joined_data}
        return joined_data_split

    # initialize a new joined_data_split
    joined_data_split = {
        "train": defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
        "valid": defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
        "leftover": defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
    }

    # check if all train/val data exist in joined_data
    for split_name, split_set in (("train", TRAIN_SET_SPLIT), ("valid", VALID_SET_SPLIT)):
        for game_id in split_set:
            for phase_id in split_set[game_id]:
                if joined_data.get(game_id, {}).get(phase_id, {}) is None:
                    raise ValueError(
                        f"in {split_name}, {game_id}-{phase_id} doesn't exist in joined_data!"
                    )

    # start splitting into train/valid/leftover
    for game_id in joined_data:
        for phase_id in joined_data[game_id]:
            for speaker_id in joined_data[game_id][phase_id]:
                which_split = joined_data[game_id][phase_id][speaker_id]["which_split"]
                # put into that split
                joined_data_split[which_split][game_id][phase_id][speaker_id] = joined_data[
                    game_id
                ][phase_id][speaker_id]

    return joined_data_split


def split_and_save_json_chunks(joined_data, save_path, split_into=500):
    """[split the json into chunks and save]

    Args:
        joined_data: data to save
        save_path: path to save the chunks
        split_into (int, optional): how many chunks to split into. Defaults to 500.
    """

    # split the whole json into 500 and save the chunks
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"[ Saving joined_json to {save_path} ... ]")
    game_ids = list(joined_data.keys())
    game_id_splits = np.array_split(game_ids, split_into)
    time1 = time.time()
    for i, split in enumerate(tqdm(game_id_splits)):
        if len(split) == 0:
            continue
        cur_dump = {int(k): joined_data[k] for k in split}
        save_json_path = os.path.join(save_path, f"joined_data_{i}.json")
        with open(save_json_path, "w") as fh:
            json.dump(cur_dump, fh)
        del cur_dump
    time2 = time.time()
    logging.info(f"[ Saved to {save_path}, took {(time2-time1)/60} mins ... ]")


def get_msg_from_speaker(
    conv, speaker_id, include_message_from, with_special_token, special_tokens_map
):
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
                        utls.COUNTRY_ID_TO_POWER[int(entry["fromCountryID"])].capitalize()
                        if int(entry["fromCountryID"]) in utls.COUNTRY_ID_TO_POWER
                        else "GameMaster"
                    )
                    listener = (
                        utls.COUNTRY_ID_TO_POWER[int(entry["toCountryID"])].capitalize()
                        if int(entry["toCountryID"]) in utls.COUNTRY_ID_TO_POWER
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
    speaker_msgs = game_formatting.add_end_token(
        speaker_msgs, "[EO_M]", with_special_token, special_tokens_map
    )

    no_msgs_selected = len(speaker_timed_msgs) == 0

    return speaker_msgs, no_msgs_selected


def get_msgs_raw_for_speaker(conv, speaker_id):
    """
    Get all messages for a spaker without doing any formatting
    """
    messages = []
    for from_to in conv:
        if from_to.startswith(str(speaker_id)):
            for entry in conv[from_to]:
                speaker = (
                    utls.COUNTRY_ID_TO_POWER[int(entry["fromCountryID"])].capitalize()
                    if int(entry["fromCountryID"]) in utls.COUNTRY_ID_TO_POWER
                    else "GameMaster"
                )
                listener = (
                    utls.COUNTRY_ID_TO_POWER[int(entry["toCountryID"])].capitalize()
                    if int(entry["toCountryID"]) in utls.COUNTRY_ID_TO_POWER
                    else "GameMaster"
                )
                message = {
                    "speaker": speaker,
                    "listener": listener,
                    "message": entry["message"],
                    "time_sent": entry["timeSent"],
                }
                messages.append(message)

    messages = sorted(messages, key=lambda x: int(x["time_sent"]))

    return messages


def get_order_history(
    game_order, cur_phase, with_special_token, special_tokens_map, flatten=False
):
    """
    get all speakers' previous orders
    #TODO under the assumption that phase in raw_order are sorted
    # the phases in game.jsons are sorted; if not sorted, can sort by fairdiplomacy.game.sort_phase_key
    """
    orders = {}
    flat_orders = []
    for phase in game_order:
        if phase != "is_partial" and phase != "partial":
            if phase == cur_phase:
                break
            else:
                orders[phase] = {}
                for _, speaker in utls.COUNTRY_ID_TO_POWER.items():
                    if "orders" in game_order[phase]:
                        order = game_order[phase]["orders"].get(speaker, [])
                    else:
                        order = []
                    if flatten:
                        flat_order = game_formatting.flatten_orders(
                            order, special_tokens_map, with_special_token,
                        )
                        flat_order = f"{phase} {speaker.capitalize()}: {flat_order}"
                        flat_orders.append(flat_order)
                    else:
                        orders[phase][speaker] = order
    if flatten:
        return "\n".join(flat_orders)
    else:
        return orders


def get_state_history(game_dct, cur_phase):
    """
    Return the state history
    """
    state_hist = {}
    for phase in game_dct:
        if phase != "is_partial" and phase != "partial":
            if phase == cur_phase:
                break

            state_hist[phase] = game_dct[phase]["state"]

    return state_hist


def check_stat(raw_order, raw_msg):
    """
    check game.jsons and message.table stats when joining the two tables
    """
    print(
        f"{len(raw_order.keys())} games in game.json, {len(raw_msg.keys())} games in message.json"
    )
    print(
        f"{len(set(raw_msg.keys())-set(raw_order.keys()))} in messgage.json but not in game.json"
    )
    print(
        f"{len(set(raw_order.keys())-set(raw_msg.keys()))} in game.json but not in messgage.json"
    )

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

    # check phase match
    num_weird_game = 0
    num_weird_phases = 0
    num_all_phases_msg = 0
    for game_id in tqdm(raw_order):
        if game_id not in raw_msg:
            # inner join by game_id: game_id must be in both raw_order and raw_msg
            continue
        phase_in_msg_not_in_game = set(raw_msg[game_id].keys()) - set(raw_order[game_id].keys())
        if len(phase_in_msg_not_in_game) > 0:
            num_weird_game += 1
        num_weird_phases += len(phase_in_msg_not_in_game)
        num_all_phases_msg += len(raw_msg[game_id].keys())

    num_total_games_intersection = len(set(raw_order.keys()).intersection(set(raw_msg.keys())))
    print(f"{num_weird_game}/{num_total_games_intersection} games have phase-mismatch")
    print(f"{num_weird_phases}/{num_all_phases_msg} phases have phase-mismatch")

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


def main(
    save_path,
    debug,
    use_fixed_split=True,
    data_format="dipcc",
    include_message_from="all_msg",
    with_special_token=False,
    special_tokens_map=None,
):
    # load raw_order and raw_msg
    raw_order = game_loading.load_sql_format(debug=debug, data_format=data_format)
    raw_order = game_loading.organize_game_dict_by_phase(raw_order)
    raw_msg = utls.select_by_game_and_phase(utls.load_message_data())

    # get basic stat for the joined_json
    check_stat(raw_order, raw_msg)

    # join the order and msg
    joined_data = join_order_and_msg(
        raw_order,
        raw_msg,
        include_message_from,
        with_special_token,
        special_tokens_map,
        use_fixed_split,
    )

    # split the train/valid according to a fix set
    joined_data_split = split_train_val(joined_data, use_fixed_split)

    for split_name in joined_data_split:
        split_save_path = os.path.join(save_path, split_name)
        split_and_save_json_chunks(
            joined_data_split[split_name], split_save_path, split_into=SPLIT_INTO
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # TODO should be depreciated once we've a more structured joined_data
        "--with_special_token",
        action="store_true",
        help="with __A__ or not, default to False",
    )
    parser.add_argument(
        "--debug", action="store_true", help="in debug mode, load fewer games",
    )
    parser.add_argument(
        "--use_fixed_split", action="store_true", help="used the fixed train/val split",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="dipcc",
        choices={"dip", "dipcc"},
        help="data format to use",
    )
    parser.add_argument(
        # TODO should be depreciated once we've a more structured joined_data
        "--include_message_from",
        type=str,
        default="all_msg",
        choices={"speaker_msg_only", "partner_msg_only", "all_msg"},
        help="whose messages to include (speaker-only, listener-only, or all messages)",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="save folder under /checkpoint/fairdiplomacy/press_diplomacy/joined_json",
    )

    args = parser.parse_args()

    if args.with_special_token:
        # if with_special_token, we load the __A__ tokens, and the saved json will look like all __A__,
        # so we don't have to insert __ in the teacher, but should be depreciated if we want to
        # process and insert __ in the teacher
        special_tokens, special_tokens_map = utls.load_special_tokens()
    else:
        special_tokens, special_tokens_map = None, None

    # save path
    save_folder = args.save_folder
    if save_folder is None:
        # create a default saved folder based on the args
        save_folder = f"include_msg={args.include_message_from}-special_tokens={args.with_special_token}-format={args.data_format}"

    save_path = os.path.join(JOINED_JSON_PATH_PREFIX, save_folder)
    assert (
        save_path != JOINED_JSON_PATH_PREFIX
    )  # must specify a folder under JOINED_JSON_PATH_PREFIX

    logging.warn(f"Will save JSONs to path: {save_path}")
    if os.path.isdir(save_path):
        # check to make sure the overwrite is intentional
        cont = input(
            f"\n\nWARNING: The folder {save_path} already exists. Type Y if you wish to continue and overwrite this path: "
        )
        if cont.upper() != "Y":
            raise RuntimeError("Exiting.")
        else:
            logging.warn("Continuing...")

    main(
        save_path=save_path,
        debug=args.debug,
        use_fixed_split=args.use_fixed_split,
        data_format=args.data_format,
        include_message_from=args.include_message_from,
    )
