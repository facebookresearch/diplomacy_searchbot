#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
from glob import glob
import re
import os
import time

import parlai_diplomacy.tasks.language_diplomacy.config as cfg

###########################################
# CONSTANTS
###########################################

DATAPATH = (
    "/checkpoint/fairdiplomacy/processed_chat_jsons/game_phases/redacted_messages_runthree_*.json"
)
ORDER_PATH = (
    "/checkpoint/fairdiplomacy/processed_orders_jsons/game_*.json*"  # some are json.partial
)
ORDER_JSON_PATH = "/checkpoint/fairdiplomacy/processed_orders_jsons/combined_json/game.json"
JOINED_JSON_PATH = "/checkpoint/fairdiplomacy/joined_jsons/msg_order.json"
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


def flatten_orders(order_list):
    """
    Flatten the order in game*.json
    """
    if type(order_list) is not list:
        order_list = [str(order_list)]
    # sort the orders
    order_list = sorted(order_list)
    # join the orders
    flat_order_str = "; ".join(order_list)
    # add end_of_order token: [EO_O]
    flat_order_str = f"{flat_order_str} [EO_O]"

    return flat_order_str


def flatten_state(state):
    """
    Flatten the state in game*.json
    """

    def flatten_country_status(key, country_status):
        status_list = []
        for country, status in country_status.items():
            if type(status) is not list:
                status = [str(status)]
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
    final_state_str = f"{final_state_str} [EO_STATE]"

    return final_state_str


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


def load_data():
    """
    Load data JSONs from DATAPATH
    """
    total_data = []
    tot = len(glob(DATAPATH))
    for i, fle in enumerate(glob(DATAPATH)):
        print(f"[ Loading data from path {i} / {tot}: {fle} ... ]")
        with open(fle, "r") as f:
            database = json.load(f)
            data = database[2]["data"]
            total_data += data

    return total_data


def load_order_data():
    total_data = {}
    tot = len(glob(ORDER_PATH))
    print(f"[ Loading order data from path {ORDER_PATH}, {tot} games in total ... ]")
    for i, fle in enumerate(tqdm(glob(ORDER_PATH))):
        with open(fle, "r") as f:
            game_id = int(re.search("game_(.*).json", fle, re.IGNORECASE).group(1))
            database = json.load(f)
            # some games contain partial order information
            total_data.setdefault(game_id, {"partial": "partial" in fle})
            for phase in database["phases"]:
                total_data[game_id].setdefault(phase["name"], phase)
    # TODO: chunk teacher / stream datas
    return total_data


def join_order_and_msg(raw_order, raw_msg):
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
                    if cfg.HOW_TO_SELECT_MSG == cfg.SPEAKER_MSG_ONLY:
                        select_this_msg = entry["fromCountryID"] == str(speaker_id)
                    elif cfg.HOW_TO_SELECT_MSG == cfg.PARTNER_MSG_ONLY:
                        select_this_msg = entry["fromCountryID"] != str(speaker_id)
                    elif cfg.HOW_TO_SELECT_MSG == cfg.ALL_MSG:
                        select_this_msg = True
                    else:
                        raise ValueError(f"Wrong msg_selection_method: {cfg.HOW_TO_SELECT_MSG}!")

                    if select_this_msg:
                        speaker = COUNTRY_ID_TO_POWER[int(entry["fromCountryID"])].capitalize()
                        speaker_timed_msg = (
                            entry["timeSent"],
                            f"{speaker}: {entry['message']}",
                        )
                        speaker_timed_msgs.append(speaker_timed_msg)

        # TODO: order all messages by time, could be problematic when we want to keep
        # the conversation interaction order
        sorted_speaker_timed_msgs = sorted(speaker_timed_msgs, key=lambda x: int(x[0]))

        # join this speaker's msgs
        speaker_msgs = "\n".join([timed_msg[1] for timed_msg in sorted_speaker_timed_msgs])
        return speaker_msgs

    joined_data = {}
    print("[ Joining message and order by Phase ID ... ]")
    for game_id in tqdm(raw_order):
        if game_id not in raw_msg:
            # inner join by game_id: game_id must be in both raw_order and raw_msg
            continue
        joined_data.setdefault(game_id, {})
        for phase in raw_order[game_id]:
            if phase != "partial":
                # set phase
                joined_data[game_id].setdefault(phase, {})
                state = flatten_state(raw_order[game_id][phase]["state"])
                if phase in raw_msg[game_id]:
                    # phase in raw_msg, (state, msg) --> orders
                    for speaker_id, speaker in COUNTRY_ID_TO_POWER.items():
                        speaker_token = f"{speaker.capitalize()} message:"
                        speaker_msg = get_msg_from_speaker(raw_msg[game_id][phase], speaker_id)
                        speaker_order = flatten_orders(
                            raw_order[game_id][phase]["orders"][speaker]
                        )
                        joined_data[game_id][phase][speaker_id] = {
                            "state_msg": f"{state} {speaker_token} {speaker_msg}",
                            "order": f"{speaker_order}",
                        }
                else:
                    # TODO: inner join or left join or right join or?
                    # phase not in raw_msg, state --> orders
                    for speaker_id, speaker in COUNTRY_ID_TO_POWER.items():
                        speaker_token = f"{speaker.capitalize()} message:"
                        speaker_msg = ""
                        speaker_order = flatten_orders(
                            raw_order[game_id][phase]["orders"][speaker]
                        )
                        joined_data[game_id][phase][speaker_id] = {
                            "state_msg": f"{state} {speaker_token} {speaker_msg}",
                            "order": f"{speaker_order}",
                        }

    print(f"[ Saving joined_json to {JOINED_JSON_PATH} ... ]")
    time1 = time.time()
    with open(JOINED_JSON_PATH, "w") as fh:
        json.dump(joined_data, fh)
    time2 = time.time()
    print(f"[ Saved to {JOINED_JSON_PATH}, took {(time2-time1)/60} mins ... ]")

    return joined_data


def select_by_game_and_turn(data):
    """
    Re-structure data into a dict of dicts of dicts.

    {Game ID --> {Turn ID: {Conversation ID: []}}}
    """
    print("[ Re-ordering data by GAME ID and TURN ID ... ]")
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
    print("[ Re-ordering data by GAME ID and PHASE ID ... ]")
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
            "state_and_message": None,
            "order": None,
        }

    def _organize_by_speakers(self):
        data = []
        for speaker_id in self.raw_data:
            msg = self._get_msg()
            msg["speaker_id"] = speaker_id
            msg["speaker"] = COUNTRY_ID_TO_POWER[int(speaker_id)].capitalize()
            msg["state_and_message"] = self.raw_data[speaker_id]["state_msg"]
            msg["order"] = f"{msg['speaker'].capitalize()}: {self.raw_data[speaker_id]['order']}"
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

    def __init__(self, raw_order=None, raw_msg=None):
        overwrite = False
        if cfg.OVERWRITE_JOINED_JSON:
            confirm_overwrite = input(
                "About to overwrite the joined_json, are you sure you want to do that? "
                "[Y/y] for YES, anything else for NO"
            )
            overwrite = confirm_overwrite in ["Y", "y"]

        if os.path.exists(JOINED_JSON_PATH) and (not overwrite):
            print(f"[ Loading {JOINED_JSON_PATH} ...]")
            time1 = time.time()
            with open(JOINED_JSON_PATH, "r") as fh:
                self.joined_data = json.load(fh)
            time2 = time.time()
            print(f"[ Loading finished, took {(time2-time1)/60} mins ...]")
        else:
            if raw_order is None:
                self.raw_order = load_order_data()
                self.raw_msg = select_by_game_and_phase(load_data())
            else:
                self.raw_order = raw_order
                self.raw_msg = raw_msg
            # game_id must be in both the raw_order and raw_msg
            self.joined_data = join_order_and_msg(self.raw_order, self.raw_msg)
        self.key_map = {i: key for i, key in enumerate(self.joined_data.keys())}
        self.num_games = len(self.joined_data.keys())
        self.num_phases = sum(len(game.keys()) for game in self.joined_data.values())

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
