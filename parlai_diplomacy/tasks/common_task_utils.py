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
import parlai_diplomacy.utils.game_loading as game_loading
import parlai_diplomacy.utils.datapath_constants as constants
import parlai_diplomacy.scripts.processing.join_game_and_message as data_joining

# TODO: this file needs some major cleanup

###########################################
# CONSTANTS
###########################################

# special token path
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
    data = load_message_data()
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


def replace_with_special_token(text):
    """replace order and state_str with special tokens
        A S LON --> __A__ __S__ __LON__

    Args:
        text ([str]): A S LON

    Returns:
        [str]: __A__ __S__ __LON__
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


def phase_abbrev_to_phase(phase):
    """
    https://github.com/diplomacy/diplomacy/blob/master/diplomacy/integration/webdiplomacy_net/game.py#L20-L27
    if phase == 'Builds':
        season = 'W'
    {'Diplomacy': 'M', 'Retreats': 'R', 'Builds': 'A'}
    """
    phase_map = {"M": "Diplomacy", "R": "Retreats", "A": "Builds"}
    season_year = phase[:5]
    phase_type = phase_map[phase[5:]]
    phase_converted = f"{season_year} {phase_type}"

    return phase_converted


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


def load_message_data():
    """
    Load data JSONs from DATAPATH
    """
    total_data = []
    tot = len(glob(constants.NONCHUNK_DIALOGUE_PATH))
    for i, fle in enumerate(glob(constants.NONCHUNK_DIALOGUE_PATH)):
        logging.info(f"[ Loading data from path {i} / {tot}: {fle} ... ]")
        with open(fle, "r") as f:
            database = json.load(f)
            data = database[2]["data"]
            total_data += data

    return total_data


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
            self.raw_data = select_by_game_and_turn(load_message_data())
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
        self, opt, raw_order=None, raw_msg=None, json_dir=None,
    ):
        self.opt = deepcopy(opt)
        # listener only, speaker only, both sides
        self.include_message_from = opt.get("include_message_from")
        # if the joined json exists, should we overwrite it?
        self.overwrite_joined_json = opt.get("overwrite_joined_json")
        # the saved joined json will contain "A" or "__A__"
        self.with_special_token = opt.get("with_special_token")
        # json path to load (if exists) or save the joined data (if not exists)
        self.json_dir = (
            json_dir
            if json_dir is not None
            else f"{data_joining.JOINED_JSON_PATH_PREFIX}/dumps_State_OrderHistory_MessageHistory-{self.include_message_from.replace('_', '-')}-SpecialToken-{self.with_special_token}_order/"
        )
        # joined_json split information for backward compatibility
        self.TOTAL_GAMES = 54622
        self.DATA_SPLIT_INTO = 500

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

        if os.path.exists(self.json_dir) and self.overwrite_joined_json:
            confirm_overwrite = input(
                f"About to overwrite the joined_json in {self.json_dir}, do you want to do that? [Y/y] for YES, anything else for NO"
            )
            overwrite = confirm_overwrite in ["Y", "y"]
        elif not os.path.exists(self.json_dir):
            print(f"[{self.json_dir} doesn't exist, will build and save it ...]")
        else:
            print(f"[{self.json_dir} exists, will load it ...]")

        if os.path.exists(self.json_dir) and (not overwrite):
            logging.info(f"[ Loading {self.json_dir} ...]")
            time1 = time.time()
            self.joined_data = self._load_joined_json(self.json_dir)
            time2 = time.time()
            logging.info(f"[ Loading finished, took {(time2-time1)/60} mins ...]")
        else:
            if not os.path.exists(self.json_dir):
                if self.opt.get("debug"):
                    confirm_save_debug_data = input(
                        f"It seems you are in debug mode, and trying to build and"
                        f"save the data (you will only save {self.opt.get('debug_game_size')} games in {self.json_dir})? "
                        f"Are you sure? [Y/y] for YES, anything else for NO"
                    )
                    confirm_save_debug_data = confirm_save_debug_data in ["Y", "y"]
                    if not confirm_save_debug_data:
                        print("Aborting... Please remove --debug in the next run")
                        sys.exit(-1)
                os.makedirs(self.json_dir)

            if raw_order is None:
                raw_order = game_loading.load_sql_format(debug=self.opt.get("debug"))
                self.raw_order = game_loading.organize_game_dict_by_phase(raw_order)
                self.raw_msg = select_by_game_and_phase(load_message_data())
            else:
                self.raw_order = raw_order
                self.raw_msg = raw_msg

            self.joined_data = data_joining.join_order_and_msg(
                self.raw_order,
                self.raw_msg,
                self.include_message_from,
                with_special_token=self.with_special_token,
                special_tokens_map=self.special_tokens_map,
            )

    def _load_joined_json(self, dump_path):
        each_dump_contains = self.TOTAL_GAMES // self.DATA_SPLIT_INTO
        how_many_dumps_to_load = self.opt.get("debug_game_size") // each_dump_contains + 1

        dump_json_path = os.path.join(dump_path, "*.json")
        dump_json_paths = glob(dump_json_path)
        if self.opt.get("debug"):
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
