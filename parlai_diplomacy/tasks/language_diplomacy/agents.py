#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher, ChunkTeacher

import parlai_diplomacy.tasks.language_diplomacy.utils as utls
from parlai.utils import misc

from glob import glob
import json
import os
import sys
import random
from typing import List, Tuple
from tqdm import tqdm
from typing import List, Tuple
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from parlai.core.dict import DictionaryAgent


TRAIN_SPLIT = 800_000  # (total game turns is 868_046)


@register_teacher("dialogue")
class DialogueTeacher(FixedDialogTeacher):
    """
    Plain diplomacy dialogue teacher.

    Does not use any game moves.

    Example use:
    ```
    python parlai_diplomacy/scripts/test_train_script.py -mf /tmp/test_stuff -m transformer/generator -t diplomacy_test
    ```
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = utls.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.min_turns = opt["min_turns"]
        self.is_train = utls.is_training(opt["datatype"])

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            self.num_exs = sum(len(conv) for conv in self.data)
        else:
            self.data = shared["data"]
            self.num_exs = shared["num_exs"]
        super().__init__(opt, shared)
        self.reset()

    def _construct_msg(self, conv, idx):
        msg = conv[idx]
        new_msg = {
            "text": msg["input"],
            "labels": [msg["response"]],
            "episode_done": idx >= (len(conv) - 1),
        }
        for k, v in msg.items():
            if k not in ["input", "response"]:
                new_msg[k] = v

        return new_msg

    def _setup_data(self, opt):
        # TODO: set in to train/test/valid split (confirm with group)
        # Load data iterator
        self.iterator = utls.DataIterator()
        # Run through all turns to get a list of conversations
        convs = []
        print(f"Total dataset Diplomacy turns: {len(self.iterator)}")
        tot_turns = 0
        for i, turn in enumerate(self.iterator):
            if "train" in opt["datatype"]:
                if i >= TRAIN_SPLIT:
                    break
            else:
                # test/valid are the same for now
                # TODO: change
                if i < TRAIN_SPLIT:
                    continue

            tot_turns += 1
            for conv in turn:
                if len(conv) >= self.min_turns:
                    convs.append(conv)

        dt = opt["datatype"].split(":")[0]
        print(f"Loaded {tot_turns} Diplomacy turns for datasplit {dt}")

        if self.is_train:
            random.shuffle(convs)

        return convs

    def get(self, episode_idx, entry_idx=0):
        ex = self._construct_msg(self.data[episode_idx], entry_idx)
        return Message(ex)

    def num_examples(self):
        # fix this
        return self.num_exs

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared["data"] = self.data
        shared["num_exs"] = self.num_exs
        return shared


@register_teacher("dialogue_chunk")
class DialogueChunkTeacher(ChunkTeacher):
    """
    Dialogue teacher but split into chunks for faster loading

    Example usage:
    ```
    parlai display_data -t internal:diplomacy:dialogue_chunk -dt train:stream
    ```
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = utls.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        if shared is None:
            # set map
            self.opt = opt
            self._set_chunk_idx_to_file()
        else:
            self.chunk_idx_to_file = shared["chunk_idx_to_file"]
        self.min_turns = opt["min_turns"]
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return utls.CHUNK_DIALOGUE_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt["datatype"]
        if "train" in datatype:
            if self.min_turns == 1:
                return 5140030, 9836290
            elif self.min_turns == 2:
                return 2244518, 6940778
            elif self.min_turns == 3:
                return 1031766, 4515274
            elif self.min_turns == 4:
                return 520942, 2982802
        if "valid" in datatype:
            if self.min_turns == 1:
                return 433562, 830248
            elif self.min_turns == 2:
                return 189932, 586618
            elif self.min_turns == 3:
                return 88058, 382870
            elif self.min_turns == 4:
                return 44384, 251848

        raise RuntimeError(
            f"Min turns {self.min_turns} for datatype {datatype} currently not supported"
        )

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        file_lst = sorted(glob(os.path.join(folder, "games_*")))
        self.chunk_idx_to_file = {i: x for i, x in enumerate(file_lst)}

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt["datatype"]
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if "train" in datatype:
            return all_chunk_idxs[:-20]
        elif "valid" in datatype:
            return all_chunk_idxs[-20:-10]
        else:
            return all_chunk_idxs[-10:]

    def load_from_chunk(self, chunk_idx: int) -> List[Tuple[str, str]]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])

        with open(chunk_path, "r") as f:
            data = json.load(f)

        iterator = utls.DataIterator(data)

        convs = []
        for _, turn in enumerate(iterator):
            for conv in turn:
                if len(conv) >= self.min_turns:
                    convs.append(conv)

        return convs

    def create_message(self, queue_output: Tuple[str, ...], entry_idx=0) -> "Message":
        """
        Given the tuple output of the queue, return an act.
        """
        conv = queue_output
        msg = conv[entry_idx]
        new_msg = {
            "text": msg["input"],
            "labels": [msg["response"]],
            "episode_done": entry_idx >= (len(conv) - 1),
        }
        for k, v in msg.items():
            if k not in ["input", "response"]:
                new_msg[k] = v

        return new_msg

    def share(self):
        shared = super().share()
        shared["chunk_idx_to_file"] = self.chunk_idx_to_file
        return shared


@register_teacher("base_order")
class BaseOrderTeacher(FixedDialogTeacher):
    """
    Plain diplomacy (message-order) teacher.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        # add common arguments
        argparser = utls.add_common_args(argparser)

        # add teacher-specific arguments
        argparser.add_argument(
            "--include-message-from",
            type=str,
            default="all_msg",
            choices={"speaker_msg_only", "partner_msg_only", "all_msg"},
            help="whose messages to include (speaker-only, listener-only, or all messages)",
        )
        argparser.add_argument(
            "--overwrite-joined-json", action="store_true", help="Overwrite the joined json",
        )
        argparser.add_argument(
            "--with-seq-len-stat", action="store_true", help="analyze the sequence length",
        )
        argparser.add_argument(
            "--load-splitted-valid",
            action="store_true",
            help="the validation set is splitted, do we want to load the splitted set or build the validation set from scratch",
        )
        argparser.add_argument(
            "--test-file-id", type=str, help="test file chunk id",
        )
        argparser.add_argument(
            "--with-special-token",
            action="store_true",
            help="with the added special tokens with underscores",
        )
        argparser.add_argument(
            "--train-valid-split-by",
            type=str,
            default="game",
            choices={"game", "data_point_with_msg",},
            help="how to split the train/validation set, by games (used for comparison with fairdip) or \
                data_point_with_msg (cannot be used for fairdip, used only to see if the msg data is helping)",
        )
        argparser.add_argument(
            "--debug",
            action="store_true",
            help="debug mode to load fewer games (--debug-game-size)",
        )
        argparser.add_argument(
            "--debug-game-size", type=int, default=500, help="how many games to use in debug mode",
        )
        argparser.add_argument(
            "--which-tokenizer-for-seq-stat",
            type=str,
            choices={"bart", "transformer",},
            help="which tokenizer to use for sequence statistics, this should be used together with --with-seq-len-stat",
        )
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.dt = opt["datatype"].split(":")[0]
        self.min_turns = opt["min_turns"]
        self.include_message_from = opt["include_message_from"]
        self.overwrite_joined_json = opt["overwrite_joined_json"]
        self.with_special_token = opt["with_special_token"]

        # calculate the sequence length stat
        self._calculate_seq_len_stat(stage="init")

        self.valid_json_dir = utls.VALID_JSON_PATH
        self.is_train = utls.is_training(opt["datatype"])

        if shared is None:
            # set map
            if opt["load_splitted_valid"]:
                self.data = self._load_valid_data(opt)
            else:
                self.data = self._setup_data(opt)
            self.num_exs = len(self.data)
        else:
            self.data = shared["data"]
            self.num_exs = shared["num_exs"]

        super().__init__(opt, shared)
        self.reset()

    def _print_example_data(self):
        nl = "\n"
        examples = [
            f"text_{i}: {self._construct_msg(self.data[i], 0)['text']}{nl}labels_{i}: "
            f"{self._construct_msg(self.data[i], 0)['labels']}{nl}game_{i}: {self._construct_msg(self.data[i], 0)['game']}{nl}"
            f"game_{i}: {self._construct_msg(self.data[i], 0)['game_phase']}{nl}{nl}"
            for i in range(5)
        ]
        example_data_text = f"example data{nl} {nl.join(examples)}"
        misc.warn_once(example_data_text)

    def _replace_with_special_token(self, msg):
        # should be used when the saved joined_json is 'A BAL',
        # 'A BAL' --> '__A__ __BAL__'
        # TODO depreciate this once we've saved the joins_json with __
        if self.with_special_token:
            msg["order"] = utls.replace_with_special_token(msg["order"])
            msg["state"] = utls.replace_with_special_token(msg["state"])

        return msg

    def _get_train_val_split_num(self, TRAIN_VAL_SPLIT_PERCENT):
        if self.opt["train_valid_split_by"] == "game":
            print(f"validation set are on game-level, should be used for fairdip comparison!")
            print(f"Total dataset Diplomacy (message+order) games: {self.iterator.num_games}")
            TRAIN_SPLIT_NUM = int(self.iterator.num_games * TRAIN_VAL_SPLIT_PERCENT)
            VALID_SPLIT_NUM = self.iterator.num_games - TRAIN_SPLIT_NUM
            print(f"Total dataset Diplomacy (message+order) games for training: {TRAIN_SPLIT_NUM}")
            print(
                f"Total dataset Diplomacy (message+order) games for validation: {VALID_SPLIT_NUM}"
            )
        elif self.opt["train_valid_split_by"] == "data_point_with_msg":
            print(
                f"validation set contains pairs with non-empty message only! Shouldn't be used for fairdip comparison"
            )
            print(f"Total dataset Diplomacy (message+order) pairs: {self.iterator.num_pairs}")
            TRAIN_SPLIT_NUM = int(self.iterator.num_pairs * TRAIN_VAL_SPLIT_PERCENT)
            VALID_SPLIT_NUM = self.iterator.num_pairs - TRAIN_SPLIT_NUM
            print(f"Total dataset Diplomacy (message+order) pairs for training: {TRAIN_SPLIT_NUM}")
            print(
                f"Total dataset Diplomacy (message+order) pairs for validation: {VALID_SPLIT_NUM}"
            )

        return TRAIN_SPLIT_NUM, VALID_SPLIT_NUM

    def _calculate_seq_len_stat(
        self,
        stage,
        pair=None,
        tot_pairs=None,
        tot_phases=None,
        tot_games=None,
        tot_unique_phases=None,
    ):
        # the stat calculation can only be done on valid set
        if not (self.opt["with_seq_len_stat"] and "valid" in self.opt["datatype"]):
            return

        if stage == "init":
            # safety check
            if self.opt["with_seq_len_stat"]:
                if "train" in self.opt["datatype"]:
                    raise ValueError(
                        "Looks like you want to check the sequence length statistics for the training set, "
                        "which will take a long time, so please change to the validation set by setting `--datatype valid`"
                    )
                if self.opt["which_tokenizer_for_seq_stat"] is None:
                    raise ValueError(
                        "It seems you want to get the sequence length statistics "
                        "Please set the tokenizer you want to use by --which-tokenizer-for-seq-stat. "
                        "--help for more info."
                    )
                if not self.opt["debug"]:
                    confirm_whole_set = input(
                        f"Are you sure you want to get the whole dataset's sequence length statistics, instead of the top 500 games by --debug? [Y/y] for YES, anything else for NO"
                    )
                    confirm_whole_set = confirm_whole_set in ["Y", "y"]
                    if not confirm_whole_set:
                        print("Aborting... Please set --debug in the next run")
                        sys.exit(-1)

            # build tokenizer
            if self.opt["which_tokenizer_for_seq_stat"] == "bart":
                with open(utls.BART_OPT_PATH, "r") as fh:
                    TOKENIZER_OPT = json.load(fh)
            else:
                with open(utls.TRANSFORMER_OPT_PATH, "r") as fh:
                    TOKENIZER_OPT = json.load(fh)
            self.tokenizer_for_stat = DictionaryAgent(TOKENIZER_OPT)

            # replace special tokens
            if self.with_special_token:
                self.special_tokens, self.special_tokens_map = utls.load_special_tokens()
                self.tokenizer_for_stat.add_additional_special_tokens(self.special_tokens)

            # different truncate size
            self.text_truncate = [256, 512, 1024, 2048]
            self.label_truncate = [256, 512, 1024, 2048]

            # book keeping
            self.input_seq_state_lens = []
            self.input_seq_msg_lens = []
            self.output_seq_lens = []
            self.truncated_at_state = [0] * len(self.text_truncate)
            self.truncated_at_msg = [0] * len(self.text_truncate)
            self.truncated_at_state_msg = [0] * len(self.text_truncate)
            self.truncated_at_order = [0] * len(self.label_truncate)
            self.no_msg = 0

        elif stage == "calculate":
            state_str, msg_str, order = pair["state"], pair["message"], pair["order"]
            if self.with_special_token:
                state_str = utls.replace_with_special_token(state_str)
                order = utls.replace_with_special_token(order)
            # print the first order to see if the sequence is expected
            if len(self.input_seq_state_lens) == 0:
                print(f"example order: {order}")
                print(f"tokenized order: {self.tokenizer_for_stat.tokenize(order)}")

            # calculate sequence length
            input_state_len = len(self.tokenizer_for_stat.tokenize(state_str))
            input_msg_len = len(self.tokenizer_for_stat.tokenize(msg_str))
            output_seq_len = len(self.tokenizer_for_stat.tokenize(order))
            self.input_seq_state_lens.append(input_state_len)
            self.input_seq_msg_lens.append(input_msg_len)
            self.output_seq_lens.append(output_seq_len)

            # calculate truncated sequences
            if "NoMsg" in pair["data_status"]:
                self.no_msg += 1
            for idx in range(len(self.text_truncate)):
                if (input_state_len) > self.text_truncate[idx]:
                    self.truncated_at_state[idx] += 1
                if (input_msg_len) > self.text_truncate[idx]:
                    self.truncated_at_msg[idx] += 1
                if (input_state_len + input_msg_len) > self.text_truncate[idx]:
                    self.truncated_at_state_msg[idx] += 1
                if (output_seq_len) > self.label_truncate[idx]:
                    self.truncated_at_order[idx] += 1

        elif stage == "summary":
            print(f"Loaded {tot_phases} Diplomacy phases for datasplit {self.dt}")
            print(f"Loaded {tot_games} Diplomacy games for datasplit {self.dt}")
            print(f"Loaded {tot_unique_phases} unique Diplomacy phases for datasplit {self.dt}")
            print(f"Loaded {tot_pairs} unique Diplomacy pairs for datasplit {self.dt}")

            print(
                f"mean_input_seq_state_len: {np.mean(self.input_seq_state_lens)} ({np.sum(self.input_seq_state_lens)}/{len(self.input_seq_state_lens)}) for datasplit {self.dt}"
            )
            print(
                f"mean_input_seq_msg_len: {np.mean(self.input_seq_msg_lens)} ({np.sum(self.input_seq_msg_lens)}/{len(self.input_seq_msg_lens)}) for datasplit {self.dt}"
            )
            print(
                f"mean_output_seq_len: {np.mean(self.output_seq_lens)} ({np.sum(self.output_seq_lens)}/{len(self.output_seq_lens)}) for datasplit {self.dt}"
            )
            print(
                f"90%_input_seq_state_len: {np.quantile(self.input_seq_state_lens, 0.9)} ({np.sum(self.input_seq_state_lens)}/{len(self.input_seq_state_lens)}) for datasplit {self.dt}"
            )
            print(
                f"90%_input_seq_msg_len: {np.quantile(self.input_seq_msg_lens, 0.9)} ({np.sum(self.input_seq_msg_lens)}/{len(self.input_seq_msg_lens)}) for datasplit {self.dt}"
            )
            print(
                f"90%_output_seq_len: {np.quantile(self.output_seq_lens, 0.9)} ({np.sum(self.output_seq_lens)}/{len(self.output_seq_lens)}) for datasplit {self.dt}"
            )
            print()
            print(
                f"no message: {self.no_msg/tot_pairs} ({self.no_msg}/{tot_pairs}) for datasplit {self.dt} for {self.text_truncate}"
            )
            for i in range(len(self.text_truncate)):
                print(f"for truncate size {self.text_truncate[i]}")
                print(
                    f"truncated at state: {self.truncated_at_state[i]/tot_pairs} ({self.truncated_at_state[i]}/{tot_pairs}) for datasplit {self.dt}"
                )
                print(
                    f"truncated at msg: {self.truncated_at_msg[i]/tot_pairs} ({self.truncated_at_msg[i]}/{tot_pairs}) for datasplit {self.dt}"
                )
                print(
                    f"truncated at state_msg: {self.truncated_at_state_msg[i]/tot_pairs} ({self.truncated_at_state_msg[i]}/{tot_pairs}) for datasplit {self.dt}"
                )
                print(
                    f"truncated at order: {self.truncated_at_order[i]/tot_pairs} ({self.truncated_at_order[i]}/{tot_pairs}) for datasplit {self.dt}"
                )
                print()

    def _construct_common_msg_fields(self, msg):
        # speaker token
        game_phase = msg["game_phase"]
        speaker_token = (
            f"{game_phase} {msg['speaker'].capitalize()}:"  # looks like "1901M France:"
        )

        # new_msg
        new_msg = {
            "labels": [msg["order"]],
            "episode_done": True,
        }

        # other fields
        for k, v in msg.items():
            new_msg[k] = v

        return speaker_token, new_msg

    def _construct_msg(self, pair, idx):
        # individual teacher should implement this
        raise NotImplementedError("Please implement _construct_msg for individual teachers!")

    def _setup_data(self, opt):
        # TODO: set in to train/test/valid split (confirm with group)
        # Load data iterator
        self.iterator = utls.MessageOrderDataIterator(opt)

        # get train/valid split number
        TRAIN_VAL_SPLIT_PERCENT = 0.95
        TRAIN_SPLIT_NUM, VALID_SPLIT_NUM = self._get_train_val_split_num(TRAIN_VAL_SPLIT_PERCENT)

        # initialize game and phase stats
        tot_phases, tot_pairs, tot_valid_pairs = 0, 0, 0
        game_ids, game_phase_ids = [], []
        # Run through all phases to get a list of (message-order) pairs
        pairs = []
        for i, phase in enumerate(tqdm(self.iterator)):
            if opt["train_valid_split_by"] == "game":
                if "train" in opt["datatype"]:
                    if self.iterator.game_idx >= TRAIN_SPLIT_NUM:
                        break
                else:
                    # test/valid are the same for now
                    # the last games are valid
                    # TODO: change
                    if self.iterator.game_idx < TRAIN_SPLIT_NUM:
                        continue
            elif opt["train_valid_split_by"] == "data_point_with_msg":
                if "train" not in opt["datatype"]:
                    # the first games with msg are valid
                    if tot_valid_pairs >= VALID_SPLIT_NUM:
                        break

            # book keeping
            tot_phases += 1
            game_ids.append(self.iterator.game_idx)
            game_phase_ids.append(f"{self.iterator.game_idx}-{self.iterator.phase_idx}")
            for pair in phase:
                # train/valid split condition, the validation dataset contains datapoints with msg only
                # the first games with msg are valid
                if opt["train_valid_split_by"] == "data_point_with_msg":
                    if "train" in opt["datatype"]:
                        # the top VALID_SPLIT_NUM "Game_Phase_Msg" will be validation
                        if (
                            pair["data_status"] == "Game_Phase_Msg"
                            and tot_valid_pairs < VALID_SPLIT_NUM
                        ):
                            tot_valid_pairs += 1
                            continue
                        else:
                            pass
                    else:
                        if (
                            pair["data_status"] == "Game_Phase_Msg"
                            and tot_valid_pairs < VALID_SPLIT_NUM
                        ):
                            tot_valid_pairs += 1
                            pass
                        else:
                            if pair["data_status"] != "Game_Phase_Msg":
                                continue
                            elif tot_valid_pairs >= VALID_SPLIT_NUM:
                                break

                # include this pair in this datatype
                pairs.append(pair)

                # book keeping
                tot_pairs += 1
                self._calculate_seq_len_stat(stage="calculate", pair=pair)

        # get data stat
        self._calculate_seq_len_stat(
            stage="summary",
            tot_pairs=tot_pairs,
            tot_phases=tot_phases,
            tot_games=len(set(game_ids)),
            tot_unique_phases=len(set(game_phase_ids)),
        )

        if self.is_train:
            random.shuffle(pairs)

        # save the json for validation evaluation
        if "valid" in opt["datatype"]:
            if not os.path.exists(self.valid_json_dir):
                with open(self.valid_json_dir, "w") as fh:
                    json.dump(pairs, fh)

        return pairs

    def _load_valid_data(self, opt):
        print(f"load valid_split_{opt['test_file_id']}.json")
        with open(
            f"{utls.TEST_SPLIT_JSON_PATH}/valid_split_{opt['test_file_id']}.json", "r"
        ) as fh:
            pairs = json.load(fh)
        return pairs

    def get(self, episode_idx, entry_idx=0):
        ex = self._construct_msg(self.data[episode_idx], entry_idx)
        return Message(ex)

    def num_examples(self):
        # fix this
        return self.num_exs

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared["data"] = self.data
        shared["num_exs"] = self.num_exs
        return shared


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


@register_teacher("state_order")
class StateOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "state_order"
        new_msg["text"] = f"{new_msg['state']} {speaker_token}"

        return new_msg


@register_teacher("message_order")
class MessageOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "message_order"
        new_msg["text"] = f"{new_msg['message']} {speaker_token}"

        return new_msg


@register_teacher("order_history_order")
class OrderHistoryOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "order_history_order"
        new_msg["text"] = f"{new_msg['order_history']} {speaker_token}"

        return new_msg


@register_teacher("speaker_token_order")
class SpeakerTokenOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "speaker_token_order"
        new_msg["text"] = f"{speaker_token}"

        return new_msg


@register_teacher("dummy_token_order")
class DummyTokenOrderTeacher(BaseOrderTeacher):
    def _construct_msg(self, pair, idx):
        speaker_token, new_msg = self._construct_common_msg_fields(pair)

        assert self.opt["task"] == "dummy_token_order"
        new_msg["text"] = f"UNK"

        return new_msg


class DefaultTeacher(DialogueTeacher):
    pass
