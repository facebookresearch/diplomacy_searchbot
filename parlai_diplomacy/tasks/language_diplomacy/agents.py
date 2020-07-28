#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher, ChunkTeacher

import parlai_diplomacy.tasks.language_diplomacy.utils as utls
import parlai_diplomacy.tasks.language_diplomacy.config as cfg

from glob import glob
import json
import os
import random
from typing import List, Tuple
from tqdm import tqdm


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


@register_teacher("message_order")
class MessageOrderTeacher(FixedDialogTeacher):
    """
    Plain diplomacy (message-order) teacher.
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
            self.num_exs = len(self.data)
        else:
            self.data = shared["data"]
            self.num_exs = shared["num_exs"]
        super().__init__(opt, shared)
        self.reset()

    def _construct_msg(self, pair, idx):
        msg = pair
        if cfg.INPUT_SEQ == cfg.STATE_AND_MSG:
            new_msg = {
                "text": msg["state_and_message"],
                "labels": [msg["order"]],
                "episode_done": True,
            }
        elif cfg.INPUT_SEQ == cfg.STATE_ONLY:
            new_msg = {
                "text": msg["state_and_message"].split("[EO_STATE]")[0].strip(),
                "labels": [msg["order"]],
                "episode_done": True,
            }
        elif cfg.INPUT_SEQ == cfg.MSG_ONLY:
            new_msg = {
                "text": msg["state_and_message"].split("[EO_STATE]")[1].strip(),
                "labels": [msg["order"]],
                "episode_done": True,
            }
        else:
            raise ValueError(f"Wrong INPUT_SEQ_content: {cfg.INPUT_SEQ}!")

        for k, v in msg.items():
            if k not in ["state_and_message", "order"]:
                new_msg[k] = v

        return new_msg

    def _setup_data(self, opt):
        # TODO: set in to train/test/valid split (confirm with group)
        # Load data iterator
        self.iterator = utls.MessageOrderDataIterator()
        # Run through all phases to get a list of (message-order) pairs
        pairs = []
        print(f"Total dataset Diplomacy (message+order) phases: {len(self.iterator)}")
        TRAIN_SPLIT_NUM = int(len(self.iterator) * 0.8)
        tot_phases = 0
        tot_input_seq_state_len = 0
        tot_input_seq_msg_len = 0
        tot_output_seq_len = 0
        tot_pair = 0
        for i, phase in enumerate(tqdm(self.iterator)):
            if "train" in opt["datatype"]:
                if i >= TRAIN_SPLIT_NUM:
                    break
            else:
                # test/valid are the same for now
                # TODO: change
                if i < TRAIN_SPLIT_NUM:
                    continue

            tot_phases += 1
            for pair in phase:
                # if len(pair) >= self.min_turns:
                state_str, msg_str = pair["state_and_message"].split("[EO_STATE]")
                tot_input_seq_state_len += len(state_str.split())
                tot_input_seq_msg_len += len(msg_str.split())
                tot_output_seq_len += len(pair["order"].split())
                tot_pair += 1
                pairs.append(pair)

        dt = opt["datatype"].split(":")[0]
        print(f"Loaded {tot_phases} Diplomacy phases for datasplit {dt}")
        print(
            f"mean_input_seq_state_len: {tot_input_seq_state_len/tot_pair} ({tot_input_seq_state_len}/{tot_pair}) for datasplit {dt}"
        )
        print(
            f"mean_input_seq_msg_len: {tot_input_seq_msg_len/tot_pair} ({tot_input_seq_msg_len}/{tot_pair}) for datasplit {dt}"
        )
        print(
            f"mean_output_seq_len: {tot_output_seq_len/tot_pair} ({tot_output_seq_len}/{tot_pair}) for datasplit {dt}"
        )

        if self.is_train:
            random.shuffle(pairs)

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


class DefaultTeacher(DialogueTeacher):
    pass
