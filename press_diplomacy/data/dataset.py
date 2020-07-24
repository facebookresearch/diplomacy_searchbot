import diplomacy
import joblib
import json
import logging
import os
import torch
import glob
import logging

from fairdiplomacy.agents import build_agent_from_cfg
from itertools import combinations, product
from fairdiplomacy.game import Game
from typing import Any, Dict, Union, List, Optional, Tuple
from fairdiplomacy.data.dataset import *
from parlai.utils.torch import padded_tensor
from fairdiplomacy.models.dipnet.train_sl import get_db_cache_args
from parlai.core.agents import create_agent_from_model_file
from glob import glob
from parlai_diplomacy.tasks.language_diplomacy.utils import select_by_game_and_phase
from fairdiplomacy.data.build_dataset import COUNTRY_ID_TO_POWER
from tqdm import tqdm
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)
logger.propagate = False


class PressDataset(Dataset, ABC):
    def __init__(
        self,
        *,
        parlai_agent_file: str,
        message_chunks: str,
        game_ids: List[int],
        data_dir: str,
        game_metadata: Dict[int, Any],
        debug_only_opening_phase=False,
        only_with_min_final_score=7,
        n_jobs=20,
        value_decay_alpha=1.0,
        cf_agent=None,
        n_cf_agent_samples=1,
        min_rating=None,
        exclude_n_holds=-1,
    ):
        super().__init__(
            game_ids=game_ids,
            data_dir=data_dir,
            game_metadata=game_metadata,
            debug_only_opening_phase=debug_only_opening_phase,
            only_with_min_final_score=only_with_min_final_score,
            n_jobs=n_jobs,
            value_decay_alpha=value_decay_alpha,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            min_rating=min_rating,
            exclude_n_holds=exclude_n_holds,
        )

        message_files = glob(message_chunks)
        if len(message_files):
            self.message_files = message_files
        else:
            raise FileNotFoundError("Message chunk glob is empty.")

        self.parlai_agent_file = parlai_agent_file
        self.dialogue_agent = create_agent_from_model_file(self.parlai_agent_file)
        self.messages = None

    @abstractmethod
    def _construct_message_dict(self, message_list):
        raise NotImplementedError("Subclasses must implement this. ")

    def _load_messages(self):
        """
        Loads message data
        """

        logging.info(f"Loading {len(self.message_files)} message chunks.")

        def load_message(message_json_path):
            try:
                with open(message_json_path, "r") as f:
                    database = json.load(f)
                    message_chunk = database[2]["data"]
            except FileNotFoundError:
                print(f"Error while loading message_chunk at {message_json_path}")
                return []
            else:
                return message_chunk

        messages_list = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(load_message)(file) for file in self.message_files
        )

        # TODO(apjacob): Perform caching to avoid loading each time
        # Flatten message chunks and constructs message_dict
        self.messages = self._construct_message_dict(
            [message for messages in messages_list for message in messages]
        )

        # Update game_ids
        self.game_ids = set(self.messages.keys()) & set(self.game_ids)

    def encode_message(self, message):
        # TODO(apjacob): Currently accessing a protected member. Fix?
        return self.dialogue_agent._vectorize_text(message, add_end=True)

    @abstractmethod
    def _encode_messages(self, game, game_id, phase_idx):
        raise NotImplementedError("Subclass must implement.")

    def _encode_phase(self, game, game_id: int, phase_idx: int, input_valid_power_idxs):
        data_fields = super(PressDataset, self)._encode_phase(
            game, game_id, phase_idx, input_valid_power_idxs
        )

        data_fields.update(self._encode_messages(game, game_id, phase_idx))

        return data_fields

    def preprocess(self):
        if self._preprocessed:
            logging.warning("Dataset has previously been preprocessed.")

        self._load_messages()

        assert not self.debug_only_opening_phase, "FIXME"

        logging.info(
            f"Building Press Dataset from {len(self.game_ids)} games, "
            f"only_with_min_final_score={self.only_with_min_final_score} "
            f"value_decay_alpha={self.value_decay_alpha} cf_agent={self.cf_agent}"
        )

        torch.set_num_threads(1)
        encoded_games = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._encode_game)(game_id) for game_id in self.game_ids
        )
        encoded_games = [
            g for g in encoded_games if g is not None
        ]  # remove "empty" games (e.g. json didn't exist)

        logging.info(f"Found data for {len(encoded_games)} / {len(self.game_ids)} games")

        encoded_games = [g for g in encoded_games if g["valid_power_idxs"][0].any()]
        logging.info(f"{len(encoded_games)} games had data for at least one power")

        game_idxs, phase_idxs, power_idxs, x_idxs = [], [], [], []
        x_idx = 0
        for game_idx, encoded_game in enumerate(encoded_games):
            for phase_idx, valid_power_idxs in enumerate(encoded_game["valid_power_idxs"]):
                assert valid_power_idxs.nelement() == len(POWERS), (
                    encoded_game["valid_power_idxs"].shape,
                    valid_power_idxs.shape,
                )
                for power_idx in valid_power_idxs.nonzero()[:, 0]:
                    game_idxs.append(game_idx)
                    phase_idxs.append(phase_idx)
                    power_idxs.append(power_idx)
                    x_idxs.append(x_idx)
                x_idx += 1

        self.game_idxs = torch.tensor(game_idxs, dtype=torch.long)
        self.phase_idxs = torch.tensor(phase_idxs, dtype=torch.long)
        self.power_idxs = torch.tensor(power_idxs, dtype=torch.long)
        self.x_idxs = torch.tensor(x_idxs, dtype=torch.long)

        # now collate the data into giant tensors!
        self.encoded_games = DataFields.cat(encoded_games)

        self.num_games = len(encoded_games)
        self.num_phases = len(self.encoded_games["x_board_state"]) if self.encoded_games else 0
        self.num_elements = len(self.x_idxs)

        for i, e in enumerate(self.encoded_games.values()):
            if isinstance(e, TensorList):
                assert len(e) == self.num_phases * len(POWERS) * MAX_SEQ_LEN
            else:
                assert len(e) == self.num_phases

        self._preprocessed = True


class ListenerDataset(PressDataset):
    def __init__(
        self,
        *,
        parlai_agent_file: str,
        message_chunks: str,
        game_ids: List[int],
        data_dir: str,
        game_metadata: Dict[int, Any],
        debug_only_opening_phase=False,
        only_with_min_final_score=7,
        n_jobs=20,
        value_decay_alpha=1.0,
        cf_agent=None,
        n_cf_agent_samples=1,
        min_rating=None,
        exclude_n_holds=-1,
    ):
        super().__init__(
            parlai_agent_file=parlai_agent_file,
            message_chunks=message_chunks,
            game_ids=game_ids,
            data_dir=data_dir,
            game_metadata=game_metadata,
            debug_only_opening_phase=debug_only_opening_phase,
            only_with_min_final_score=only_with_min_final_score,
            n_jobs=n_jobs,
            value_decay_alpha=value_decay_alpha,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            min_rating=min_rating,
            exclude_n_holds=exclude_n_holds,
        )

    def _construct_message_dict(self, message_list):
        message_dict = {}
        for message in tqdm(message_list):
            from_cnt_id = int(message["fromCountryID"])
            to_cnt_id = int(message["toCountryID"])

            if from_cnt_id not in COUNTRY_ID_TO_POWER or to_cnt_id not in COUNTRY_ID_TO_POWER:
                continue

            from_cnt = COUNTRY_ID_TO_POWER[from_cnt_id]
            to_cnt = COUNTRY_ID_TO_POWER[to_cnt_id]

            # select keys
            game_id = int(message["hashed_gameID"])
            phase_id = str(message["game_phase"])
            txt = str(message["message"])
            # set game dictionary
            message_dict.setdefault(game_id, {})
            # set turn dictionary
            message_dict[game_id].setdefault(phase_id, {})
            # isolate conversations between two players

            message_dict[game_id][phase_id].setdefault(to_cnt, [])

            # Only add listener messages
            message_dict[game_id][phase_id][to_cnt].append(txt)

        return message_dict

    def _encode_messages(self, game, game_id, phase_idx):
        assert game_id in self.messages

        phase_name = game.get_phase_name(phase_idx)
        power_tensors = []
        phase_message_dict = self.messages[game_id].get(phase_name, dict())
        # TODO(apjacob): Verify country idx mapping

        for power_idx, power in enumerate(POWERS):
            power_message_list = phase_message_dict.get(power, [])
            power_message = " ".join(power_message_list)
            power_tensor = self.encode_message(power_message)
            power_tensors.append(power_tensor)
        padded_tensors, _ = padded_tensor(power_tensors, pad_idx=-1)
        input_message = TensorList.from_padded(padded_tensors, padding_value=-1)

        return DataFields(x_input_message=input_message)


def build_press_db_cache_from_cfg(cfg):
    game_metadata, min_rating, train_game_ids, val_game_ids = get_db_cache_args(cfg)

    train_dataset = ListenerDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        message_chunks=cfg.message_chunks,
        game_ids=train_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=1,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )
    train_dataset.preprocess()

    val_dataset = ListenerDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        message_chunks=cfg.message_chunks,
        game_ids=val_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=1,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )

    val_dataset.preprocess()

    logger.info(f"Saving datasets to {cfg.data_cache}")
    torch.save((train_dataset, val_dataset), cfg.data_cache)
