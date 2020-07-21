import diplomacy
import joblib
import json
import logging
import os
import torch
import glob
import logging

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.data.dataset import Dataset

from itertools import combinations, product
from typing import Any, Dict, Union, List, Optional, Tuple

from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import SEASONS, POWERS, MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.dipnet.encoding import board_state_to_np, prev_orders_to_np
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state
from fairdiplomacy.utils.tensorlist import TensorList
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.data.dataset import (
    Dataset,
    ORDER_VOCABULARY_TO_IDX,
    ORDER_VOCABULARY,
    LOC_IDX,
    MAX_VALID_LEN,
)
from fairdiplomacy.models.dipnet.train_sl import get_db_cache_args
from parlai.core.agents import create_agent_from_model_file


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)
logger.propagate = False


class PressDataset(Dataset):
    def __init__(self, *, parlai_agent_file, **kwargs):
        self.parlai_agent_file = parlai_agent_file
        self.dialogue_agent = create_agent_from_model_file(
            self.dialogue_agent_file, {"skip_generation": False}
        )

        super().__init__(**kwargs)
        pass


def build_press_db_cache_from_cfg(cfg):
    game_metadata, min_rating, train_game_ids, val_game_ids = get_db_cache_args(cfg)

    train_dataset = PressDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        game_ids=train_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=cfg.num_dataloader_workers,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )

    val_dataset = PressDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        game_ids=val_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=cfg.num_dataloader_workers,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )

    logger.info(f"Saving datasets to {cfg.data_cache}")
    torch.save((train_dataset, val_dataset), cfg.data_cache)
