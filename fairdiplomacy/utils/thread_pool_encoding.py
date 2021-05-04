# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Optional
import numpy as np
import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN

KEYS_STATE_ONLY = [
    "x_board_state",
    "x_prev_state",
    "x_prev_orders",
    "x_season",
    "x_in_adj_phase",
    "x_build_numbers",
]

KEYS_ALL = KEYS_STATE_ONLY + ["x_loc_idxs", "x_possible_actions", "x_max_seq_len"]


class FeatureEncoder:

    nothread_pool_singleton: Optional[pydipcc.ThreadPool] = None

    def __init__(self, num_threads: int = 0):
        if num_threads <= 0:
            self.thread_pool = self._get_nothread_pool()
        else:
            self.thread_pool = pydipcc.ThreadPool(
                num_threads, ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN
            )

    @classmethod
    def _get_nothread_pool(cls) -> pydipcc.ThreadPool:
        if cls.nothread_pool_singleton is None:
            cls.nothread_pool_singleton = pydipcc.ThreadPool(
                0, ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN
            )
        return cls.nothread_pool_singleton

    def encode_inputs(self, games: Sequence[pydipcc.Game]) -> DataFields:
        return DataFields(self.thread_pool.encode_inputs_multi(games))

    def encode_inputs_state_only(self, games: Sequence[pydipcc.Game]) -> DataFields:
        return DataFields(self.thread_pool.encode_inputs_state_only_multi(games))

    def decode_order_idxs(self, order_idxs):
        return self.thread_pool.decode_order_idxs(order_idxs)

    def process_multi(self, games: Sequence[pydipcc.Game]) -> None:
        self.thread_pool.process_multi(games)
