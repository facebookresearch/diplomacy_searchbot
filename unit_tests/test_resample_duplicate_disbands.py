# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import torch

from fairdiplomacy.agents.model_wrapper import resample_duplicate_disbands_inplace

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestResampleDuplicateDisbands(unittest.TestCase):
    def test_2020_09_01(self):
        X = torch.load(
            UNIT_TEST_DIR + "/data/resample_duplicate_disbands_inplace.debug.2020.09.01.pt",
            map_location="cpu",
        )
        resample_duplicate_disbands_inplace(
            X["order_idxs"],
            X["sampled_idxs"],
            X["logits"],
            X["x_possible_actions"],
            X["x_in_adj_phase"],
        )

        # for these inputs, there are only five valid disbands for England (1),
        # so check that all sampled idxs are valid (< 5)
        self.assertTrue((X["sampled_idxs"][0, 1] < 5).all())
