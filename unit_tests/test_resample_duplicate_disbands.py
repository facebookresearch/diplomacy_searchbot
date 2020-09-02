import os
import unittest
import torch

from fairdiplomacy.agents.dipnet_agent import resample_duplicate_disbands_inplace

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestResampleDuplicateDisbands(unittest.TestCase):
    def test_2020_09_01(self):
        X = torch.load(
            UNIT_TEST_DIR + "/data/resample_duplicate_disbands_inplace.debug.2020.09.01.pt"
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
