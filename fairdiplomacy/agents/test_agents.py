# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from fairdiplomacy.agents.base_search_agent import n_move_phases_later


class TestNMovePhaseLater(unittest.TestCase):
    def test_0_from_fall(self):
        self.assertEqual(n_move_phases_later("S1902M", 0), "S1902M")

    def test_0_from_winter(self):
        self.assertEqual(n_move_phases_later("W1901A", 0), "W1901A")
