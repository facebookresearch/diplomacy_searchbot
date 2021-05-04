# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Sequence


class BaseAgent:
    def get_orders(self, game, power) -> List[str]:
        """Return a list of orders that should be taken based on the game state

        Arguments:
        - game: a Game object
        - power: str, one of {'AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY',
                              'ITALY', 'RUSSIA', 'TURKEY'}

        Returns a list of order strings, e.g.
            ["A TYR - TRI", "F ION - TUN", "A VEN S A TYR - TRI"]
        """
        raise NotImplementedError("Subclasses must implement")

    def get_orders_many_powers(self, game, powers: Sequence[str]) -> Dict[str, List[str]]:
        """Return a set of orders that should be taken based on the game state per power

        Arguments:
        - game: a Game object
        - powers: a set of powers that need orders

        Returns a dict of orders for each power.
        """
        return {p: self.get_orders(game, p) for p in powers}
