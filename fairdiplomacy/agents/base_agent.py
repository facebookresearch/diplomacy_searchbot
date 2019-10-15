from typing import List


class BaseAgent:
    def get_orders(self, game, power) -> List[str]:
        """Return a list of orders that should be taken based on the game state

        Arguments:
        - game: a diplomacy.Game object
        - power: str, one of {'AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY',
                              'ITALY', 'RUSSIA', 'TURKEY'}

        Returns a list of order strings, e.g.
            ["A TYR - TRI", "F ION - TUN", "A VEN S A TYR - TRI"]
        """
        raise NotImplementedError("Subclasses must implement")
