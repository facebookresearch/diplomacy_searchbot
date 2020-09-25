import unittest
import diplomacy
import json

from fairdiplomacy import pydipcc


class TestPydipcc(unittest.TestCase):
    def test_export_to_mila(self):
        game = pydipcc.Game()
        d = json.loads(game.to_json())
        py_game = diplomacy.utils.export.from_saved_game_format(d)

    def test_import_from_mila(self):
        py_game = diplomacy.Game()
        js = json.dumps(diplomacy.utils.export.to_saved_game_format(py_game))
        game = pydipcc.Game.from_json(js)
