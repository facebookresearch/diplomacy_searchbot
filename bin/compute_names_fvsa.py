import pprint
import sqlite3

import diplomacy

game = diplomacy.Game()
MAP = game.map

MISSPELLINGS = {
    "SKAGERRAK": "SKAGERRACK",
    "HELGOLAND BIGHT": "HELIGOLAND BIGHT",
    "GULF OF LYON": "GULF OF LYONS",
    "BULGARIA (EAST COAST)": "BULGARIA (NORTH COAST)",
}

# map from full name to short code, e.g. "Trieste" -> "TRI"
FULL_TO_SHORT = {}
for full_name in MAP.loc_name.keys():
    x = full_name.upper().replace(".", "")
    FULL_TO_SHORT[full_name] = MAP.loc_name.get(x) or MAP.loc_name[MISSPELLINGS[x]]
    if x in MISSPELLINGS:
        FULL_TO_SHORT[MISSPELLINGS[x]] = FULL_TO_SHORT[x]


db = sqlite3.connect("/checkpoint/fairdiplomacy/facebook_notext.sqlite3")

TERR_ID_TO_LOC = {
    terr_id: FULL_TO_SHORT[full_name.upper().replace(".", "")]
    for (terr_id, full_name) in db.execute(
        "SELECT id, name FROM redacted_territories WHERE mapID=15"
    )
}


pprint.pprint(TERR_ID_TO_LOC)
