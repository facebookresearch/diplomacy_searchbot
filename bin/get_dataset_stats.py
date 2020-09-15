#!/usr/bin/env python

import argparse
import glob
import os
import json
import numpy as np
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor

from fairdiplomacy.utils.game_scoring import (
    compute_game_sos_from_state,
    compute_game_dss_from_state,
)


def get_scores(game_json) -> Optional[List[float]]:
    try:
        with open(game_json) as f:
            j = json.load(f)
        state = j["phases"][-1]["state"]
        sos = compute_game_sos_from_state(state)
        dss = compute_game_dss_from_state(state)
        return (sos, dss)
    except:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("games_dir")
args = parser.parse_args()

game_jsons = glob.glob(os.path.join(args.games_dir, "game_*.json"))

pool = ProcessPoolExecutor()


success, error = 0, 0
sos_tot = np.zeros(7, dtype=np.double)
dss_tot = np.zeros(7, dtype=np.double)
for r in pool.map(get_scores, game_jsons):
    if r is None:
        error += 1
    else:
        success += 1
        sos, dss = r
        sos_tot += sos
        dss_tot += dss

assert success + error == len(game_jsons)

print(f"success: {success} / {success + error}")
print("SOS:", sos_tot / success)
print("DSS:", dss_tot / success)
