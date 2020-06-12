from typing import List
import json
import logging
import pathlib
import random
import time

import numpy as np
import torch

from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents import build_agent_from_cfg


def load_games(game_jsons) -> List[Game]:
    if not game_jsons:
        return [Game()]

    games = []
    for path in game_jsons:
        with open(path) as f:
            game_json = json.load(f)
        games.append(Game.from_saved_game_format(game_json))
    return games


def run(cfg: "conf.conf_pb2.BenchmarkAgent"):

    logger = logging.getLogger("timing")
    logger.setLevel(logging.DEBUG)
    timing_outpath = pathlib.Path(".").resolve() / "timing.log"
    logger.info("Will write timing logs to %s", timing_outpath)
    logger.addHandler(logging.FileHandler(timing_outpath))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    agent = build_agent_from_cfg(cfg.agent)
    games = load_games(cfg.game_jsons)

    logger.info("Warmup")
    agent.get_orders(games[0], POWERS[0])

    logger.info("Running benchmark")
    times = []
    for _ in range(cfg.repeats):
        times.append([])
        for game in games:
            times[-1].append([])
            for power in POWERS:
                start = time.time()
                agent.get_orders(game, power)
                times[-1][-1].append(time.time() - start)
    times = np.array(times)
    logger.info("Total time: %s", times.sum())
    per_repeat = times.sum(-1).sum(-1)
    logger.info("Total time per repeat: %s +- %s", per_repeat.mean(), per_repeat.std())
    per_call = times.flatten()
    logger.info("Total time per call: %s +- %s", per_call.mean(), per_call.std())
