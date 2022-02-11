# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple, Sequence

import collections
import json
import logging
import pathlib
import random
import time

import pandas as pd
import psutil
import torch

import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger
from fairdiplomacy import pydipcc
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.get_xpower_supports import compute_xpower_supports
from fairdiplomacy.env import OneSixPolicyProfile
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.selfplay.ckpt_syncer import build_search_agent_with_syncs
from fairdiplomacy.selfplay.search_rollout import ReSearchRolloutBatch, yield_game
from fairdiplomacy.selfplay.search_utils import unparse_device
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()

# Do not dump a game on disk more often that this.
GAME_WRITE_TIMEOUT = 60

MIN_GAMES_FOR_STATS = 50


class H2HEvaler:
    def __init__(
        self,
        *,
        log_dir,
        h2h_cfg,
        agent_one_cfg,
        device,
        ckpt_sync_path,
        num_procs,
        game_kwargs: Dict,
        cores: Optional[Tuple[int, ...]],
        game_json_paths: Optional[Sequence[str]],
    ):

        logging.info(f"Creating eval h2h {h2h_cfg.tag} rollout workers")
        self.queue = mp.Queue(maxsize=4000)
        self.procs = []
        for i in range(num_procs):
            log_path = log_dir / f"eval_h2h_{h2h_cfg.tag}_{i:03d}.log"
            kwargs = dict(
                queue=self.queue,
                device=device,
                ckpt_sync_path=ckpt_sync_path,
                agent_one_cfg=agent_one_cfg,
                agent_six_cfg=h2h_cfg.agent_six,
                game_json_paths=game_json_paths,
                game_kwargs=game_kwargs,
                seed=i,
                num_zero_epoch_evals=MIN_GAMES_FOR_STATS // num_procs + 5,
                use_trained_value=h2h_cfg.use_trained_value,
                use_trained_policy=h2h_cfg.use_trained_policy,
            )
            kwargs["log_path"] = log_path
            kwargs["log_level"] = logging.WARNING
            logging.info(
                f"H2H Rollout process {h2h_cfg.tag}/{i} will write logs to {log_path} at level %s",
                kwargs["log_level"],
            )
            self.procs.append(
                ExceptionHandlingProcess(target=self.eval_worker, kwargs=kwargs, daemon=True)
            )
        logging.info(f"Adding main h2h {h2h_cfg.tag} worker")
        self.procs.append(
            ExceptionHandlingProcess(
                target=self.aggregate_worker,
                kwargs=dict(queue=self.queue, tag=h2h_cfg.tag, save_every_secs=GAME_WRITE_TIMEOUT),
                daemon=True,
            )
        )

        logging.info(f"Starting h2h {h2h_cfg.tag} workers")
        for p in self.procs:
            p.start()
        if cores:
            logging.info("Setting affinities")
            for p in self.procs:
                psutil.Process(p.pid).cpu_affinity(cores)
        logging.info("Done")

    @classmethod
    def eval_worker(
        cls,
        *,
        seed,
        queue: mp.Queue,
        device: str,
        ckpt_sync_path: str,
        log_path: pathlib.Path,
        log_level,
        agent_one_cfg,
        agent_six_cfg,
        game_json_paths,
        game_kwargs: Dict,
        num_zero_epoch_evals: int,
        use_trained_value: bool,
        use_trained_policy: bool,
    ):

        # We collect this many games for the first ckpt before loading new
        # ckpt. This is to establish an accurate BL numbers where RL agent net
        # in equivalent to the blueprint it's initialized from.
        num_evals_without_reload_left = num_zero_epoch_evals

        heyhi.setup_logging(console_level=None, fpath=log_path, file_level=log_level)

        device_id = unparse_device(device)
        assert agent_one_cfg.searchbot is not None, "Must be searchbot agent"
        agent_one, do_sync_fn = build_search_agent_with_syncs(
            agent_one_cfg.searchbot,
            ckpt_sync_path=ckpt_sync_path,
            use_trained_policy=use_trained_policy,
            use_trained_value=use_trained_value,
            device_id=device_id,
        )
        agent_six = build_agent_from_cfg(agent_six_cfg, device=device_id)
        random.seed(seed)
        torch.manual_seed(seed)

        # Hack: using whatever syncer is listed first to detect epoch.
        main_meta = next(iter(do_sync_fn().values()))

        if main_meta["epoch"] > 0:
            # First ckpt is not on zero epoch. Disabling.
            num_evals_without_reload_left = 0

        for game_id, game in yield_game(seed, game_json_paths, game_kwargs):
            start_phase = game.current_short_phase
            if num_evals_without_reload_left > 0:
                num_evals_without_reload_left -= 1
            else:
                main_meta = next(iter(do_sync_fn().values()))

            # Agent one must be alive at the start of the game.
            starting_sos = game.get_square_scores()
            agent_one_power = random.choice(
                [p for p, score in zip(POWERS, starting_sos) if score > 1e-3]
            )
            policy_profile = OneSixPolicyProfile(
                agent_one=agent_one,
                agent_six=agent_six,
                agent_one_power=agent_one_power,
                seed=random.randint(0, 100000),
            )

            while not game.is_game_done:
                power_orders = policy_profile.get_all_power_orders(game)
                for power, orders in power_orders.items():
                    if not game.get_orderable_locations().get(power):
                        continue
                    game.set_orders(power, orders)
                game.process()

            queue.put(
                {
                    "last_ckpt_meta": main_meta,
                    "game_json": game.to_json(),
                    "agent_one_power": agent_one_power,
                    "game_id": game_id,
                    "start_phase": start_phase,
                }
            )

    @classmethod
    def aggregate_worker(cls, *, queue: mp.Queue, tag: str, save_every_secs: float):
        logger = fairdiplomacy.selfplay.remote_metric_logger.get_remote_logger(
            tag=f"eval_h2h_{tag}"
        )
        counters = collections.defaultdict(fairdiplomacy.selfplay.metrics.FractionCounter)
        max_seen_epoch = -1
        num_games = 0

        def process_metrics(epoch, game_json, power):
            nonlocal logger
            nonlocal counters
            nonlocal max_seen_epoch
            nonlocal num_games

            if max_seen_epoch < epoch:
                if num_games >= MIN_GAMES_FOR_STATS:
                    metrics = {
                        f"eval_h2h_{tag}/{key}": value.value() for key, value in counters.items()
                    }
                    metrics[f"eval_h2h_{tag}/num_games"] = num_games
                    logger.log_metrics(metrics, max_seen_epoch)
                    counters.clear()
                    num_games = 0
                max_seen_epoch = epoch

            num_games += 1
            game = pydipcc.Game.from_json(game_json)
            counters["episode_length"].update(len(game.get_phase_history()))
            scores = game.get_square_scores()
            counters["r_draw_all"].update(max(scores) < 0.99)
            counters["r_solo"].update(scores[POWERS.index(power)] > 0.99)
            counters["r_draw"].update(0.001 < scores[POWERS.index(power)] < 0.99)
            counters["r_dead"].update(scores[POWERS.index(power)] < 0.001)
            counters["r_square_score"].update(scores[POWERS.index(power)])

            x_supports_power = compute_xpower_supports(game, only_power=power)
            counters["sup_to_all_share"].update(x_supports_power["s"], x_supports_power["o"])
            counters["sup_xpower_to_sup_share"].update(
                x_supports_power["x"], x_supports_power["s"]
            )

        last_save = 0
        game_dump_path = pathlib.Path(f"games_h2h_{tag}").absolute()
        game_dump_path.mkdir(exist_ok=True, parents=True)
        while True:
            data = queue.get()
            try:
                epoch = data["last_ckpt_meta"]["epoch"]
            except KeyError:
                logging.error("Bad Meta: %s", data["last_ckpt_meta"])
                raise

            process_metrics(epoch, data["game_json"], data["agent_one_power"])

            now = time.time()
            if now - last_save > save_every_secs:
                save_game(
                    game_json=data["game_json"],
                    epoch=epoch,
                    dst_dir=game_dump_path,
                    game_id=data["game_id"],
                    start_phase=data["start_phase"],
                    agent_one_power=data["agent_one_power"],
                )
                last_save = now

    def terminate(self):
        logging.info("Killing H2H processes")
        for proc in self.procs:
            proc.kill()
        self.procs = []


def _rollout_batch_to_dataframe(tensors: ReSearchRolloutBatch) -> pd.DataFrame:
    data = {}
    for i, p in enumerate(POWERS):
        data[f"reward_{p}"] = tensors.rewards[:, i].numpy()
    data["done"] = tensors.done.numpy()
    for i, p in enumerate(POWERS):
        data[f"is_explore_{p}"] = tensors.is_explore[:, i].numpy()
    df = pd.DataFrame(data)
    df.index.name = "timestamp"
    return df


def save_game(
    *,
    game_json: str,
    epoch: int,
    dst_dir: pathlib.Path,
    game_id: str,
    start_phase: str,
    tensors: Optional[ReSearchRolloutBatch] = None,
    agent_one_power: Optional[str] = None,
):
    counter = 0
    while True:
        name = f"game_{epoch:06d}_{counter:05d}_{game_id}"
        if agent_one_power:
            name += f"_{agent_one_power}"
        path = dst_dir / f"{name}.json"
        path_meta = dst_dir / f"{name}.meta.csv"
        if not path.exists():
            break
        counter += 1
    game_dict = json.loads(game_json)
    game_dict["viz"] = dict(game_id=game_id, start_phase=start_phase)
    with path.open("w") as stream:
        json.dump(game_dict, stream)
    if tensors is not None:
        _rollout_batch_to_dataframe(tensors).to_csv(path_meta)
