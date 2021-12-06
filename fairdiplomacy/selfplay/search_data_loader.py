# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Generator, Optional, Tuple, Sequence, List
import collections
import itertools
import json
import logging
import pathlib
import socket
import time
import queue as queue_lib

import nest
import numpy as np
import postman
import psutil
import torch
import torch.utils.tensorboard


import conf.conf_cfgs
import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger
import fairdiplomacy.selfplay.vtrace
import fairdiplomacy.game
from fairdiplomacy import pydipcc
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.plausible_order_sampling import are_supports_coordinated
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.get_xpower_supports import compute_xpower_supports
from fairdiplomacy.selfplay import rela
from fairdiplomacy.selfplay.ckpt_syncer import ValuePolicyCkptSyncer
from fairdiplomacy.selfplay.execution_context import ExecutionContext
from fairdiplomacy.selfplay.h2h_evaler import H2HEvaler, save_game, GAME_WRITE_TIMEOUT
from fairdiplomacy.selfplay.search_rollout import (
    RollloutResult,
    ReSearchRolloutBatch,
    yield_rollouts,
)
from fairdiplomacy.selfplay.search_utils import unparse_device, perform_retry_loop
from fairdiplomacy.selfplay.staged_metrics_writer import StagedLogger
from fairdiplomacy.situation_check import run_situation_check
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()

QUEUE_PUT_TIMEOUT = 1.0


def get_trainer_server_fname(training_ddp_rank: int):
    return pathlib.Path(f"buffer_postman{training_ddp_rank}.txt")


CKPT_SYNC_DIR = "ckpt_syncer/ckpt"


ScoreDict = Dict[str, float]
ScoreDictPerPower = Dict[Optional[str], ScoreDict]


X_POSSIBLE_ACTIONS = "observations/x_possible_actions"

# Folder to write TB logs.
TB_FOLDER = "tb"


def flatten_dict(nested_tensor_dict):
    def _recursive(nested_tensor_dict, prefix):
        for k, v in nested_tensor_dict.items():
            if isinstance(v, dict):
                yield from _recursive(v, prefix=f"{prefix}{k}/")
            else:
                yield f"{prefix}{k}", v

    return dict(_recursive(nested_tensor_dict, prefix=""))


def compress_and_flatten(nested_tensor_dict):
    d = flatten_dict(nested_tensor_dict)
    d[X_POSSIBLE_ACTIONS] = d[X_POSSIBLE_ACTIONS].to(torch.short)
    offsets = {}
    last_offset = 0
    for k, v in d.items():
        if v.dtype == torch.float32:
            offsets[k] = (last_offset, last_offset + v.numel())
            last_offset += v.numel()
    storage = torch.empty(last_offset)
    for key, (start, end) in offsets.items():
        storage[start:end] = d[key].view(-1)
        d[key] = storage[start:end].view(d[key].size())

    return d


def unflatten_dict(flat_tensor_dict):
    d = {}
    for k, v in flat_tensor_dict.items():
        parts = k.split("/")
        subd = d
        for p in parts[:-1]:
            if p not in subd:
                subd[p] = {}
            subd = subd[p]
        subd[parts[-1]] = v
    return d


def decompress_and_unflatten(flat_tensor_dict):
    d = flat_tensor_dict.copy()
    d[X_POSSIBLE_ACTIONS] = d[X_POSSIBLE_ACTIONS].to(torch.int32)
    d = unflatten_dict(d)
    return d


def queue_rollouts(out_queue: mp.Queue, log_path, log_level, need_games, **kwargs) -> None:
    if log_path:
        heyhi.setup_logging(console_level=None, fpath=log_path, file_level=log_level)
    try:
        rollout_result: RollloutResult
        for rollout_result in yield_rollouts(**kwargs):
            if need_games:
                game_meta = rollout_result.game_meta.copy()
                game_meta["game"] = game_meta["game"].to_json()
                item = (rollout_result.batch, game_meta, rollout_result.last_ckpt_metas)
            else:
                item = rollout_result.batch
            try:
                out_queue.put(item, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def _drop_meta(generator):
    for rollout_result in generator:
        yield rollout_result.batch


def _join_batches(batches: Sequence[ReSearchRolloutBatch]) -> ReSearchRolloutBatch:
    merged = {}
    for k in ReSearchRolloutBatch._fields:
        values = []
        for b in batches:
            values.append(getattr(b, k))
        if k == "observations":
            values = DataFields.cat(values)
        else:
            values = torch.cat(values, 0)
        merged[k] = values
    return ReSearchRolloutBatch(**merged)


class TestSituationEvaller:
    """A process that run situation_check on a single GPU."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.p = ExceptionHandlingProcess(
            target=self.worker, args=[], kwargs=self.kwargs, daemon=True
        )
        self.p.start()

    @classmethod
    def worker(
        cls, *, cfg, agent_cfg, ckpt_dir: pathlib.Path, device: str, log_file: Optional[str]
    ):
        if log_file is not None:
            heyhi.setup_logging(console_level=None, fpath=log_file, file_level=logging.INFO)

        # FIXME(akhti): better
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=TB_FOLDER)

        last_ckpt = None
        get_ckpt_id = lambda p: int(p.name.split(".")[0][5:])
        agents: Dict[str, Dict] = {}
        agents["bl"] = dict()
        agents["rol0_it256"] = {"rollouts_cfg.max_rollout_length": 0, "n_rollouts": 256}
        agents["rol1_it256"] = {"rollouts_cfg.max_rollout_length": 1, "n_rollouts": 256}
        agents["rol2_it256"] = {"rollouts_cfg.max_rollout_length": 2, "n_rollouts": 256}
        with open(heyhi.PROJ_ROOT / "test_situations.json") as f:
            meta = json.load(f)
        while True:
            newest_ckpt = max(ckpt_dir.glob("epoch*.ckpt"), key=get_ckpt_id, default=None)
            if newest_ckpt == last_ckpt:
                time.sleep(20)
                continue
            logging.info("Loading checkpoint: %s", newest_ckpt)
            last_ckpt = newest_ckpt
            for name, agent_kwargs in agents.items():
                agent = build_agent_from_cfg(
                    agent_cfg,
                    device=unparse_device(device),
                    value_model_path=last_ckpt,
                    **agent_kwargs,
                )
                scores = run_situation_check(meta, agent)
                del agent
                writer.add_scalar(
                    f"eval_test_sit/{name}",
                    float(np.mean(list(scores.values()))),
                    global_step=get_ckpt_id(last_ckpt),
                )

    def terminate(self):
        if self.p is not None:
            self.p.kill()
            self.p = None


class EvalPlayer:
    """A group of processes that run eval selfplay on one GPU."""

    def __init__(
        self,
        *,
        self_play,
        log_dir,
        rollout_cfg,
        device,
        ckpt_sync_path,
        num_procs=5,
        game_kwargs: Dict,
        cores: Optional[Tuple[int, ...]],
        game_json_paths: Optional[Sequence[str]],
    ):
        def _build_rollout_kwargs(proc_id):
            assert rollout_cfg.agent.WhichOneof("agent") == "searchbot", rollout_cfg.agent
            return dict(
                searchbot_cfg=rollout_cfg.agent.searchbot,
                game_json_paths=game_json_paths,
                seed=proc_id,
                device=device,
                ckpt_sync_path=ckpt_sync_path,
                need_games=True,
                game_kwargs=game_kwargs,
                extra_params_cfg=rollout_cfg.extra_params,
            )

        logging.info("Creating eval rollout queue")
        self.queue = mp.Queue(maxsize=4000)
        tag = "sp" if self_play else "exploit"
        assert self_play
        logging.info("Creating eval rollout workers")
        self.procs = []
        for i in range(num_procs):
            kwargs = _build_rollout_kwargs(i)
            log_path = log_dir / f"eval_rollout_{tag}_{i:03d}.log"
            kwargs["log_path"] = log_path
            kwargs["log_level"] = logging.WARNING
            logging.info(
                f"Eval Rollout process {i} will write logs to {log_path} at level %s",
                kwargs["log_level"],
            )
            kwargs["collect_game_logs"] = True
            kwargs["eval_mode"] = True
            self.procs.append(
                ExceptionHandlingProcess(
                    target=queue_rollouts, args=[self.queue], kwargs=kwargs, daemon=True
                )
            )
        logging.info("Adding saving worker")
        self.procs.append(
            ExceptionHandlingProcess(
                target=self.aggregate_worker,
                args=[],
                kwargs=dict(
                    queue=self.queue,
                    dst_dir=pathlib.Path(f"games_{tag}").absolute(),
                    tag=tag,
                    save_every_secs=GAME_WRITE_TIMEOUT,
                ),
                daemon=True,
            )
        )

        logging.info("Starting eval rollout workers")
        for p in self.procs:
            p.start()
        if cores:
            logging.info("Setting affinities")
            for p in self.procs:
                psutil.Process(p.pid).cpu_affinity(cores)
        logging.info("Done")

    @classmethod
    def aggregate_worker(
        cls, *, queue: mp.Queue, tag: str, dst_dir: pathlib.Path, save_every_secs: float
    ):
        dst_dir.mkdir(exist_ok=True, parents=True)
        logger = fairdiplomacy.selfplay.remote_metric_logger.get_remote_logger(tag="eval_sp")
        counters = collections.defaultdict(fairdiplomacy.selfplay.metrics.FractionCounter)
        max_seen_epoch = -1
        num_games = 0

        def process_metrics(epoch, game_json):
            nonlocal logger
            nonlocal counters
            nonlocal max_seen_epoch
            nonlocal num_games

            MIN_GAMES_FOR_STATS = 50

            if max_seen_epoch < epoch:
                if num_games >= MIN_GAMES_FOR_STATS:
                    metrics = {
                        f"eval_{tag}/{key}": value.value() for key, value in counters.items()
                    }
                    metrics[f"eval_{tag}/num_games"] = num_games
                    logger.log_metrics(metrics, max_seen_epoch)
                    counters.clear()
                    num_games = 0
                max_seen_epoch = epoch
            num_games += 1
            game = pydipcc.Game.from_json(game_json)
            x_supports = compute_xpower_supports(game)
            counters["num_orders"].update(x_supports["o"])
            counters["sup_to_all_share"].update(x_supports["s"], x_supports["o"])
            counters["sup_xpower_to_sup_share"].update(x_supports["x"], x_supports["s"])
            counters["episode_length"].update(len(game.get_phase_history()))
            n_orders, n_coordinated = 0, 0
            for phase_data in game.get_phase_history():
                for action in phase_data.orders.values():
                    n_orders += 1
                    n_coordinated += are_supports_coordinated(action)
            counters["coordindated_share"].update(n_coordinated, n_orders)
            counters["r_draw"].update(max(game.get_square_scores()) < 0.99)

        last_save = 0.0
        while True:
            tensors, game_meta, last_ckpt_metas = queue.get()
            try:
                epoch = max(meta["epoch"] for meta in last_ckpt_metas.values())
            except KeyError:
                logging.error("Bad Meta %s:", last_ckpt_metas)
                raise

            game_json = game_meta["game"]
            process_metrics(epoch, game_json)

            now = time.time()
            if now - last_save > save_every_secs:
                save_game(
                    tensors=tensors,
                    game_json=game_json,
                    epoch=epoch,
                    dst_dir=dst_dir,
                    game_id=game_meta["game_id"],
                    start_phase=game_meta["start_phase"],
                )
                last_save = now

    def terminate(self):
        logging.info("Killing eval processes")
        for proc in self.procs:
            proc.kill()
        self.procs = []


def _read_game_json_paths(initial_games_index_file):
    game_json_paths = []
    with open(initial_games_index_file) as stream:
        for line in stream:
            line = line.split("#")[0].strip()
            if line:
                game_json_paths.append(line)
    assert game_json_paths, initial_games_index_file
    return game_json_paths


class Rollouter:
    """A supervisor for a group of processes that are doing rollouts.

    The group consist of a bunch of rollout workers and single communication
    worker that aggregates the data and sends it to the master. The user is
    expected to call start_rollout_procs and then either
    run_communicator_loop or start_communicator_loop.

    The class also provides functionality to query rollouts directly without
    launching any processes. In this case the client is expected to pass
    local_mode=True and call get_local_batch_iterator.
    """

    def __init__(
        self,
        rollout_cfg: conf.conf_cfgs.ExploitTask.SearchRollout,
        log_dir: pathlib.Path,
        *,
        game_json_paths: Optional[Sequence[str]] = None,
        local_mode: bool = False,
        game_kwargs: Dict,
        devices: Sequence[str],
        cores: Optional[Sequence[int]] = None,
        ddp_world_size: int,
    ):
        self.rollout_cfg = rollout_cfg
        self._cores = cores
        self._rollout_devices = devices
        job_env = heyhi.get_job_env()
        self._rank = job_env.global_rank
        self._local_mode = local_mode
        self._log_dir = log_dir
        self._game_json_paths = game_json_paths
        self._game_kwargs = game_kwargs
        self._communicator_proc = None
        self._rollout_procs = None
        self._ddp_world_size = ddp_world_size

        # Machine 2, if at least 2 machines, and the only machine otherwise.
        is_writing_stats = not local_mode and (self._rank == 1 or job_env.num_nodes == 1)
        if is_writing_stats:
            logging.info("This machine will write Rollouter logs")
            self._stats_server, self._stats_server_addr = self._initialze_postman_server()
            self._logger = StagedLogger(tag="rollouter", min_samples=100)
        else:
            self._stats_server = None
            self._logger = None

    def _initialze_postman_server(self):
        def add_stats(stats):
            stats = {k: v.item() for k, v in stats.items()}
            epoch = stats.pop("epoch")
            self._logger.add_metrics(data=stats, global_step=epoch)

        host = socket.gethostname()
        server = postman.Server("0.0.0.0:0")
        server.bind("add_stats", add_stats, batch_size=1)
        server.run()
        master_addr = f"{host}:{server.port()}"
        logging.info("Kicked of postman stats server on %s", master_addr)
        return server, master_addr

    def terminate(self):
        if self._communicator_proc is not None:
            logging.info("Killing collector process")
            self._communicator_proc.kill()
        if self._rollout_procs is not None:
            logging.info("Killing rollout processes")
            for proc in self._rollout_procs:
                proc.kill()
        if self._stats_server is not None:
            logging.info("Stopping Rollouter stats PostMan server")
            self._stats_server.stop()
        if self._logger is not None:
            self._logger.close()

    def num_alive_workers(self):
        if self._local_mode:
            return 1
        return sum(int(proc.is_alive()) for proc in self._rollout_procs)

    def is_communicator_alive(self):
        assert self._communicator_proc is not None
        return self._communicator_proc.is_alive()

    def get_local_batch_iterator(self):
        assert self._local_mode, "Not in local mode"
        rollout_generator = iter(
            _drop_meta(yield_rollouts(**self._build_rollout_kwargs(proc_id=0)))
        )
        return iter(
            self._yield_batches(
                chunk_length=self.rollout_cfg.chunk_length, rollout_iterator=rollout_generator
            )
        )

    @classmethod
    def _connect_to_masters(cls, ddp_world_size: int):
        timeout_secs = 60 * 60  # Wait 1h and die.
        sleep_secs = 10

        addrs_and_clients = []
        for training_ddp_rank in range(ddp_world_size):
            trainer_server_fname = get_trainer_server_fname(training_ddp_rank)
            success = False
            for _ in range(timeout_secs // sleep_secs + 1):
                if not trainer_server_fname.exists():
                    logging.info("Waiting for %s to appear", trainer_server_fname)
                    time.sleep(5)
                    continue
                with open(trainer_server_fname) as stream:
                    master_addr = stream.read().strip()
                logging.info("Trying to connect to %s", master_addr)
                try:
                    buffer_client = postman.Client(master_addr)
                    buffer_client.connect()
                    buffer_client.heartbeat(torch.zeros(1))
                except Exception as e:
                    logging.error("Got error: %s", e)
                    time.sleep(5)
                    continue
                logging.info("Successfully connected to %s", master_addr)
                addrs_and_clients.append((master_addr, buffer_client))
                success = True
                break
            if not success:
                raise RuntimeError(
                    "Failed to connect to the trainer after %d minutes", timeout_secs // 60
                )
        return addrs_and_clients

    def _build_rollout_kwargs(self, proc_id: int) -> Dict:
        assert self.rollout_cfg.agent.WhichOneof("agent") == "searchbot", self.rollout_cfg.agent
        return dict(
            searchbot_cfg=self.rollout_cfg.agent.searchbot,
            game_json_paths=self._game_json_paths,
            seed=proc_id,
            device=self._rollout_devices[proc_id % len(self._rollout_devices)],
            ckpt_sync_path=CKPT_SYNC_DIR,
            game_kwargs=self._game_kwargs,
            extra_params_cfg=self.rollout_cfg.extra_params,
        )

    def start_rollout_procs(self):
        logging.info("Rollout devices: %s", self._rollout_devices)
        if not self._rollout_devices:
            logging.warning("No devices available. No rollout workers will be launched")
            self._rollout_procs = []
            return

        rollout_cfg = self.rollout_cfg
        assert not self._local_mode
        assert rollout_cfg.num_workers_per_gpu > 0
        logging.info("Creating rollout queue")
        queue = mp.Queue(maxsize=40)
        logging.info("Creating rollout workers")
        procs = []
        for i in range(rollout_cfg.num_workers_per_gpu * len(self._rollout_devices)):
            kwargs = self._build_rollout_kwargs(i)
            kwargs["need_games"] = False
            if self._stats_server is not None:
                kwargs["stats_server"] = self._stats_server_addr
            if rollout_cfg.verbosity >= 1:
                log_path = self._log_dir / f"rollout_{self._rank:03d}_{i:03d}.log"
                kwargs["log_path"] = log_path
                kwargs["log_level"] = (
                    logging.INFO if i == 0 or rollout_cfg.verbosity >= 2 else logging.WARNING
                )
                logging.info(
                    f"Rollout process {i} will write logs to {log_path} at level %s",
                    kwargs["log_level"],
                )
            procs.append(
                ExceptionHandlingProcess(
                    target=queue_rollouts, args=[queue], kwargs=kwargs, daemon=True
                )
            )
        logging.info("Starting rollout workers")
        for p in procs:
            p.start()
        if self._cores:
            logging.info("Setting affinities")
            for p in procs:
                psutil.Process(p.pid).cpu_affinity(self._cores)
        logging.info("Done")
        self._rollout_generator = (queue.get() for _ in itertools.count())
        self._rollout_procs = procs
        self.queue = queue

    @classmethod
    def _communicator_worker(cls, chunk_length, queue, ddp_world_size: int):
        addrs_and_clients = cls._connect_to_masters(ddp_world_size)
        rollout_generator = (queue.get() for _ in itertools.count())
        data_generator = cls._yield_batches(
            chunk_length=chunk_length, rollout_iterator=iter(rollout_generator)
        )

        num_clients = len(addrs_and_clients)
        job_env = heyhi.get_job_env()
        next_client_idx = job_env.global_rank % num_clients

        for i, batch in enumerate(data_generator):
            # TODO (akhti): add hearbeat and reconnect.
            if i & (i + 1) == 0:
                logging.info("Collector got batch %d", i + 1)

            _master_addr, client = addrs_and_clients[next_client_idx]
            next_client_idx = (next_client_idx + 1) % num_clients

            # The master may be unresponsive while spawning and joining on training helper processes from
            # the multiprocessing module, or on torch.distributed.init_process_group
            # So try again a bunch of times before actually failing
            def send_to_client():
                client.add_replay(batch._asdict())

            perform_retry_loop(send_to_client, max_tries=20, sleep_seconds=10)

    def run_communicator_loop(self):
        """Run communication loop in the current process. Never returns."""
        assert self.rollout_cfg.num_workers_per_gpu > 0
        assert self._rollout_procs
        assert self.queue is not None
        self._communicator_worker(
            queue=self.queue,
            chunk_length=self.rollout_cfg.chunk_length,
            ddp_world_size=self._ddp_world_size,
        )

    def start_communicator_proc(self):
        """Starts a process that reads rollout_iterator and pushes into the buffer."""
        if not self._rollout_procs:
            logging.warning(
                "Not starting communcator process as no rollouts procs on this machine"
            )
            return

        assert self.queue is not None
        self._communicator_proc = ExceptionHandlingProcess(
            target=self._communicator_worker,
            kwargs=dict(
                queue=self.queue,
                chunk_length=self.rollout_cfg.chunk_length,
                ddp_world_size=self._ddp_world_size,
            ),
        )
        self._communicator_proc.start()

    @classmethod
    def _yield_batches(
        cls, chunk_length: int, rollout_iterator: Generator[ReSearchRolloutBatch, None, None]
    ) -> Generator[ReSearchRolloutBatch, None, None]:
        accumulated_batches = []
        size = 0
        assert chunk_length > 0
        for _ in itertools.count():
            rollout: ReSearchRolloutBatch
            rollout = next(rollout_iterator)
            size += len(rollout.rewards)
            accumulated_batches.append(rollout)
            if size > chunk_length:
                # Use strict > to simplify the code.
                joined_batch = _join_batches(accumulated_batches)
                while size > chunk_length:
                    extracted_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[:chunk_length], joined_batch
                    )
                    joined_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[chunk_length:], joined_batch
                    )
                    size -= chunk_length
                    yield extracted_batch
                accumulated_batches = [joined_batch]


class DataLoader:
    def __init__(
        self,
        model_path,
        rollout_cfg: conf.conf_cfgs.ExploitTask.SearchRollout,
        *,
        num_train_gpus: int,
        ectx: ExecutionContext,
    ):
        del model_path  # Not used.
        self.rollout_cfg = rollout_cfg

        self._train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._num_train_gpus = num_train_gpus
        self._ectx = ectx

        self._game_json_paths: Optional[List[str]]
        if rollout_cfg.initial_games_index_file:
            self._game_json_paths = _read_game_json_paths(rollout_cfg.initial_games_index_file)
        else:
            self._game_json_paths = None

        assert heyhi.is_master() == (ectx.training_ddp_rank is not None)

        self._rank = heyhi.get_job_env().global_rank

        if self.rollout_cfg.num_workers_per_gpu > 0 and not self.rollout_cfg.benchmark_only:
            self._use_buffer = True
        else:
            self._use_buffer = False

        self._game_kwargs = dict(draw_on_stalemate_years=rollout_cfg.draw_on_stalemate_years)

        self._assign_devices()
        self._log_dir = pathlib.Path("rollout_logs").absolute()
        self._log_dir.mkdir(exist_ok=True, parents=True)
        self._start_eval_procs()
        if self._ectx.is_training_master:
            self._ckpt_syncer = ValuePolicyCkptSyncer(CKPT_SYNC_DIR, create_dir=True)
        else:
            self._ckpt_syncer = None

        if self._ectx.is_training_helper:
            self._rollouter = None
        else:
            self._rollouter = Rollouter(
                rollout_cfg,
                log_dir=self._log_dir,
                devices=self._rollout_devices,
                cores=self.cores,
                game_json_paths=self._game_json_paths,
                game_kwargs=self._game_kwargs,
                local_mode=not self._use_buffer,
                ddp_world_size=ectx.ddp_world_size,
            )

        if self._use_buffer:
            self._initialze_buffer()
            self._need_warmup = True
            if self._ectx.is_training_master or self._ectx.is_training_helper:
                self._initialze_postman_server()

            if not self._ectx.is_training_helper:
                self._rollouter.start_rollout_procs()
                if not self._ectx.is_training_master:
                    # will stay here until the job is dead or the master is dead.
                    self._rollouter.run_communicator_loop()
                else:
                    self._rollouter.start_communicator_proc()
        else:
            assert self._ectx.is_training_master
            self._local_episode_iterator = self._rollouter.get_local_batch_iterator()

    def _assign_devices(self):
        """Sets what CPUs and GPUs to use.

        Sets
            self.cores
            self._rollout_devices
            self._sitcheck_device
        """
        self.cores: Optional[Tuple[int, ...]]
        self._rollout_devices: List[str]
        self._sitcheck_device: Optional[str]
        self._eval_sp_device: Optional[str]
        self._eval_h2h_devices: Optional[List[str]]
        self._num_eval_procs: int

        if self.rollout_cfg.num_cores_to_reserve and self._ectx.is_training_master:
            self.cores = tuple(range(80 - self.rollout_cfg.num_cores_to_reserve, 80))
        else:
            self.cores = None

        if not torch.cuda.is_available():
            # CircleCI probably.
            self._rollout_devices = ["cpu"]
            self._sitcheck_device = "cpu"
            self._eval_sp_device = "cpu"
            self._eval_h2h_devices = ["cpu"] * len(self.h2h_evals)
            self._num_eval_procs = 1
        else:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if (
                len(devices) > 1
                and self._ectx.is_training_master
                and not self.rollout_cfg.benchmark_only
            ):
                # If the trainer and has more than 1 gpu, don't use GPU used for training.
                devices = devices[self._num_train_gpus :]
            full_machine = torch.cuda.device_count() == 8
            if self.rollout_cfg.test_situation_eval.do_eval and self._ectx.is_training_master:
                self._sitcheck_device = devices.pop(0) if full_machine else devices[0]
            else:
                self._sitcheck_device = None
            if self._ectx.is_training_master:
                self._eval_sp_device = devices.pop(0) if full_machine else devices[0]
            else:
                self._eval_sp_device = None
            if self._ectx.is_training_master:
                self._eval_h2h_devices = [
                    devices.pop(0) if full_machine else devices[0] for _ in self.h2h_evals
                ]
            else:
                self._eval_h2h_devices = None
            self._rollout_devices = devices
            self._num_eval_procs = 5 if full_machine else 1

        logging.info(f"Sit check device {self._sitcheck_device}")
        logging.info(f"Eval SelfPlay device {self._eval_sp_device}")
        logging.info(f"Eval H2H devices {self._eval_h2h_devices}")
        logging.info(f"Procs to use for evals: {self._num_eval_procs}")
        logging.info(f"Rollout devices {self._rollout_devices}")

    def _initialze_buffer(self):
        replay_params = dict(
            seed=10001,
            alpha=1.0,
            beta=0.4,
            prefetch=self.rollout_cfg.buffer.prefetch or 3,
            capacity=self.rollout_cfg.buffer.capacity // self._ectx.ddp_world_size,
            shuffle=self.rollout_cfg.buffer.shuffle,
        )
        self._buffer = rela.NestPrioritizedReplay(**replay_params)

        self._save_buffer_at = None
        self._preloaded_size = 0
        if self.rollout_cfg.buffer.load_path:
            logging.info("Loading buffer from: %s", self.rollout_cfg.buffer.load_path)
            if self._ectx.ddp_world_size > 1:
                logging.warning("Buffers on all machines will load the same content")
            assert pathlib.Path(
                self.rollout_cfg.buffer.load_path
            ).exists(), f"Cannot find the buffer dump: {self.rollout_cfg.buffer.load_path}"
            self._buffer.load(self.rollout_cfg.buffer.load_path)
            logging.info("Loaded. New size: %s", self._buffer.size())
            self._preloaded_size = self._buffer.size()
            if self.rollout_cfg.buffer.save_at:
                logging.warning("buffer.save_at is ignored")
        elif self.rollout_cfg.buffer.save_at > 0:
            self._save_buffer_at = self.rollout_cfg.buffer.save_at

        # If true, will go noop on postman buffer add calls.
        self._skip_buffer_adds = False

        # Timestamps where get_buffer_stats is called.
        self._first_call = time.time()
        self._last_call = time.time()
        self._last_size = self._buffer.num_add()
        # For throttling and stats.
        self._num_sampled = 0

    def _initialze_postman_server(self):
        num_added = 0

        def add_replay(data):
            nonlocal num_added
            if self._skip_buffer_adds:
                return
            if num_added < 10:
                logging.info(
                    "adding:\n\tdata=%s\n\tbuffer sz=%s",
                    nest.map(lambda x: x.shape, data),
                    self._buffer.size(),
                )
                num_added += 1
            data = compress_and_flatten(nest.map(lambda x: x.squeeze(0), data))
            priority = 1.0
            self._buffer.add_one(data, priority)

        def heartbeat(arg):
            del arg  # Unused.

        host = socket.gethostname()
        server = postman.Server("0.0.0.0:0")
        server.bind("add_replay", add_replay, batch_size=1)
        server.bind("heartbeat", heartbeat, batch_size=1)
        server.run()
        self._buffer_server = server
        master_addr = f"{host}:{server.port()}"
        logging.info("Kicked of postman buffer server on %s", master_addr)
        trainer_server_fname = get_trainer_server_fname(self._ectx.training_ddp_rank)
        with open(trainer_server_fname, "w") as stream:
            print(master_addr, file=stream)

    @property
    def h2h_evals(self):
        return [
            cfg for cfg in [self.rollout_cfg.h2h_eval_0, self.rollout_cfg.h2h_eval_1] if cfg.tag
        ]

    def _start_eval_procs(self):
        if not self._ectx.is_training_master:
            return
        if self.rollout_cfg.test_situation_eval.do_eval:
            log_file = self._log_dir / "eval_sitcheck.log"
            logging.info("Starting situation check process. Logs: %s", log_file)
            self._sitcheck = TestSituationEvaller(
                cfg=self.rollout_cfg.test_situation_eval.do_eval,
                agent_cfg=self.rollout_cfg.agent,
                # FIXME(akhti): better
                ckpt_dir=pathlib.Path("ckpt/"),
                device=self._sitcheck_device,
                log_file=log_file,
            )
        assert self._eval_sp_device is not None
        self._eval_player = EvalPlayer(
            self_play=True,
            log_dir=self._log_dir,
            rollout_cfg=self.rollout_cfg,
            device=self._eval_sp_device,
            num_procs=5,
            cores=self.cores,
            ckpt_sync_path=CKPT_SYNC_DIR,
            game_json_paths=self._game_json_paths,
            game_kwargs=self._game_kwargs,
        )
        if self._eval_h2h_devices:
            self._h2h_evalers = []
            assert len(self._eval_h2h_devices) == len(self.h2h_evals)
            for device, cfg in zip(self._eval_h2h_devices, self.h2h_evals):
                if cfg.initial_games_index_file:
                    json_paths = _read_game_json_paths(cfg.initial_games_index_file)
                else:
                    json_paths = self._game_json_paths
                self._h2h_evalers.append(
                    H2HEvaler(
                        h2h_cfg=cfg,
                        log_dir=self._log_dir,
                        agent_one_cfg=self.rollout_cfg.agent,
                        cores=self.cores,
                        ckpt_sync_path=CKPT_SYNC_DIR,
                        num_procs=self._num_eval_procs,
                        device=device,
                        game_json_paths=json_paths,
                        game_kwargs=self._game_kwargs,
                    )
                )

    def extract_eval_scores(self) -> Optional[ScoreDict]:
        return None

    def terminate(self):
        logging.warning(
            "DataLoader is being destroyed. Any data read before this point may misbehave"
        )
        if (self._ectx.is_training_master or self._ectx.is_training_helper) and self._use_buffer:
            logging.info("Stopping buffer PostMan server")
            self._buffer_server.stop()
        logging.info("Killing Rollouter")
        if self._rollouter is not None:
            self._rollouter.terminate()
        if getattr(self, "_sitcheck", None) is not None:
            self._sitcheck.terminate()
        if getattr(self, "_eval_player", None) is not None:
            self._eval_player.terminate()

    def get_buffer_stats(self, *, prefix: str) -> Dict[str, float]:
        if not self._use_buffer:
            return {}
        # Note: added stores num_add only for this buffer.
        now, added = time.time(), self._buffer.num_add()
        num_buffers = self._ectx.ddp_world_size
        # Stores total added for all (ddp) buffers.
        added_with_preload_all = added * num_buffers + self._preloaded_size
        stats = {
            f"{prefix}size_examples": self._buffer.size(),
            f"{prefix}num_add_examples": self._buffer.num_add(),
            f"{prefix}speed_examples": (added - self._last_size)
            / max(1e-3, now - self._last_call),
            f"{prefix}avg_speed_examples": added / (now - self._first_call),
            f"{prefix}avg_read_speed_examples": self._num_sampled * (now - self._first_call),
        }
        stats = {k: v * self.rollout_cfg.chunk_length * num_buffers for k, v in stats.items()}
        stats[f"{prefix}size_bytes"] = self._buffer.total_bytes() * num_buffers
        stats[f"{prefix}size_numel"] = self._buffer.total_numel() * num_buffers
        if self._rollouter is not None:
            stats[f"{prefix}num_alive_workers"] = self._rollouter.num_alive_workers()
        stats[f"{prefix}overuse"] = (
            self._num_sampled * num_buffers / max(1, added_with_preload_all)
        )
        self._last_call, self._last_size = now, added
        return stats

    def sample_raw_batch_from_buffer(self):
        if self.rollout_cfg.enforce_train_gen_ratio > 0:
            while (
                self._num_sampled / (self._buffer.num_add() + self._preloaded_size)
                > self.rollout_cfg.enforce_train_gen_ratio
            ):
                print("sleep")
                time.sleep(1)
        per_dataloader_batch_size = self.rollout_cfg.batch_size // self._ectx.ddp_world_size
        batch, _ = self._buffer.sample(per_dataloader_batch_size, self._train_device)
        # all_rewards = torch.cat([x for x in batch["rewards"]])
        # print("YY", len(all_rewards), all_rewards.mean(0))
        self._num_sampled += per_dataloader_batch_size
        # TODO(akhti): better
        self._buffer.keep_priority()
        return batch

    def get_batch(self) -> ReSearchRolloutBatch:
        assert self._ectx.is_training_master or self._ectx.is_training_helper
        per_dataloader_batch_size = self.rollout_cfg.batch_size // self._ectx.ddp_world_size
        if self._use_buffer:
            # list_of_dicts, _ = self._buffer.get_all_content()
            # if list_of_dicts:
            #     all_rewards = torch.cat([x["rewards"] for x in list_of_dicts])
            #     print("XX", len(all_rewards), all_rewards.mean(0))
            if self._need_warmup:
                warmup_size = per_dataloader_batch_size * self.rollout_cfg.warmup_batches
                while self._buffer.size() < warmup_size:
                    logging.info("Warming up the buffer: %d/%d", self._buffer.size(), warmup_size)
                    if self._rollout_devices and self._rollouter is not None:
                        assert self._rollouter.is_communicator_alive(), "Oh shoot!"
                        assert self._rollouter.num_alive_workers() > 0, "Oh shoot!"
                    time.sleep(10)
                self._need_warmup = False
            if self._save_buffer_at and self._save_buffer_at < self._buffer.num_add():
                save_path = pathlib.Path(f"buffer{self._ectx.training_ddp_rank}.bin").absolute()
                logging.info("Saving buffer to %s", save_path)
                # To avoid overflowing, set buffer to read only mode.
                self._skip_buffer_adds = True
                self._buffer.save(str(save_path))
                self._skip_buffer_adds = False
                logging.info("Done.")
                self._save_buffer_at = None

            batch = self.sample_raw_batch_from_buffer()
            batch = decompress_and_unflatten(batch)
            batch = ReSearchRolloutBatch(**batch)
        else:
            batch = list(itertools.islice(self._local_episode_iterator, per_dataloader_batch_size))
            batch = [x._asdict() for x in batch]
            batch = ReSearchRolloutBatch(**nest.map_many(lambda x: torch.stack(x, 1), *batch))
        return batch

    def update_model(self, model, *, as_policy=True, as_value=True, **kwargs):
        if self._ectx.is_training_helper:
            return
        assert self._ectx.is_training_master
        assert self._ckpt_syncer is not None
        assert as_policy or as_value, "Should update policy, value, or both"
        if as_policy:
            self._ckpt_syncer.policy.save_state_dict(model, **kwargs)
        if as_value:
            self._ckpt_syncer.value.save_state_dict(model, **kwargs)
