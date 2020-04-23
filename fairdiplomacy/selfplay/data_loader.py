from typing import Dict, Generator, Optional, Tuple, Sequence
import collections
import itertools
import json
import logging
import multiprocessing as mp
import pathlib
import queue as queue_lib

from google.protobuf.json_format import MessageToDict
import torch
import torch.utils.tensorboard


import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.vtrace
from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils import game_scoring
from fairdiplomacy.selfplay.rollout import (
    ExploitRollout,
    InferencePool,
    RolloutMode,
    model_output_transform_blueprint,
    model_output_transform_exploit,
    yield_rollouts,
)
import heyhi

QUEUE_PUT_TIMEOUT = 1.0


ScoreDict = Dict[str, float]


RolloutBatch = collections.namedtuple(
    "RolloutBatch", "power_ids, observations, rewards, actions, logprobs, done"
)


def get_ckpt_sync_dir():
    # TODO(akhti): check for multinode.
    if heyhi.is_on_slurm():
        return f"/scratch/slurm_tmpdir/{heyhi.get_slurm_job_id()}/ckpt_syncer/ckpt"
    else:
        # Use NFS. Slow, but at least don't have to clean or resolve conflicts.
        return "ckpt_syncer/ckpt"


def yield_rewarded_rollouts(reward_kwargs, rollout_kwargs, output_games):
    for item in yield_rollouts(**rollout_kwargs):
        game_json = item.game_json if output_games else None
        yield (rollout_to_batch(item, **reward_kwargs), game_json)


def queue_rollouts(out_queue: mp.Queue, reward_kwargs, rollout_kwargs, output_games=False) -> None:
    """A rollout worker function to push RolloutBatch's into the queue."""
    try:
        for item in yield_rewarded_rollouts(reward_kwargs, rollout_kwargs, output_games):
            try:
                out_queue.put(item, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def queue_scores(out_queue: mp.Queue, reward_kwargs, rollout_kwargs) -> None:
    """A rollout worker function to push scores from eval into the queue."""
    try:
        for (_, scores), _ in yield_rewarded_rollouts(
            reward_kwargs, rollout_kwargs, output_games=False
        ):
            try:
                out_queue.put(scores, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def rollout_to_batch(
    rollout: ExploitRollout,
    *,
    score_name: str = None,
    delay_penalty=False,
    differential_reward=False,
) -> Tuple[RolloutBatch, ScoreDict]:
    assert score_name is not None, "score_name is required"
    N = len(rollout.actions)
    scores = game_scoring.compute_game_scores(rollout.power_id, rollout.game_json)._asdict()
    if differential_reward:
        assert len(rollout.game_json["phases"]) == N + 1, (len(rollout.game_json["phases"]), N)
        target_score_history = torch.FloatTensor(
            [
                getattr(game_scoring.compute_phase_scores(rollout.power_id, phase), score_name,)
                for phase in rollout.game_json["phases"]
            ]
        )
        rewards = target_score_history[1:] - target_score_history[:-1]
    else:
        rewards = torch.zeros([N], dtype=torch.float)
        rewards[-1] = scores[score_name]

    if delay_penalty:
        rewards -= delay_penalty

    is_final = torch.zeros([N], dtype=torch.bool)
    is_final[-1] = True

    # Prepare observation to be used for training. Drop information about all
    # powers, but current.
    obs = list(rollout.observations)
    obs[4] = obs[4][:, rollout.power_id].clone()
    obs[5] = obs[5][:, rollout.power_id].clone()

    rollout_batch = RolloutBatch(
        power_ids=torch.full([N], rollout.power_id, dtype=torch.long),
        rewards=rewards,
        observations=obs,
        actions=rollout.actions,
        logprobs=rollout.logprobs,
        done=is_final,
    )
    return rollout_batch, scores


def get_default_rollout_scores() -> ScoreDict:
    scores = {k: 0 for k in fairdiplomacy.utils.game_scoring.GameScores._fields}
    scores["queue_size"] = 0
    return scores


def _join_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    merged = {}
    for k in RolloutBatch._fields:
        values = []
        for b in batches:
            values.append(getattr(b, k))
        if k == "observations":
            values = [torch.cat(x, 0) for x in zip(*values)]
        else:
            values = torch.cat(values, 0)
        merged[k] = values
    return RolloutBatch(**merged)


class Evaler:
    def __init__(
        self,
        *,
        model_path,
        reward_cfg,
        num_procs,
        blueprint_hostports,
        exploit_hostports,
        temperature,
        max_length=100,
    ):
        self.model_path = model_path

        game_json = Game().to_saved_game_format()

        def _build_rollout_kwargs(proc_id):
            return dict(
                reward_kwargs=MessageToDict(reward_cfg, preserving_proto_field_name=True),
                rollout_kwargs=dict(
                    mode=RolloutMode.EVAL,
                    initial_power_index=(proc_id % len(POWERS)),
                    blueprint_hostports=blueprint_hostports,
                    exploit_hostports=exploit_hostports,
                    game_json=game_json,
                    temperature=temperature,
                    max_rollout_length=max_length,
                    batch_size=1,
                ),
            )

        logging.info("Creating eval rollout queue")
        self.queue = mp.Queue(maxsize=4000)
        logging.info("Creating eval rollout workers")
        self.procs = [
            ExceptionHandlingProcess(
                target=queue_scores,
                args=[self.queue],
                kwargs=_build_rollout_kwargs(i),
                daemon=True,
            )
            for i in range(num_procs)
        ]
        logging.info("Starting eval rollout workers")
        for p in self.procs:
            p.start()
        logging.info("Done")

    def extract_scores(self) -> ScoreDict:
        qsize = self.queue.qsize()
        aggregated_scores = get_default_rollout_scores()
        for _ in range(qsize):
            try:
                scores = self.queue.get_nowait()
            except queue_lib.Empty:
                logging.warning("Kind of odd...")
                break
            for k, v in scores.items():
                aggregated_scores[k] += v
        for k in list(aggregated_scores):
            aggregated_scores[k] /= aggregated_scores["num_games"]
        del aggregated_scores["queue_size"]
        return aggregated_scores

    def terminate(self):
        logging.info("Killing eval processes")
        for proc in self.procs:
            proc.kill()


class DataLoader:
    """Starts rollout processes and inference servers to generate impala data.

    Yields rollout in chunks. Each chunk contains concatenated
    block_size rollouts.

    Yields tuples (RolloutBatch, metrics).

    Metrics is a dict of some end-of-rollout metrics summed over all rollouts.
    """

    def __init__(self, model_path, rollout_cfg: "conf.conf_pb2.ExploitTask.Rollout"):
        self.model_path = model_path
        self.rollout_cfg = rollout_cfg

        mp.set_start_method("spawn")

        self._start_inference_procs()
        self._start_rollout_procs()
        self._maybe_start_eval_procs()
        self._batch_iterator = iter(self._yield_batches())

    def _start_inference_procs(self):
        if torch.cuda.device_count() == 2:
            if self.rollout_cfg.single_rollout_gpu:
                inference_gpus = [1]
            else:
                inference_gpus = [0, 1]
        else:
            assert torch.cuda.device_count() == 8, torch.cuda.device_count()
            if self.rollout_cfg.single_rollout_gpu:
                inference_gpus = [1, 2]
            else:
                inference_gpus = [1, 2, 3, 4, 5, 6, 7]

        if len(inference_gpus) == 1:
            exploit_gpus = blueprint_gpus = inference_gpus
        elif self.rollout_cfg.selfplay:
            # Assign only one GPU for eval.
            blueprint_gpus = inference_gpus[:1]
            exploit_gpus = inference_gpus[1:]
        else:
            exploit_gpus = inference_gpus[: len(inference_gpus) // 2]
            blueprint_gpus = inference_gpus[len(inference_gpus) // 2 :]

        logging.info("Starting exploit PostMan servers on gpus: %s", exploit_gpus)
        exploit_inference_pool = InferencePool(
            model_path=self.model_path,
            gpu_ids=exploit_gpus,
            ckpt_sync_path=get_ckpt_sync_dir(),
            ckpt_sync_every=self.rollout_cfg.inference_ckpt_sync_every,
            max_batch_size=self.rollout_cfg.inference_batch_size,
            server_procs_per_gpu=self.rollout_cfg.server_procs_per_gpu,
            model_output_transform=model_output_transform_exploit,
        )
        logging.info("Starting blueprint PostMan servers on gpus: %s", blueprint_gpus)
        blueprint_inference_pool = InferencePool(
            model_path=self.model_path,
            gpu_ids=blueprint_gpus,
            ckpt_sync_path=None,
            max_batch_size=self.rollout_cfg.inference_batch_size,
            server_procs_per_gpu=self.rollout_cfg.server_procs_per_gpu,
            model_output_transform=model_output_transform_blueprint,
        )

        # Must save reference to the pools to keep the resources alive.
        self.blueprint_inference_pool = blueprint_inference_pool
        self.exploit_inference_pool = exploit_inference_pool

        self.blueprint_hostports = blueprint_inference_pool.hostports
        self.exploit_hostports = exploit_inference_pool.hostports

    def _start_rollout_procs(self):
        game_json = Game().to_saved_game_format()
        rollout_cfg = self.rollout_cfg

        def _build_rollout_kwargs(proc_id):
            if not rollout_cfg.selfplay:
                mode = RolloutMode.EXPLOIT
            else:
                mode = RolloutMode.SELFPLAY
            assert rollout_cfg.blueprint_temperature > 0
            return dict(
                reward_kwargs=MessageToDict(rollout_cfg.reward, preserving_proto_field_name=True),
                rollout_kwargs=dict(
                    mode=mode,
                    initial_power_index=(proc_id % len(POWERS)),
                    blueprint_hostports=self.blueprint_hostports,
                    exploit_hostports=self.exploit_hostports,
                    temperature=rollout_cfg.blueprint_temperature,
                    game_json=game_json,
                    max_rollout_length=rollout_cfg.rollout_max_length,
                    batch_size=rollout_cfg.rollout_batch_size,
                    fast_finish=rollout_cfg.fast_finish,
                ),
                output_games=rollout_cfg.dump_games_every > 0,
            )

        if rollout_cfg.num_rollout_processes > 0:
            logging.info("Creating rollout queue")
            queue = mp.Queue(maxsize=40)
            logging.info("Creating rollout workers")
            procs = [
                ExceptionHandlingProcess(
                    target=queue_rollouts,
                    args=[queue],
                    kwargs=_build_rollout_kwargs(i),
                    daemon=True,
                )
                for i in range(rollout_cfg.num_rollout_processes)
            ]
            logging.info("Starting rollout workers")
            for p in procs:
                p.start()
            logging.info("Done")
            rollout_generator = (queue.get() for _ in itertools.count())
            # Keeping track of there to prevent garbage collection.
            self._rollout_procs = procs
        else:
            queue = None
            rollout_generator = yield_rewarded_rollouts(**_build_rollout_kwargs(0))
            self._rollout_procs = None

        self.rollout_iterator = iter(rollout_generator)
        self.queue = queue

    def _maybe_start_eval_procs(self):
        if not self.rollout_cfg.selfplay or self.rollout_cfg.num_eval_rollout_processes == 0:
            self.evaler = None
        else:
            self.evaler = Evaler(
                model_path=self.model_path,
                reward_cfg=self.rollout_cfg.reward,
                num_procs=self.rollout_cfg.num_eval_rollout_processes,
                blueprint_hostports=self.blueprint_hostports,
                exploit_hostports=self.exploit_hostports,
                temperature=self.rollout_cfg.blueprint_temperature,
            )

    def extract_eval_scores(self) -> Optional[ScoreDict]:
        if self.evaler is None:
            return None
        else:
            return self.evaler.extract_scores()

    def terminate(self):
        logging.warning(
            "DataLoader is being destroyed. Any data read before this point may misbehave"
        )
        if self._rollout_procs is not None:
            logging.info("Killing rollout processes")
            for proc in self._rollout_procs:
                proc.kill()
        if self.evaler is not None:
            self.evaler.terminate()
        self.blueprint_inference_pool.terminate()
        self.exploit_inference_pool.terminate()

    def _yield_batches(self) -> Generator[Tuple[RolloutBatch, ScoreDict], None, None]:
        rollout_cfg = self.rollout_cfg
        accumulated_batches = []
        aggregated_scores = get_default_rollout_scores()
        size = 0
        batch_size = rollout_cfg.batch_size
        for rollout_id in itertools.count():
            rollout: RolloutBatch
            scores: ScoreDict
            (rollout, scores), game_json = next(self.rollout_iterator)
            if rollout_cfg.dump_games_every and rollout_id % rollout_cfg.dump_games_every == 0:
                power_id = rollout.power_ids[0].item()
                game_dump_folder = pathlib.Path(f"dumped_games")
                game_dump_folder.mkdir(exist_ok=True, parents=True)
                dump_path = game_dump_folder / f"game.{POWERS[power_id]}.{rollout_id:06d}.json"
                with (dump_path).open("w") as stream:
                    json.dump(game_json, stream)

            size += len(rollout.rewards)
            accumulated_batches.append(rollout)
            for k, v in scores.items():
                aggregated_scores[k] += v
            if rollout_cfg.do_not_split_rollouts:
                if size >= batch_size:
                    yield _join_batches(accumulated_batches), aggregated_scores
                    # Reset.
                    accumulated_batches = []
                    aggregated_scores = get_default_rollout_scores()
                    size = 0
            elif size > batch_size:
                # Use strict > to simplify the code.
                joined_batch = _join_batches(accumulated_batches)
                while size > batch_size:
                    extracted_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[:batch_size], joined_batch
                    )
                    joined_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[batch_size - rollout_cfg.batch_interleave_size :], joined_batch
                    )
                    size -= batch_size - rollout_cfg.batch_interleave_size
                    if self.queue is not None:
                        # Hack. We want queue size to be the size of the queue when batch is produced.
                        aggregated_scores["queue_size"] = (
                            self.queue.qsize() * aggregated_scores["num_games"]
                        )
                    yield extracted_batch, aggregated_scores
                    # Reset.
                    aggregated_scores = get_default_rollout_scores()
                accumulated_batches = [joined_batch]

    def get_batch(self) -> Tuple[RolloutBatch, ScoreDict]:
        return next(self._batch_iterator)