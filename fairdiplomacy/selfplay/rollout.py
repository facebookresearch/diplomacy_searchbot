from typing import Generator, List, Optional, Tuple, Sequence
import collections
import enum
import json
import faulthandler
import functools
import itertools
import logging
import multiprocessing as mp
import os
import signal

import numpy as np
import torch

import postman

from fairdiplomacy.agents.dipnet_agent import (
    encode_inputs,
    encode_state,
    decode_order_idxs,
    resample_duplicate_disbands_inplace,
)
from fairdiplomacy.agents.multiproc_search_agent import run_server, cat_pad_inputs
from fairdiplomacy.game import Game
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS, N_SCS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.timing_ctx import TimingCtx


DEFAULT_BATCH_SIZE = 128


class RolloutMode(enum.Enum):
    EXPLOIT = enum.auto()
    SELFPLAY = enum.auto()
    EVAL = enum.auto()


ExploitRollout = collections.namedtuple(
    "ExploitRollout", "power_id, game_json, actions, logprobs, observations, first_phase"
)


def order_logits_to_action_logprobs(logits, order_ids, mask=None):
    """Combine logits for orders to get log probs for actions (sequence of orders).

    Args:
        logits: float tensor of shape [..., seq, vocab]
        order_ids: long tensor of shape [..., seq].
        mask: float tensor of shape [..., seq] where 1.0 stands for real
            order, 0 for EOS_IDX. If not given, will be computed from
            order_ids.

    Returns:
        action_logprobs: tensor of shape [...].
    """
    if mask is None:
        mask = (order_ids != EOS_IDX).float()
    assert EOS_IDX == -1, "The code expected this. Re-write the code"
    order_ids = order_ids.clamp(0)  # -1 -> 0 to make nll_loss work.
    order_logprobs = -(
        torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(torch.flatten(logits, end_dim=-2), dim=-1),
            torch.flatten(order_ids),
            reduction="none",
        ).view_as(order_ids)
    )
    return (order_logprobs * mask).sum(-1)


def _noop_transform(inputs, args):
    return args


def model_output_transform_exploit(inputs, args):
    order_idxs, sampled_idxs, logits, final_scores = args
    resample_duplicate_disbands_inplace(
        order_idxs, sampled_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
    )
    del final_scores  # Not used.
    # Assuming no temperature.
    steps = logits.shape[2]
    sampled_idxs[order_idxs == EOS_IDX] = EOS_IDX
    action_logprobs = order_logits_to_action_logprobs(logits, sampled_idxs[..., :steps])
    # Returning non-truncated version of sampled_idxs to simplify concatenation.
    return order_idxs, sampled_idxs, action_logprobs


def model_output_transform_blueprint(inputs, args):
    order_idxs, sampled_idxs, logits, final_scores = args
    resample_duplicate_disbands_inplace(
        order_idxs, sampled_idxs, logits, inputs["x_possible_actions"], inputs["x_in_adj_phase"]
    )
    del sampled_idxs  # Not used.
    del logits  # Not used.
    del final_scores  # Not used.
    return (order_idxs,)


class InferencePool:
    """Starts a pool of postman servers that can do forward on the model.
    """

    def __init__(
        self,
        *,
        model_path: str,
        gpu_ids: Sequence[Optional[int]],
        ckpt_sync_path: Optional[str],
        ckpt_sync_every: int = 0,
        model_output_transform=_noop_transform,
        max_batch_size: int = 0,
        server_procs_per_gpu: int = 1,
    ):
        if not max_batch_size:
            max_batch_size = DEFAULT_BATCH_SIZE
        logging.info("Launching servers")
        self.servers = []
        for gpu_id in gpu_ids:
            for _ in range(server_procs_per_gpu):
                seed = int(torch.rand(1).item() * 100000)
                q = mp.SimpleQueue()
                server = ExceptionHandlingProcess(
                    target=run_server,
                    kwargs=dict(
                        port=0,
                        batch_size=max_batch_size,
                        load_model_fn=functools.partial(
                            load_dipnet_model,
                            model_path,
                            map_location=f"cuda:{gpu_id}" if gpu_id is not None else "cpu",
                            eval=True,
                        ),
                        ckpt_sync_path=ckpt_sync_path,
                        ckpt_sync_every=ckpt_sync_every,
                        output_transform=model_output_transform,
                        seed=seed,
                        device=gpu_id,
                        port_q=q,
                    ),
                    daemon=True,
                )
                server.start()
                self.servers.append((server, q))

        logging.info("Waiting for servers")
        self._hostports = tuple(f"127.0.0.1:{q.get()}" for _, q in self.servers)
        logging.info(f"Servers: {self._hostports}")

    @property
    def hostports(self) -> Tuple[str, ...]:
        return self._hostports

    def terminate(self):
        logging.info(
            "Terminating inference pool. The following servers will go down: %s", self.hostports
        )
        for server, _ in self.servers:
            server.kill()


def do_model_request(
    client, x: DataFields, temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synchronous request to model server

    Arguments:
        - x: a DataFields dict of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature

    Returns:
        - whatever the client returns
    """
    x = dict(x)
    B = x["x_board_state"].shape[0]
    x["temperature"] = torch.full((B, 1), temperature)
    try:
        return client.evaluate(x)
    except Exception:
        logging.error(
            "Caught server error with inputs {}".format(
                [(k, v.shape, v.dtype) for k, v in x.items()]
            )
        )
        raise


def strigify_orders_idxs(order_idxs: torch.Tensor) -> List[List[Tuple[str]]]:
    """Convert a tensor of order indices to string for the environment."""
    assert (
        len(order_idxs.shape) == 3
    ), f"Expected tensor of shape (batch, 7, max_orders), got: {order_idxs.shape}"
    order_idxs = order_idxs.numpy()
    string_orders = [
        [tuple(decode_order_idxs(order_idxs[b, p])) for p in range(len(POWERS))]
        for b in range(order_idxs.shape[0])
    ]
    return string_orders


def yield_rollouts(
    *,
    exploit_hostports: Sequence[str],
    blueprint_hostports: Optional[Sequence[str]],
    game_json_paths: Optional[Sequence[str]],
    mode: RolloutMode,
    fast_finish: bool = False,
    temperature=0.05,
    max_rollout_length=40,
    batch_size=1,
    seed=0,
) -> Generator[ExploitRollout, None, None]:
    """Do non-stop rollout for 1 (exploit) vs 6 (blueprint).

    This method can safely be called in a subprocess

    Arguments:
    - exploit_hostports: list of "{host}:{port}" of model servers for the
        agents that is training.
    - blueprint_hostports: list of "{host}:{port}" of model servers for the
        agents that is exploited. Ignored in SELFPLAY model
    - game_jsons: either None or a list of paths to json-serialized games.
    - mode: what kind of rollout to do. Defiens what will be outputed.
    - fast_finish: if True, the rollout is stopped once all non-blueprint agents lost.
    - temperature: model softmax temperature for rollout policy on the blueprint agent.
    - max_rollout_length: return SC count after at most # steps
    - batch_size: rollout # of games in parallel
    - seed: random seed.

    yields a ExploitRollout.

    """
    timings = TimingCtx()

    def create_client_selector(hostports):
        clients = []
        for hostport in hostports:
            client = postman.Client(hostport)
            client.connect(3)
            clients.append(client)
        return iter(itertools.cycle(clients))

    exploit_client_selector = create_client_selector(exploit_hostports)
    if mode != RolloutMode.SELFPLAY:
        assert blueprint_hostports is not None
        blueprint_client_selector = create_client_selector(blueprint_hostports)

    faulthandler.register(signal.SIGUSR2)
    torch.set_num_threads(1)

    exploit_power_selector = itertools.cycle(tuple(range(len(POWERS))))
    for _ in range(seed % len(POWERS)):
        next(exploit_power_selector)

    def yield_game():
        nonlocal game_json_paths
        nonlocal seed

        while True:
            if game_json_paths is None:
                yield Game()
            else:
                rng = np.random.RandomState(seed=seed)
                p = game_json_paths[rng.choice(len(game_json_paths))]
                with open(p) as stream:
                    game_serialized = json.load(stream)
                yield Game.from_saved_game_format(game_serialized)

    game_selector = yield_game()

    for exploit_power_id in exploit_power_selector:
        if mode == RolloutMode.SELFPLAY:
            # Not used.
            del exploit_power_id
            exploit_ids = frozenset(range(len(POWERS)))
        else:
            exploit_ids = frozenset([exploit_power_id])

        with timings("setup"):
            games = [next(game_selector) for _ in range(batch_size)]
            first_phases = [len(game.order_history) for game in games]
            turn_idx = 0
            observations = {i: [] for i in range(batch_size)}
            actions = {i: [] for i in range(batch_size)}
            cand_actions = {i: [] for i in range(batch_size)}
            logprobs = {i: [] for i in range(batch_size)}

        while turn_idx < max_rollout_length:
            with timings("prep"):
                batch_data = []
                for batch_idx, game in enumerate(games):
                    if game.is_game_done:
                        continue
                    if fast_finish:
                        last_centers = game.get_state()["centers"]
                        if not any(
                            len(last_centers[p]) > 0
                            for i, p in enumerate(POWERS)
                            if i not in exploit_ids
                        ):
                            continue
                    inputs = encode_inputs(
                        game,
                        all_possible_orders=game.get_all_possible_orders(),  # expensive
                        game_state=encode_state(game),
                    )
                    batch_data.append((game, inputs, batch_idx))

            if not batch_data:
                # All games are done.
                break

            with timings("cat_pad"):
                xs: List[Tuple] = [b[1] for b in batch_data]
                batch_inputs = cat_pad_inputs(xs)

            with timings("model"):
                if mode != RolloutMode.SELFPLAY:
                    (blueprint_batch_order_ids,) = do_model_request(
                        next(blueprint_client_selector), batch_inputs, temperature
                    )
                (
                    exploit_batch_order_ids,
                    exploit_cand_ids,
                    exploit_order_logprobs,
                ) = do_model_request(next(exploit_client_selector), batch_inputs)

            with timings("merging"):
                if mode == RolloutMode.SELFPLAY:
                    batch_order_idx = exploit_batch_order_ids
                else:
                    # Using all orders from the blueprint model except for ones for the epxloit power.
                    batch_order_idx = blueprint_batch_order_ids
                    batch_order_idx[:, exploit_power_id] = exploit_batch_order_ids[
                        :, exploit_power_id
                    ]

                batch_orders = strigify_orders_idxs(batch_order_idx)

            with timings("env"):
                assert len(batch_data) == len(batch_orders), "{} != {}".format(
                    len(batch_data), len(batch_orders)
                )

                # set_orders and process
                for (game, _, _), power_orders in zip(batch_data, batch_orders):
                    for power, orders in zip(POWERS, power_orders):
                        game.set_orders(power, list(orders))

                for inner_index, (game, _, i) in enumerate(batch_data):
                    if not game.is_game_done:
                        game.process()
                        if mode == RolloutMode.SELFPLAY:
                            actions[i].append(exploit_batch_order_ids[inner_index])
                            cand_actions[i].append(exploit_cand_ids[inner_index])
                            logprobs[i].append(exploit_order_logprobs[inner_index])
                        else:
                            actions[i].append(
                                exploit_batch_order_ids[inner_index, exploit_power_id]
                            )
                            cand_actions[i].append(exploit_cand_ids[inner_index, exploit_power_id])
                            logprobs[i].append(
                                exploit_order_logprobs[inner_index, exploit_power_id]
                            )
                        observations[i].append(batch_inputs.select(inner_index))

            turn_idx += 1

        logging.debug(
            f"end do_rollout pid {os.getpid()} for {batch_size} games in {turn_idx} turns. timings: "
            f"{ {k : float('{:.3}'.format(v)) for k, v in timings.items()} }."
        )
        for i in range(batch_size):
            final_game_json = games[i].to_saved_game_format()
            if mode == RolloutMode.SELFPLAY:
                for power_id in range(len(POWERS)):
                    extended_obs = DataFields.stack(observations[i])
                    extended_obs["cand_indices"] = torch.stack(
                        [x[power_id] for x in cand_actions[i]], 0
                    )
                    yield ExploitRollout(
                        power_id=power_id,
                        game_json=final_game_json,
                        actions=torch.stack([x[power_id] for x in actions[i]], 0),
                        logprobs=torch.stack([x[power_id] for x in logprobs[i]], 0),
                        observations=extended_obs,
                        first_phase=first_phases[i],
                    )
            else:
                extended_obs = DataFields.stack(observations[i])
                extended_obs["cand_indices"] = torch.stack(cand_actions[i], 0)
                yield ExploitRollout(
                    power_id=exploit_power_id,
                    game_json=final_game_json,
                    actions=torch.stack(actions[i], 0),
                    logprobs=torch.stack(logprobs[i], 0),
                    observations=extended_obs,
                    first_phase=first_phases[i],
                )
