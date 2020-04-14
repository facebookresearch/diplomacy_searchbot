from typing import Generator, List, Optional, Tuple, Sequence
import collections
import faulthandler
import itertools
import logging
import multiprocessing as mp
import functools
import signal
import os

import torch

import postman

from fairdiplomacy.agents.base_search_agent import run_server, cat_pad_inputs
from fairdiplomacy.agents.dipnet_agent import encode_inputs, encode_state, decode_order_idxs
from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS, N_SCS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.timing_ctx import TimingCtx


DEFAULT_BATCH_SIZE = 128


ExploitRollout = collections.namedtuple(
    "ExploitRollout", "power_id, game_json, actions, logprobs, observations"
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


def _noop_transform(args):
    return args


def model_output_transform_exploit(args):
    order_idxs, sampled_idxs, logits, final_scores = args
    del final_scores  # Not used.
    # Assuming no temperature.
    steps = logits.shape[2]
    sampled_idxs[order_idxs == EOS_IDX] = EOS_IDX
    action_logprobs = order_logits_to_action_logprobs(logits, sampled_idxs[..., :steps])
    # Returning non-truncated version of sampled_idxs to simplify concatenation.
    return order_idxs, sampled_idxs, action_logprobs


def model_output_transform_blueprint(args):
    order_idxs, sampled_idxs, logits, final_scores = args
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
        gpu_ids: Sequence[int],
        ckpt_sync_path: Optional[str],
        model_output_transform=_noop_transform,
        max_batch_size: int = 0,
        server_procs_per_gpu: int = 1,
    ):
        if not max_batch_size:
            max_batch_size = DEFAULT_BATCH_SIZE
        logging.info("Launching servers",)
        self.servers = []
        for gpu_id in gpu_ids:
            for proc_id in range(server_procs_per_gpu):
                q = mp.SimpleQueue()
                server = ExceptionHandlingProcess(
                    target=run_server,
                    kwargs=dict(
                        port=0,
                        batch_size=max_batch_size,
                        load_model_fn=functools.partial(
                            load_dipnet_model,
                            model_path,
                            map_location=f"cuda:{gpu_id}",
                            eval=True,
                        ),
                        ckpt_sync_path=ckpt_sync_path,
                        output_transform=model_output_transform,
                        seed=proc_id,
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


def do_model_request(client, x, temperature=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synchronous request to model server

    Arguments:
    - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
    - temperature: model softmax temperature

    Returns:
    - whatever the client returns
    """
    temp_array = torch.full((x[0].shape[0], 1), temperature)
    req = (*x, temp_array)
    try:
        return client.evaluate(req)
    except Exception:
        logging.exception(
            "Caught server error with inputs {}".format([(x.shape, x.dtype) for x in req])
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
    game_json,
    temperature=0.05,
    max_rollout_length=40,
    batch_size=1,
    initial_power_index=0,
) -> Generator[ExploitRollout, None, None]:
    """Do non-stop rollout for 1 (exploit) vs 6 (blueprint).

    This method can safely be called in a subprocess

    Arguments:
    - exploit_hostports: list of "{host}:{port}" of model servers for the
        agents that is training.
    - blueprint_hostports: list of "{host}:{port}" of model servers for the
        agents that is exploited. In selfplay, this is None.
    - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
    - temperature: model softmax temperature for rollout policy on the blueprint agent.
    - max_rollout_length: return SC count after at most # steps
    - batch_size: rollout # of games in parallel
    - initial_power_index: index of the country in POWERS to start the
        rollout with. Then will do round robin over all countries.

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
    self_play = blueprint_hostports is None
    blueprint_client_selector = (
        create_client_selector(blueprint_hostports) if not self_play else None
    )

    faulthandler.register(signal.SIGUSR2)
    torch.set_num_threads(1)

    exploit_power_selector = itertools.cycle(tuple(range(len(POWERS))))
    for _ in range(initial_power_index):
        next(exploit_power_selector)

    for exploit_power_id in exploit_power_selector:
        if self_play:
            # Not used.
            del exploit_power_id
        with timings("setup"):
            games = [Game.from_saved_game_format(game_json) for _ in range(batch_size)]
            turn_idx = 0
            observations = {i: [] for i in range(batch_size)}
            actions = {i: [] for i in range(batch_size)}
            cand_actions = {i: [] for i in range(batch_size)}
            logprobs = {i: [] for i in range(batch_size)}

        while not all(game.is_game_done for game in games) and turn_idx < max_rollout_length:
            with timings("prep"):
                batch_data = []
                for game in games:
                    if game.is_game_done:
                        continue
                    inputs = encode_inputs(
                        game,
                        all_possible_orders=game.get_all_possible_orders(),  # expensive
                        game_state=encode_state(game),
                    )
                    batch_data.append((game, inputs))

            assert batch_data

            with timings("cat_pad"):
                xs: List[Tuple] = [b[1] for b in batch_data]
                batch_inputs, _ = cat_pad_inputs(xs)

            with timings("model"):
                if not self_play:
                    (blueprint_batch_order_ids,) = do_model_request(
                        next(blueprint_client_selector), batch_inputs, temperature
                    )
                (
                    exploit_batch_order_ids,
                    exploit_cand_ids,
                    exploit_order_logprobs,
                ) = do_model_request(next(exploit_client_selector), batch_inputs)

            with timings("merging"):
                if self_play:
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
                for (game, _), power_orders in zip(batch_data, batch_orders):
                    for power, orders in zip(POWERS, power_orders):
                        game.set_orders(power, list(orders))

                inner_index = 0
                for i, game in enumerate(games):
                    if not game.is_game_done:
                        game.process()
                        if self_play:
                            actions[i].append(exploit_batch_order_ids[inner_index])
                            cand_actions[i].append(exploit_batch_order_ids[inner_index])
                            logprobs[i].append(exploit_order_logprobs[inner_index])
                        else:
                            actions[i].append(
                                exploit_batch_order_ids[inner_index, exploit_power_id]
                            )
                            cand_actions[i].append(exploit_cand_ids[inner_index, exploit_power_id])
                            logprobs[i].append(
                                exploit_order_logprobs[inner_index, exploit_power_id]
                            )
                        observations[i].append([x[inner_index] for x in batch_inputs])
                        inner_index += 1

            turn_idx += 1

        logging.debug(
            f"end do_rollout pid {os.getpid()} for {batch_size} games in {turn_idx} turns. timings: "
            f"{ {k : float('{:.3}'.format(v)) for k, v in timings.items()} }."
        )
        for i in range(batch_size):
            final_game_json = games[i].to_saved_game_format()
            if self_play:
                assert False, "add cand index"
                for power_id in range(len(POWERS)):
                    yield ExploitRollout(
                        power_id=power_id,
                        game_json=final_game_json,
                        actions=torch.stack([x[power_id] for x in actions[i]], 0),
                        logprobs=torch.stack([x[power_id] for x in logprobs[i]], 0),
                        observations=[torch.stack(obs, 0) for obs in zip(*observations[i])],
                    )
            else:
                yield ExploitRollout(
                    power_id=exploit_power_id,
                    game_json=final_game_json,
                    actions=torch.stack(actions[i], 0),
                    logprobs=torch.stack(logprobs[i], 0),
                    observations=[torch.stack(obs, 0) for obs in zip(*observations[i])]
                    + [torch.stack(cand_actions[i], 0)],
                )
