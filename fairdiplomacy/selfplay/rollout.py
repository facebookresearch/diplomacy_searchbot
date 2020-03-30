from typing import Generator, List, Tuple, Sequence
import collections
import faulthandler
import itertools
import logging
import functools
import signal
import os

import torch.multiprocessing as mp
import torch

import postman

import diplomacy
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from fairdiplomacy.agents.base_search_agent import (
    run_server,
    cat_pad_inputs,
)
from fairdiplomacy.agents.dipnet_agent import encode_inputs, encode_state, decode_order_idxs
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS, N_SCS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.timing_ctx import TimingCtx


ExploitRollout = collections.namedtuple(
    "ExploitRollout", "power_id, game_jsons, actions, logprobs, observations"
)


def model_output_transform_exploit(args):
    order_idxs, order_scores, final_scores = args
    del final_scores  # Not used.
    # Assuming no temperature.
    order_logprobs = torch.nn.functional.log_softmax(order_scores, -1)
    return order_idxs, order_logprobs


def model_output_transform_blueprint(args):
    order_idxs, order_scores, final_scores = args
    del order_scores  # Not used.
    del final_scores  # Not used.
    return (order_idxs,)


class InferencePool:
    """Starts a pool of postman servers that can do forward on the model.

    Half of the servers will track ckpt_sync_path and reload the model
    (exploit servers), while the others (blueprint servers) will use the
    static version.
    """

    def __init__(self, *, model_path: str, gpu_ids: Sequence[int], ckpt_sync_path: str):
        n_server_procs = len(gpu_ids) * 1
        assert n_server_procs >= 2, "Not enought servers to have 2 models"
        max_batch_size = 128
        logging.info("Launching servers",)
        self.servers = []
        mp.set_start_method("spawn")
        gpu_selector = iter(itertools.cycle(gpu_ids))
        for i in range(n_server_procs):
            q = mp.SimpleQueue()
            is_exploit = i % 2 == 0
            if is_exploit:
                model_output_transform = model_output_transform_exploit
            else:
                model_output_transform = model_output_transform_blueprint

            server = ExceptionHandlingProcess(
                target=run_server,
                kwargs=dict(
                    port=0,
                    batch_size=max_batch_size,
                    load_model_fn=functools.partial(
                        load_dipnet_model, model_path, map_location="cuda", eval=True,
                    ),
                    ckpt_sync_path=ckpt_sync_path if is_exploit else None,
                    output_transform=model_output_transform,
                    seed=0,  # TODO(akhti): should this be something else?
                    device=next(gpu_selector),
                    port_q=q,
                ),
                daemon=True,
            )
            server.start()
            self.servers.append((server, q, is_exploit))

        logging.info("Waiting for servers")
        self._exploit_hostports = [
            f"127.0.0.1:{q.get()}" for _, q, is_exploit in self.servers if is_exploit
        ]
        self._bluprint_hostports = [
            f"127.0.0.1:{q.get()}" for _, q, is_exploit in self.servers if not is_exploit
        ]
        logging.info(f"Explout servers: {self._exploit_hostports}")
        logging.info(f"Blueprint servers: {self._bluprint_hostports}")

    @property
    def blueprint_agent_hostports(self) -> List[str]:
        return self._bluprint_hostports

    @property
    def exploit_agent_hostports(self) -> List[str]:
        return self._exploit_hostports


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
        logging.error(
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
        [
            tuple(decode_order_idxs(order_idxs[b, p]))
            for p in range(len(POWERS))
        ]
        for b in range(order_idxs.shape[0])
    ]
    return string_orders


def yield_rollouts(
    *,
    blueprint_hostports: Sequence[str],
    exploit_hostports: Sequence[str],
    game_json,
    temperature=0.05,
    max_rollout_length=40,
    batch_size=1,
    initial_power_index=0,
) -> Generator[ExploitRollout, None, None]:
    """Do non-stop rollout for 1 (exploit) vs 6 (blueprint).

    This method can safely be called in a subprocess

    Arguments:
    - hostport: string, "{host}:{port}" of model server
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

    blueprint_client_selector = create_client_selector(blueprint_hostports)
    exploit_client_selector = create_client_selector(exploit_hostports)

    faulthandler.register(signal.SIGUSR1)
    torch.set_num_threads(1)

    exploit_power_selector = itertools.cycle(tuple(range(len(POWERS))))
    for _ in range(initial_power_index):
        next(exploit_power_selector)

    for exploit_power_id in exploit_power_selector:
        with timings("setup"):
            games = [from_saved_game_format(game_json) for _ in range(batch_size)]
            turn_idx = 0
            observations = {i: [] for i in range(batch_size)}
            actions = {i: [] for i in range(batch_size)}
            logprobs = {i: [] for i in range(batch_size)}
            game_history = {i: [to_saved_game_format(game)] for i, game in enumerate(games)}

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
                (blueprint_batch_order_ids,) = do_model_request(
                    next(blueprint_client_selector), batch_inputs, temperature
                )
                (exploit_batch_order_ids, exploit_order_logprobs) = do_model_request(
                    next(exploit_client_selector), batch_inputs
                )

            with timings("merging"):
                # Using all orders from the blueprint model except for ones for the epxloit power.
                batch_order_idx = blueprint_batch_order_ids
                batch_order_idx[:, exploit_power_id] = exploit_batch_order_ids[:, exploit_power_id]

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
                        actions[i].append(exploit_batch_order_ids[inner_index, exploit_power_id])
                        observations[i].append([x[inner_index] for x in batch_inputs])
                        logprobs[i].append(exploit_order_logprobs[inner_index, exploit_power_id])
                        game_history[i].append(to_saved_game_format(game))
                        inner_index += 1

            turn_idx += 1

        logging.debug(
            f"end do_rollout pid {os.getpid()} for {batch_size} games in {turn_idx} turns. timings: "
            f"{ {k : float('{:.3}'.format(v)) for k, v in timings.items()} }."
        )
        for i in range(batch_size):
            yield ExploitRollout(
                power_id=exploit_power_id,
                game_jsons=game_history[i],
                actions=torch.stack(actions[i], 0),
                logprobs=torch.stack(logprobs[i], 0),
                observations=[torch.stack(obs, 0) for obs in zip(*observations[i])],
            )
