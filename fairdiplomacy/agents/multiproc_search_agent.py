import faulthandler
import logging
import json
import os
import signal
import time
from collections import Counter
from functools import partial
from typing import List, Tuple, Dict, Union, Sequence, Optional

import numpy as np
import postman
import torch
import torch.multiprocessing as mp

import pydipcc
from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    model_output_transform,
    average_score_dicts,
    filter_keys,
    are_supports_coordinated,
    safe_idx,
    n_move_phases_later,
    get_square_scores_from_game,
)
from fairdiplomacy.agents.dipnet_agent import encode_state, encode_inputs, decode_order_idxs
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.selfplay.ckpt_syncer import CkptSyncer
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state


class MultiprocSearchAgent(BaseSearchAgent):
    def __init__(
        self,
        *,
        model_path,
        max_batch_size,
        max_rollout_length=3,
        rollout_temperature,
        rollout_top_p=1.0,
        n_server_procs=1,
        n_gpu=1,
        n_rollout_procs=70,
        use_predicted_final_scores=True,
        postman_wait_till_full=False,
        use_server_addr=None,
        use_value_server_addr=None,
        device=None,
        mix_square_ratio_scoring=0,
        value_model_path=None,
        rollout_value_frac=0,
    ):
        super().__init__()
        self.n_rollout_procs = n_rollout_procs
        self.n_server_procs = n_server_procs
        self.use_predicted_final_scores = use_predicted_final_scores
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p
        self.max_batch_size = max_batch_size
        self.max_rollout_length = max_rollout_length
        self.mix_square_ratio_scoring = mix_square_ratio_scoring
        self.rollout_value_frac = rollout_value_frac
        device = int(device.lstrip("cuda:")) if type(device) == str else device

        logging.info("Launching servers")
        assert n_gpu <= n_server_procs and n_server_procs % n_gpu == 0
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            logging.warning("Failed mp.set_start_method")

        if use_server_addr is not None:
            assert value_model_path is None, "Not implemented"
            if n_server_procs != 1:
                raise ValueError(
                    f"Bad args use_server_addr={use_server_addr} n_server_procs={n_server_procs}"
                )
            self.hostports = [use_server_addr]
            self.value_hostport = use_value_server_addr
        else:
            _servers, _qs, self.hostports = zip(
                *[
                    make_server_process(
                        model_path=model_path,
                        device=i % n_gpu if device is None else device,
                        max_batch_size=max_batch_size,
                        wait_till_full=postman_wait_till_full,
                        # if torch's seed is set, then we want the server seed to be a deterministic
                        # function of that. Yet we don't want it to be the same for each agent.
                        # So pick a random number from the torch rng
                        seed=int(torch.randint(1000000000, (1,))),
                    )
                    for i in range(n_server_procs)
                ]
            )

            if value_model_path is not None:
                _, _, self.value_hostport = make_server_process(
                    model_path=value_model_path,
                    device=n_server_procs % n_gpu if device is None else device,
                    max_batch_size=max_batch_size,
                    wait_till_full=postman_wait_till_full,
                    seed=int(torch.randint(1000000000, (1,))),
                )
            else:
                self.value_hostport = None

        self.client = postman.Client(self.hostports[0])
        logging.info(f"Connecting to {self.hostports[0]} [{os.uname().nodename}]")
        self.client.connect(20)
        logging.info(f"Connected to {self.hostports[0]}")

        if n_rollout_procs > 0:
            self.proc_pool = mp.Pool(n_rollout_procs)
            logging.info("Warming up pool")
            self.proc_pool.map(float, range(n_rollout_procs))
            logging.info("Done warming up pool")
        else:
            logging.info("Debug mode: using fake process poll")
            fake_pool_class = type("FakePool", (), {"map": lambda self, *a, **k: map(*a, **k)})
            self.proc_pool = fake_pool_class()

    def do_model_request(
        self, x: DataFields, temperature: float, top_p: float, client: postman.Client = None
    ) -> Tuple[List[List[Tuple[str]]], np.ndarray]:
        """Synchronous request to model server

        Arguments:
        - x: a DataFields dict of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature
        - top_p: probability mass to samples from

        Returns:
        - a list (len = batch size) of lists (len=7) of order-tuples
        - [7] float32 array of estimated final scores
        """
        if client is None:
            client = self.client
        B = x["x_board_state"].shape[0]
        x["temperature"] = torch.zeros(B, 1).fill_(temperature)
        x["top_p"] = torch.zeros(B, 1).fill_(top_p)
        try:
            order_idxs, order_logprobs, final_scores = client.evaluate(x)
        except Exception:
            logging.error(
                "Caught server error with inputs {}".format(
                    [(k, v.shape, v.dtype) for k, v in x.items()]
                )
            )
            raise
        assert x["x_board_state"].shape[0] == final_scores.shape[0], (
            x["x_board_state"].shape[0],
            final_scores.shape[0],
        )
        if hasattr(final_scores, "numpy"):
            final_scores = final_scores.numpy()

        return (
            [
                [tuple(decode_order_idxs(order_idxs[b, p, :])) for p in range(len(POWERS))]
                for b in range(order_idxs.shape[0])
            ],
            order_logprobs,
            final_scores,
        )

    def distribute_rollouts(
        self, game, set_orders_dicts: List[Dict], average_n_rollouts=1, log_timings=False
    ) -> List[Tuple[Dict, Dict]]:
        """Run average_n_rollouts x len(set_orders_dicts) rollouts

        Arguments:
        - game: fairdiplomacy.Game
        - set_orders_dicts: List[Dict[power, orders]], each dict representing
          the orders to set for the first turn
        - average_n_rollouts: int, # of rollouts to run in parallel for each dict

        Returns: List[Tuple[order_dict, final_scores]], len=average_n_rollouts * len(set_orders_dicts)
            -> order_dict: Dict[power, orders],
               e.g. {"ITALY": ("A ROM H", "F NAP - ION", "A VEN H"), ...}
            -> final_scores: Dict[power, supply count],
               e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        """
        if type(game) == pydipcc.Game:
            game_json = game.to_json()
        else:
            game_json = json.dumps(game.to_saved_game_format())

        # divide up the rollouts among the processes
        all_results, all_timings = zip(
            *self.proc_pool.map(
                call,
                [
                    partial(
                        self.do_rollout,
                        game_json=game_json,
                        set_orders_dict=d,
                        hostport=self.hostports[i % self.n_server_procs],
                        value_hostport=self.value_hostport,
                        temperature=self.rollout_temperature,
                        top_p=self.rollout_top_p,
                        max_rollout_length=self.max_rollout_length,
                        batch_size=average_n_rollouts,
                        use_predicted_final_scores=self.use_predicted_final_scores,
                        mix_square_ratio_scoring=self.mix_square_ratio_scoring,
                        rollout_value_frac=self.rollout_value_frac,
                    )
                    for i, d in enumerate(set_orders_dicts)
                ],
            )
        )

        if log_timings:
            TimingCtx.pprint_multi(all_timings, logging.getLogger("timings").info)

        return [
            (order_dict, average_score_dicts(list_of_scores_dicts))
            for order_dict, list_of_scores_dicts in all_results
        ]

    @classmethod
    def do_rollout(
        cls,
        *,
        game_json,
        hostport,
        set_orders_dict={},
        temperature,
        top_p,
        max_rollout_length,
        batch_size=1,
        use_predicted_final_scores,
        mix_square_ratio_scoring=0,
        value_hostport=None,
        rollout_value_frac=0,
    ) -> Tuple[Tuple[Dict, List[Dict]], TimingCtx]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game string, e.g. output of to_saved_game_format(game)
        - hostport: string, "{host}:{port}" of model server
        - set_orders_dict: Dict[power, orders] to set for current turn
        - temperature: model softmax temperature for rollout policy
        - top_p: probability mass to samples from for rollout policy
        - max_rollout_length: return SC count after at most # steps
        - batch_size: rollout # of games in parallel
        - use_predicted_final_scores: if True, use model's value head for final SC predictions

        Returns a 2-tuple:
        - results, a 2-tuple:
          - set_orders_dict: Dict[power, orders]
          - list of Dict[power, final_score], len=batch_size
        - timings: a TimingCtx
        """
        timings = TimingCtx()

        with timings("postman.client"):
            client = postman.Client(hostport)
            client.connect(3)
            if value_hostport is not None:
                value_client = postman.Client(value_hostport)
                value_client.connect(3)
            else:
                value_client = client

        with timings("setup"):
            faulthandler.register(signal.SIGUSR2)
            torch.set_num_threads(1)

            games = [pydipcc.Game.from_json(game_json) for _ in range(batch_size)]
            for i in range(len(games)):
                games[i].game_id += f"_{i}"

            est_final_scores = {}  # game id -> np.array len=7

            # set orders if specified
            for power, orders in set_orders_dict.items():
                for game in games:
                    game.set_orders(power, list(orders))

            other_powers = [p for p in POWERS if p not in set_orders_dict]

        rollout_start_phase = games[0].current_short_phase
        rollout_end_phase = n_move_phases_later(rollout_start_phase, max_rollout_length)
        while True:
            if max_rollout_length == 0:
                # Handled separately.
                break

            # exit loop if all games are done before max_rollout_length
            ongoing_game_phases = [
                game.current_short_phase for game in games if not game.is_game_done
            ]
            if len(ongoing_game_phases) == 0:
                break

            # step games together at the pace of the slowest game, e.g. process
            # games with retreat phases alone before moving on to the next move phase
            min_phase = min(ongoing_game_phases, key=sort_phase_key)

            batch_data = []
            for game in games:
                if not game.is_game_done and game.current_short_phase == min_phase:
                    with timings("encode.all_poss_orders"):
                        all_possible_orders = game.get_all_possible_orders()
                    with timings("encode.state"):
                        encoded_state = encode_state(game)
                    with timings("encode.inputs"):
                        inputs = encode_inputs(
                            game, all_possible_orders=all_possible_orders, game_state=encoded_state
                        )
                    batch_data.append((game, inputs))

            with timings("cat_pad"):
                xs: List[Tuple] = [b[1] for b in batch_data]
                batch_inputs = self.cat_pad_inputs(xs)

            with timings("model"):
                if client != value_client:
                    assert (
                        rollout_value_frac == 0
                    ), "If separate value model, you can't add in value each step (slow)"

                cur_client = value_client if min_phase == rollout_end_phase else client

                batch_orders, _, batch_est_final_scores = self.do_model_request(
                    batch_inputs, temperature, top_p, client=cur_client
                )

            if min_phase == rollout_end_phase:
                with timings("score.accumulate"):
                    for game_idx, (game, _) in enumerate(batch_data):
                        est_final_scores[game.game_id] = np.array(batch_est_final_scores[game_idx])

                # skip env step and exit loop once we've accumulated the estimated
                # scores for all games up to max_rollout_length
                break

            with timings("env"):
                assert len(batch_data) == len(batch_orders), "{} != {}".format(
                    len(batch_data), len(batch_orders)
                )

                # set_orders and process
                assert len(batch_data) == len(batch_orders)
                for (game, _), power_orders in zip(batch_data, batch_orders):
                    if game.is_game_done:
                        continue
                    power_orders = dict(zip(POWERS, power_orders))
                    for other_power in other_powers:
                        game.set_orders(other_power, list(power_orders[other_power]))

                    assert game.current_short_phase == min_phase
                    game.process()

            for (game, _) in batch_data:
                if game.is_game_done:
                    with timings("score.gameover"):
                        final_scores = np.array(get_square_scores_from_game(game))
                        est_final_scores[game.game_id] = final_scores

            other_powers = POWERS  # no set orders on subsequent turns

        # out of rollout loop

        if max_rollout_length > 0:
            assert len(est_final_scores) == len(games)
        else:
            assert (
                not other_powers
            ), "If max_rollout_length=0 it's assumed that all orders are pre-defined."
            # All orders are set. Step env. Now only need to get values.
            game.process()

            batch_data = []
            for game in games:
                if not game.is_game_done:
                    with timings("encode.all_poss_orders"):
                        all_possible_orders = game.get_all_possible_orders()
                    with timings("encode.state"):
                        encoded_state = encode_state(game)
                    with timings("encode.inputs"):
                        inputs = encode_inputs(
                            game, all_possible_orders=all_possible_orders, game_state=encoded_state
                        )
                    batch_data.append((game, inputs))
                else:
                    est_final_scores[game.game_id] = get_square_scores_from_game(game)

            if batch_data:
                with timings("cat_pad"):
                    xs: List[Tuple] = [b[1] for b in batch_data]
                    batch_inputs = self.cat_pad_inputs(xs)

                with timings("model"):
                    _, _, batch_est_final_scores = self.do_model_request(
                        batch_inputs, temperature, top_p, client=value_client
                    )
                    assert batch_est_final_scores.shape[0] == len(batch_data)
                    assert batch_est_final_scores.shape[1] == len(POWERS)
                    for game_idx, (game, _) in enumerate(batch_data):
                        est_final_scores[game.game_id] = batch_est_final_scores[game_idx]

        with timings("final_scores"):
            # get GameScores objects for current game state
            current_game_scores = [
                {
                    p: compute_game_scores_from_state(i, game.get_state())
                    for i, p in enumerate(POWERS)
                }
                for game in games
            ]

            # get estimated or current sum of squares scoring
            final_game_scores = [
                dict(zip(POWERS, est_final_scores[game.game_id]))
                for game, current_scores in zip(games, current_game_scores)
            ]

            # mix in current sum of squares ratio to encourage losing powers to try hard
            if mix_square_ratio_scoring > 0:
                for game, final_scores, current_scores in zip(
                    games, final_game_scores, current_game_scores
                ):
                    for p in POWERS:
                        final_scores[p] = (1 - mix_square_ratio_scoring) * final_scores[p] + (
                            mix_square_ratio_scoring * current_scores[p].square_ratio
                        )
            result = (set_orders_dict, final_game_scores)

        return result, timings


def make_server_process(*, model_path, device, max_batch_size, wait_till_full, seed):
    q = mp.SimpleQueue()
    server = ExceptionHandlingProcess(
        target=run_server,
        kwargs=dict(
            port=0,
            batch_size=max_batch_size,
            load_model_fn=partial(
                load_dipnet_model,
                model_path,
                map_location=f"cuda:{device}" if torch.cuda.is_available() else "cpu",
                eval=True,
            ),
            output_transform=model_output_transform,
            seed=seed,
            device=device,
            port_q=q,
            wait_till_full=wait_till_full,
            model_path=model_path,  # debugging only
        ),
        daemon=True,
    )
    server.start()
    port = q.get()
    hostport = f"127.0.0.1:{port}"
    return server, q, hostport


def run_server(port, batch_size, port_q=None, **kwargs):
    def set_seed(seed):
        seed = seed.item()
        logging.info(f"Set server seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)
    max_port = port + 10
    try:
        logging.info(f"Starting server port={port} batch={batch_size}")
        eval_queue = postman.ComputationQueue(batch_size)
        for p in range(port, max_port):
            server = postman.Server(f"127.0.0.1:{p}")
            server.bind(
                "set_batch_size", lambda x: eval_queue.set_batch_size(x.item()), batch_size=1
            )
            server.bind("set_seed", set_seed, batch_size=1)
            server.bind_queue_batched("evaluate", eval_queue)
            try:
                server.run()
                break  # port is good
            except RuntimeError:
                continue  # try a different port
        else:
            raise RuntimeError(f"Couldn't start server on ports {port}:{max_port}")

        bound_port = server.port()
        assert bound_port != 0

        logging.info(f"Started server on port={bound_port} pid={os.getpid()}")
        if port_q is not None:
            port_q.put(bound_port)  # send port to parent proc

        server_handler(eval_queue, **kwargs)  # FIXME: try multiple threads?
    except Exception as e:
        logging.exception("Caught exception in the server (%s)", e)
        raise
    finally:
        eval_queue.close()
        server.stop()


def server_handler(
    q: postman.ComputationQueue,
    load_model_fn,
    seed,
    output_transform=None,
    device: Optional[int] = 0,
    ckpt_sync_path=None,
    ckpt_sync_every=0,
    wait_till_full=False,
    empty_cache=True,
    model_path=None,  # for debugging only
):

    if not torch.cuda.is_available() and device is not None:
        logging.warning(f"Cannot run on GPU {device} as not GPUs. Will do CPU")
        device = None
    if device is not None and device > 0:
        # device=[None] stands for CPU.
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info(f"Server {os.getpid()} loaded model, device={device}, seed={seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    frame_count, batch_count, total_batches = 0, 0, 0
    timings = TimingCtx()
    totaltic = time.time()

    if ckpt_sync_path is not None:
        ckpt_syncer = CkptSyncer(ckpt_sync_path)
        last_ckpt_version = ckpt_syncer.maybe_load_state_dict(model, last_version=None)
        if ckpt_sync_every:
            next_ckpt_sync_time = time.time() + ckpt_sync_every

    with torch.no_grad():
        while True:
            if empty_cache and device is not None:
                torch.cuda.empty_cache()
            try:
                with q.get(wait_till_full=wait_till_full) as batch:
                    with timings("ckpt_sync"):
                        if ckpt_sync_path is not None and (
                            not ckpt_sync_every or time.time() >= next_ckpt_sync_time
                        ):
                            last_ckpt_version = ckpt_syncer.maybe_load_state_dict(
                                model, last_ckpt_version
                            )
                            if ckpt_sync_every:
                                next_ckpt_sync_time = time.time() + ckpt_sync_every

                    with timings("next_batch"):
                        inputs = batch.get_inputs()[0]

                    with timings("to_cuda"):
                        if device is not None:
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    with timings("model"):
                        y = model(**inputs)

                    with timings("transform"):
                        if output_transform is not None:
                            y = output_transform(inputs, y)

                    with timings("to_cpu"):
                        y = tuple(x.to("cpu") for x in y)

                    with timings("reply"):
                        batch.set_outputs(y)

                    # Do some performance logging here
                    batch_count += 1
                    total_batches += 1
                    frame_count += inputs["x_board_state"].shape[0]
                    if total_batches > 16 and (total_batches & (total_batches - 1)) == 0:
                        delta = time.time() - totaltic
                        logging.info(
                            f"Server thread: performed {batch_count} forwards of avg batch size {frame_count / batch_count} "
                            f"in {delta} s, {frame_count / delta} forward/s."
                        )
                        TimingCtx.pprint_multi([timings], logging.info)
                        batch_count = frame_count = 0
                        timings.clear()
                        totaltic = time.time()

            except TimeoutError as e:
                logging.info("TimeoutError: %s", e)

    logging.info("SERVER DONE")

    @classmethod
    def cat_pad_inputs(cls, xs: List[DataFields]) -> DataFields:
        return cat_pad_inputs(xs)


# imported by RL code
def cat_pad_inputs(xs: List[DataFields]) -> DataFields:
    batch = DataFields({k: [x[k] for x in xs] for k in xs[0].keys()})
    for k, v in batch.items():
        if k == "x_possible_actions":
            batch[k] = cat_pad_sequences(v, pad_value=-1, pad_to_len=MAX_SEQ_LEN)
        elif k == "x_loc_idxs":
            batch[k] = cat_pad_sequences(v, pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN)
        else:
            batch[k] = torch.cat(v)

    return batch


def call(f):
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()
