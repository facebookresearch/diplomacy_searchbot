import faulthandler
import logging
import os
import signal
import time
from collections import Counter
from functools import partial
from typing import List, Tuple, Set, Dict

import diplomacy
import numpy as np
import postman
import torch
import torch.multiprocessing as mp
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import (
    encode_inputs,
    zero_inputs,
    encode_state,
    decode_order_idxs,
)
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.selfplay.ckpt_syncer import CkptSyncer
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences

if os.path.exists(diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH):
    try:
        os.remove(diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH)
    except FileNotFoundError:
        pass


class BaseSearchAgent(BaseAgent):
    def __init__(
        self,
        *,
        model_path,
        max_batch_size,
        max_rollout_length,
        rollout_temperature,
        n_server_procs=1,
        n_gpu=1,
        n_rollout_procs=70,
        use_predicted_final_scores=True,
        postman_wait_till_full=False,
        use_server_addr=None,
        device=None,
    ):
        super().__init__()

        self.n_rollout_procs = n_rollout_procs
        self.n_server_procs = n_server_procs
        self.use_predicted_final_scores = use_predicted_final_scores
        self.rollout_temperature = rollout_temperature
        self.max_batch_size = max_batch_size
        self.max_rollout_length = max_rollout_length
        device = int(device.lstrip("cuda:")) if type(device) == str else device

        logging.info("Launching servers")
        servers = []
        assert n_gpu <= n_server_procs and n_server_procs % n_gpu == 0
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            logging.warning("Failed mp.set_start_method")

        if use_server_addr is not None:
            if n_server_procs != 1:
                raise ValueError(
                    f"Bad args use_server_addr={use_server_addr} n_server_procs={n_server_procs}"
                )
            self.hostports = [use_server_addr]
        else:
            for i in range(n_server_procs):
                q = mp.SimpleQueue()
                device = i % n_gpu if device is None else device
                server = ExceptionHandlingProcess(
                    target=run_server,
                    kwargs=dict(
                        port=0,
                        batch_size=max_batch_size,
                        load_model_fn=partial(
                            load_dipnet_model, model_path, map_location=f"cuda:{device}", eval=True
                        ),
                        output_transform=model_output_transform,
                        seed=0,
                        device=device,
                        port_q=q,
                        wait_till_full=postman_wait_till_full,
                    ),
                    daemon=True,
                )
                server.start()
                servers.append((server, q))

            logging.info("Waiting for servers")
            ports = [q.get() for _, q in servers]
            logging.info(f"Servers started on ports {ports}")
            self.hostports = [f"127.0.0.1:{p}" for p in ports]

        self.client = postman.Client(self.hostports[0])
        logging.info(f"Connecting to {self.hostports[0]} [{os.uname().nodename}]")
        self.client.connect(20)
        logging.info(f"Connected to {self.hostports[0]}")

        self.proc_pool = mp.Pool(n_rollout_procs)
        logging.info("Warming up pool")
        self.proc_pool.map(float, range(n_rollout_procs))
        logging.info("Done warming up pool")

    @classmethod
    def do_model_request(cls, client, x, temperature) -> Tuple[List[List[Tuple[str]]], np.ndarray]:
        """Synchronous request to model server

        Arguments:
        - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature

        Returns:
        - a list (len = batch size) of lists (len=7) of order-tuples
        - [7] float32 array of estimated final scores
        """
        temp_array = torch.zeros(x[0].shape[0], 1).fill_(temperature)
        req = (*x, temp_array)
        try:
            order_idxs, final_scores = client.evaluate(req)
        except Exception:
            logging.error(
                "Caught server error with inputs {}".format([(x.shape, x.dtype) for x in req])
            )
            raise

        if hasattr(final_scores, "numpy"):
            final_scores = final_scores.numpy()

        return (
            [
                [tuple(decode_order_idxs(order_idxs[b, p, :])) for p in range(len(POWERS))]
                for b in range(order_idxs.shape[0])
            ],
            np.mean(final_scores, axis=0),
        )

    def get_plausible_orders(
        self, game, *, n=1000, temperature=0.5, limit=8, batch_size=500
    ) -> Dict[str, Set[Tuple[str]]]:
        assert n % batch_size == 0, f"{n}, {batch_size}"

        # trivial return case: all powers have at most `limit` actions
        orderable_locs = game.get_orderable_locations()
        if max(map(len, orderable_locs.values())) <= 1:
            all_orders = game.get_all_possible_orders()
            pow_orders = {
                p: all_orders[orderable_locs[p][0]] if orderable_locs[p] else [] for p in POWERS
            }
            if max((map(len, pow_orders.values()))) <= limit:
                return {p: set((x,) for x in orders) for p, orders in pow_orders.items()}

        # non-trivial return case: query model
        counters = {p: Counter() for p in POWERS}
        x = [encode_inputs(game)] * n
        for x_chunk in [x[i : i + batch_size] for i in range(0, n, batch_size)]:
            batch_inputs, seq_lens = cat_pad_inputs(x_chunk)
            batch_orders, _ = self.do_model_request(self.client, batch_inputs, temperature)
            batch_orders = list(zip(*batch_orders))
            for p, power in enumerate(POWERS):
                counters[power].update(batch_orders[p])

        logging.debug(
            "get_plausible_orders(n={}, t={}) found {} unique sets, choosing top {}".format(
                n, temperature, list(map(len, counters)), limit
            )
        )
        return {
            power: set([orders for orders, _ in counter.most_common(limit)])
            for power, counter in counters.items()
        }

    def distribute_rollouts(
        self, game, set_orders_dicts: List[Dict], N
    ) -> List[Tuple[Dict, Dict]]:
        """Run N x len(set_orders_dicts) rollouts

        Arguments:
        - game: diplomacy.Game
        - set_orders_dicts: List[Dict[power, orders]], each dict representing
          the orders to set for the first turn
        - N: int

        Returns: List[Tuple[order_dict, final_scores]], where
            -> order_dict: Dict[power, orders],
               e.g. {"ITALY": ("A ROM H", "F NAP - ION", "A VEN H"), ...}
            -> final_scores: Dict[power, supply count],
               e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        """
        game_json = to_saved_game_format(game)

        # divide up the rollouts among the processes
        procs_per_order = max(1, self.n_rollout_procs // len(set_orders_dicts))
        logging.info(f"num_orders={len(set_orders_dicts)} , procs_per_order={procs_per_order}")
        batch_sizes = [len(x) for x in torch.arange(N).chunk(procs_per_order) if len(x) > 0]
        logging.info(f"procs_per_order={procs_per_order} , batch_sizes={batch_sizes}")
        all_results, all_timings = zip(
            *self.proc_pool.map(
                call,
                [
                    partial(
                        self.do_rollout,
                        game_json=game_json,
                        set_orders_dict=d,
                        hostport=self.hostports[i % self.n_server_procs],
                        temperature=self.rollout_temperature,
                        max_rollout_length=self.max_rollout_length,
                        batch_size=batch_size,
                        use_predicted_final_scores=self.use_predicted_final_scores,
                    )
                    for d in set_orders_dicts
                    for i, batch_size in enumerate(batch_sizes)
                ],
            )
        )
        logging.info(
            "Timings[avg.do_rollout, n={}, len={}] {}".format(
                len(all_timings), self.max_rollout_length, sum(all_timings) / len(all_timings)
            )
        )
        return [
            (order_dict, scores)
            for order_dict, list_of_scores_dicts in all_results
            for scores in list_of_scores_dicts
        ]

    @classmethod
    def do_rollout(
        cls,
        *,
        game_json,
        hostport,
        set_orders_dict={},
        temperature,
        max_rollout_length,
        batch_size=1,
        use_predicted_final_scores,
    ) -> Tuple[Tuple[Dict, List[Dict]], TimingCtx]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
        - hostport: string, "{host}:{port}" of model server
        - set_orders_dict: Dict[power, orders] to set for current turn
        - temperature: model softmax temperature for rollout policy
        - max_rollout_length: return SC count after at most # steps
        - batch_size: rollout # of games in parallel
        - use_predicted_final_scores: if True, use model's value head for final SC predictions

        Returns a 2-tuple:
        - results, a 2-tuple:
          - set_orders_dict: Dict[power, orders]
          - list of Dict[power, final_score]
        - timings: a TimingCtx
        """
        timings = TimingCtx()

        with timings("postman.client"):
            client = postman.Client(hostport)
            client.connect(3)

        with timings("setup"):
            faulthandler.register(signal.SIGUSR2)
            torch.set_num_threads(1)

            games = [from_saved_game_format(game_json) for _ in range(batch_size)]
            for i in range(len(games)):
                games[i].game_id += f"_{i}"
            est_final_scores = {}

            # set orders if specified
            for power, orders in set_orders_dict.items():
                for game in games:
                    game.set_orders(power, list(orders))

            other_powers = [p for p in POWERS if p not in set_orders_dict]

        for turn_idx in range(max_rollout_length):
            with timings("prep"):
                batch_data = []
                for game in games:
                    if game.is_game_done:
                        inputs = zero_inputs()
                    else:
                        inputs = encode_inputs(
                            game,
                            all_possible_orders=game.get_all_possible_orders(),  # expensive
                            game_state=encode_state(game),
                        )
                    batch_data.append((game, inputs))

            with timings("cat_pad"):
                xs: List[Tuple] = [b[1] for b in batch_data]
                batch_inputs, seq_lens = cat_pad_inputs(xs)

            with timings("model"):
                batch_orders, final_scores = cls.do_model_request(
                    client, batch_inputs, temperature
                )
            with timings("final_scores"):
                for game in games:
                    if not game.is_game_done:
                        est_final_scores[game.game_id] = final_scores

            with timings("env"):
                assert len(batch_data) == len(batch_orders), "{} != {}".format(
                    len(batch_data), len(batch_orders)
                )

                # set_orders and process
                for (game, _), power_orders in zip(batch_data, batch_orders):
                    if game.is_game_done:
                        continue
                    power_orders = dict(zip(POWERS, power_orders))
                    for other_power in other_powers:
                        game.set_orders(other_power, list(power_orders[other_power]))

                for game in games:
                    if not game.is_game_done:
                        game.process()

            other_powers = POWERS  # no set orders on subsequent turns

        with timings("final_scores"):
            final_scores = [
                {k: len(v) for k, v in game.get_state()["centers"].items()}
                if game.is_game_done or not use_predicted_final_scores
                else dict(zip(POWERS, est_final_scores[game.game_id]))
                for game in games
            ]
            result = (set_orders_dict, final_scores)

        return result, timings


def run_server(port, batch_size, port_q=None, **kwargs):
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
    finally:
        eval_queue.close()
        server.stop()


def server_handler(
    q: postman.ComputationQueue,
    load_model_fn,
    output_transform=None,
    seed=None,
    device=0,
    ckpt_sync_path=None,
    wait_till_full=False,
    empty_cache=True,
):

    if device > 0:
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info(f"Server {os.getpid()} loaded model, device={device}")

    if seed is not None:
        torch.manual_seed(seed)

    frame_count, batch_count, total_batches = 0, 0, 0
    timings = TimingCtx()
    totaltic = time.time()

    ckpt_sync_every = 0  # Setting timeout in order not to handle with FS too often.
    if ckpt_sync_path is not None:
        ckpt_syncer = CkptSyncer(ckpt_sync_path)
        last_ckpt_version = ckpt_syncer.maybe_load_state_dict(model, last_version=None)
        if ckpt_sync_every:
            next_ckpt_sync_time = time.time() + ckpt_sync_every

    with torch.no_grad():
        while True:
            if empty_cache:
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
                        inputs = tuple(x.to("cuda") for x in inputs)

                    with timings("model"):
                        y = model(*inputs)

                    with timings("transform"):
                        if output_transform is not None:
                            y = output_transform(y)

                    with timings("to_cpu"):
                        y = tuple(x.to("cpu") for x in y)

                    with timings("reply"):
                        batch.set_outputs(y)

                    # Do some performance logging here
                    batch_count += 1
                    total_batches += 1
                    frame_count += inputs[0].shape[0]
                    if (total_batches & (total_batches - 1)) == 0:
                        delta = time.time() - totaltic
                        logging.info(
                            f"Performed {batch_count} forwards of avg batch size {frame_count / batch_count} "
                            f"in {delta} s, {frame_count / delta} forward/s."
                        )
                        logging.info(f"Timings[server] {str(timings)}")
                        batch_count = frame_count = 0
                        timings.clear()
                        totaltic = time.time()

            except TimeoutError as e:
                logging.info("TimeoutError:", e)

    logging.info("SERVER DONE")


def cat_pad_inputs(xs):
    batch = list(zip(*xs))

    # first cat_pad_sequences on sequence inputs
    padded_loc_idxs_seqs, _ = cat_pad_sequences(batch[-2], pad_value=-1, pad_to_len=MAX_SEQ_LEN)

    padded_mask_seqs, seq_lens = cat_pad_sequences(
        batch[-1], pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN
    )

    # then cat all tensors
    batch_inputs = [torch.cat(ts) for ts in batch[:-2]] + [padded_loc_idxs_seqs, padded_mask_seqs]
    return batch_inputs, seq_lens


def model_output_transform(y):
    # return only order_idxs, final_scores
    return y[0], y[-1]


def call(f):
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()


def unpad_lists(lists, lens):
    return [lst[:length] for lst, length in zip(lists, lens)]
