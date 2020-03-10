import faulthandler
import logging
import os
import signal
import sys
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
from fairdiplomacy.agents.dipnet_agent import encode_inputs, encode_state, ORDER_VOCABULARY
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess

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
        n_rollout_procs,
        n_server_procs,
        n_gpu,
        max_batch_size,
        use_predicted_final_scores=True,
    ):
        super().__init__()

        self.n_rollout_procs = n_rollout_procs
        self.n_server_procs = n_server_procs
        self.use_predicted_final_scores = use_predicted_final_scores

        logging.info("Launching servers")
        self.servers = []
        assert n_gpu <= n_server_procs and n_server_procs % n_gpu == 0
        mp.set_start_method("spawn")
        for i in range(n_server_procs):
            q = mp.SimpleQueue()
            server = ExceptionHandlingProcess(
                target=run_server,
                kwargs=dict(
                    port=0,
                    batch_size=max_batch_size,
                    load_model_fn=partial(
                        load_dipnet_model, model_path, map_location="cuda", eval=True
                    ),
                    output_transform=model_output_transform,
                    seed=0,
                    device=i % n_gpu,
                    port_q=q,
                ),
                daemon=True,
            )
            server.start()
            self.servers.append((server, q))

        logging.info("Waiting for servers")
        ports = [q.get() for _, q in self.servers]
        self.hostports = [f"127.0.0.1:{p}" for p in ports]
        logging.info(f"Servers started on ports {ports}")

        self.client = postman.Client(self.hostports[0])
        logging.info(f"Connecting to {self.hostports[0]} [{os.uname().nodename}]")
        self.client.connect(20)
        logging.info(f"Connected to {self.hostports[0]}")

        self.proc_pool = mp.Pool(n_rollout_procs)
        logging.info("Warming up pool")
        self.proc_pool.map(float, range(n_rollout_procs))
        logging.info("Done warming up pool")

    @classmethod
    def do_model_request(cls, client, x, temperature=1.0) -> Tuple[List[Tuple[str]], np.ndarray]:
        """Synchronous request to model server

        Arguments:
        - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature

        Returns:
        - a list (len = batch size) of order-sets
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
                tuple(ORDER_VOCABULARY[idx] for idx in order_idxs[i, :])
                for i in range(order_idxs.shape[0])
            ],
            np.mean(final_scores, axis=0),
        )

    def get_plausible_orders(
        self, game, power, n=1000, temperature=0.5, limit=8, batch_size=100
    ) -> Set[Tuple[str]]:
        # check for trivial return case
        orderable_locs = game.get_orderable_locations(power)
        if len(orderable_locs) == 0:
            return set()
        elif len(orderable_locs) == 1:
            possible_orders = game.get_all_possible_orders()[orderable_locs[0]]
            if len(possible_orders) <= limit:
                return set((x,) for x in possible_orders)

        # non-trivial return: query model
        generated_orders = []
        x = [encode_inputs(game, power)] * n
        for x_chunk in [x[i : i + batch_size] for i in range(0, n, batch_size)]:
            batch_inputs, seq_lens = cat_pad_inputs(x_chunk)
            generated_orders += unpad_lists(
                self.do_model_request(self.client, batch_inputs, temperature)[0], seq_lens
            )
        unique_orders = Counter(generated_orders)

        logging.debug(
            "get_plausible_orders(n={}, t={}) found {} unique sets, choosing top {}".format(
                n, temperature, len(unique_orders), limit
            )
        )
        return set([orders for orders, _ in unique_orders.most_common(limit)])

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
        all_results = self.proc_pool.map(
            call,
            [
                partial(
                    self.do_rollout,
                    game_json=game_json,
                    set_orders_dict=d,
                    hostport=self.hostports[i % self.n_server_procs],
                    temperature=0.05,
                    max_rollout_length=self.max_rollout_length,
                    batch_size=batch_size,
                    use_predicted_final_scores=self.use_predicted_final_scores,
                )
                for d in set_orders_dicts
                for i, batch_size in enumerate(batch_sizes)
            ],
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
        temperature=0.05,
        max_rollout_length=40,
        batch_size=1,
        use_predicted_final_scores=True,
    ) -> Tuple[Dict, List[Dict]]:
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
        - set_orders_dict: Dict[power, orders]
        - list of Dict[power, final_score]
        """
        assert batch_size <= 8
        timings = TimingCtx()

        with timings("postman.client"):
            client = postman.Client(hostport)
            client.connect(3)

        with timings("setup"):
            faulthandler.register(signal.SIGUSR1)
            torch.set_num_threads(1)

            games = [from_saved_game_format(game_json) for _ in range(batch_size)]
            est_final_scores = {}

            # set orders if specified
            for power, orders in set_orders_dict.items():
                for game in games:
                    game.set_orders(power, list(orders))

            turn_idx = 0
            other_powers = [p for p in POWERS if p not in set_orders_dict]

        while not all(game.is_game_done for game in games) and turn_idx < max_rollout_length:
            with timings("prep"):
                batch_data = []
                for game in games:
                    if game.is_game_done:
                        continue
                    all_possible_orders = game.get_all_possible_orders()  # expensive
                    game_state = encode_state(game)
                    batch_data += [
                        (
                            game,
                            p,
                            encode_inputs(
                                game,
                                p,
                                all_possible_orders=all_possible_orders,
                                game_state=game_state,
                            ),
                        )
                        for p in other_powers
                        if game.get_orderable_locations(p)
                    ]

            if len(batch_data) > 0:
                with timings("cat_pad"):
                    xs: List[Tuple] = [b[2] for b in batch_data]
                    batch_inputs, seq_lens = cat_pad_inputs(xs)

                with timings("model"):
                    batch_orders, final_scores = cls.do_model_request(
                        client, batch_inputs, temperature
                    )
                    batch_orders = unpad_lists(batch_orders, seq_lens)
                with timings("final_scores"):
                    est_final_scores[game.game_id] = final_scores
            else:
                batch_orders = []

            with timings("env"):
                assert len(batch_data) == len(batch_orders), "{} != {}".format(
                    len(batch_data), len(batch_orders)
                )

                # set_orders and process
                for (game, other_power, _), orders in zip(batch_data, batch_orders):
                    game.set_orders(other_power, list(orders))
                for game in games:
                    if not game.is_game_done:
                        game.process()

            turn_idx += 1
            other_powers = POWERS  # no set orders on subsequent turns

        with timings("final_scores"):
            final_scores = [
                {k: len(v) for k, v in game.get_state()["centers"].items()}
                if game.is_game_done or not use_predicted_final_scores
                else dict(zip(POWERS, est_final_scores[game.game_id]))
                for game in games
            ]
            result = (set_orders_dict, final_scores)
        logging.debug(
            f"end do_rollout pid {os.getpid()} for {batch_size} games in {turn_idx} turns. timings: "
            f"{ {k : float('{:.3}'.format(v)) for k, v in timings.items()} }."
        )
        return result


def run_server(port, batch_size, port_q=None, **kwargs):
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)
    max_port = port + 10
    try:
        logging.info(f"Starting server port={port} batch={batch_size}")
        eval_queue = postman.ComputationQueue(batch_size)
        for p in range(port, max_port):
            server = postman.Server(f"127.0.0.1:{p}")
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
    q: postman.ComputationQueue, load_model_fn, output_transform=None, seed=None, device=0
):

    if device != 0:
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info(f"Server {os.getpid()} loaded model")

    if seed is not None:
        torch.manual_seed(seed)

    frame_count, batch_count, total_batches = 0, 0, 0
    timings = TimingCtx()
    totaltic = time.time()
    with torch.no_grad():
        while True:
            try:
                with q.get() as batch:

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
                        logging.info(str(timings))
                        batch_count = frame_count = 0
                        timings.clear()
                        totaltic = time.time()

            except TimeoutError as e:
                logging.info("TimeoutError:", e)

    logging.info("SERVER DONE")


def cat_pad_sequences(tensors, pad_value=0, pad_to_len=None):
    """
    Arguments:
    - tensors: a list of [B x S x ...] formatted tensors
    - pad_value: the value used to fill padding
    - pad_to_len: the desired total length. If none, use the longest sequence.

    Returns:
    - the result of torch.cat(tensors, dim=0) where each sequence has been
      padded to pad_to_len or the largest S
    - a list of S values for each tensor
    """
    seq_lens = [t.shape[1] for t in tensors]
    max_len = max(seq_lens) if pad_to_len is None else pad_to_len

    padded = [
        torch.cat(
            [
                t,
                torch.zeros(t.shape[0], max_len - t.shape[1], *t.shape[2:], dtype=t.dtype).fill_(
                    pad_value
                ),
            ],
            dim=1,
        )
        if t.shape[1] < max_len
        else t
        for t in tensors
    ]
    return torch.cat(padded, dim=0), seq_lens


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
    return y[0], y[2]


def call(f):
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()


def unpad_lists(lists, lens):
    return [lst[:length] for lst, length in zip(lists, lens)]
