
import copy
import faulthandler
import itertools
import logging
import os
import signal
import sys

import time
import torch

import torch.multiprocessing as mp

from collections import Counter, defaultdict

from concurrent.futures import ProcessPoolExecutor
import diplomacy.utils.export
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from functools import partial
from typing import List, Tuple, Set, Dict
import torchbeast
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import encode_inputs, encode_state, ORDER_VOCABULARY
from fairdiplomacy.models.consts import MAX_SEQ_LEN
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.serving import ModelServer, ModelClient

if os.path.exists(diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH):
    logging.warn(f"You should delete {diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH}" +
                 f"for faster startup time!")


def gen_hostport(host="localhost", port=12345):
    import socket
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:  # socket already exists
                port += 1
            else:
                return f"{host}:{port}"


def call(f):
    """Helper to be able to do pool.map(call, [partial(f, foo=42)])

    Using pool.starmap(f, [(42,)]) is shorter, but it doesn't support keyword
    arguments. It appears going through partial is the only way to do that.
    """
    return f()


def div_round_up(A, B):
    return (A + B - 1) // B


def server_handler(q: torchbeast.ComputationQueue,
                   load_model_fn,
                   output_transform=None,
                   seed=None,
                   device=0):

    logging.info(f"STARTED SERVER! {os.getpid()}")
    if device != 0:
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info("LOADED MODEL")

    if seed is not None:
        torch.manual_seed(seed)

    frame_count, batch_count, total_batches = 0, 0, 0
    timings = defaultdict(float)
    jit_model = None
    totaltic = time.time()
    with torch.no_grad():
        while True:
            logging.info("Waiting for batch")
            try:
                tic = time.time()
                for batch in q:
                    # logging.info("Goot batch")
                    inputs = batch.get_inputs()[0]
                    timings['next_batch'] += time.time() - tic; tic = time.time()

                    inputs = tuple(x.to("cuda") for x in inputs)
                    timings['to_cuda'] += time.time() - tic; tic = time.time()

                    y = model(*inputs) # jit_model(*batch)
                    timings['model'] += time.time() - tic; tic = time.time()
                    if output_transform is not None:
                        y = output_transform(y)
                    timings['transform'] += time.time() - tic; tic = time.time()
                    y = tuple(x.to("cpu") for x in y)
                    timings['to_cpu'] += time.time() - tic; tic = time.time()
                    batch.set_outputs(y)
                    timings['reply'] += time.time() - tic; tic = time.time()

                    # Do some performance logging here
                    batch_count += 1
                    total_batches += 1
                    frame_count += inputs[0].shape[0]
                    if (total_batches & (total_batches - 1)) == 0:
                        delta = time.time() - totaltic
                        logging.info(f"Performed {batch_count} forwards of avg batch size {frame_count / batch_count} "
                                     f"in {delta} s, {frame_count / delta} forward/s.")
                        logging.info(str(timings))
                        batch_count = frame_count = 0
                        timings.clear()
                        totaltic = time.time()
            except TimeoutError as e:
                logging.info("TimeoutError:", e)

    print("SERVER DONE")


def run_server(hostport, batch_size, **kwargs):
    logging.info(f"STARTED SERVER on {hostport} batch= {batch_size}")
    server = torchbeast.Server(hostport)
    eval_queue = torchbeast.ComputationQueue(batch_size)
    server.bind_queue("evaluate", eval_queue)
    server.run()
    server_handler(eval_queue, **kwargs)  # FIXME: try multiple threads?


class SimpleSearchDipnetAgent(BaseAgent):
    """One-ply search with dipnet-policy rollouts

    ## Policy
    1. Consider a set of orders that are suggested by the dipnet policy network.
    2. For each set of orders, perform a number of rollouts using the dipnet
    policy network for each power.
    3. Score each order set by the average supply center count at the end
    of the rollout.
    4. Choose the order set with the highest score.

    ## Implementation Details
    - __init__ forks some number of server processes running a ModelServer instance
      listening on different ports, and a ProcesssPoolExecutor for rollouts
    - get_orders first gets plausible orders to search through, then launches
      rollouts for each plausible order via the proc pool
    """

    def __init__(
        self,
        model_path,
        n_rollout_procs=70,
        n_server_procs=1,
        n_gpu=1,
        max_batch_size=1000,
        rollouts_per_plausible_order=10,
        max_rollout_length=20,
    ):
        super().__init__()

        self.rollouts_per_plausible_order = rollouts_per_plausible_order
        self.max_rollout_length = max_rollout_length
        self.hostports = [gen_hostport() for _ in range(n_server_procs)]
        self.servers = []
        assert n_gpu <= n_server_procs and n_server_procs % n_gpu == 0
        for i in range(n_server_procs):
            server = mp.Process(
                target=run_server,
                kwargs=dict(
                    hostport=self.hostports[i],
                    batch_size=max_batch_size,
                    load_model_fn=partial(load_dipnet_model, model_path, map_location="cuda", eval=True),
                    output_transform=model_output_transform,
                    seed=0,
                    device=i % n_gpu,
                ),
                daemon=True,
            )
            server.start()
            self.servers.append(server)

        self.client = torchbeast.Client(self.hostports[0])
        logging.info(f"Connecting to {self.hostports[0]}")

        self.client.connect(20)
        logging.info(f"Connected to {self.hostports[0]}")
        self.n_server_procs = n_server_procs
        self.n_rollout_procs = n_rollout_procs
        self.proc_pool = mp.Pool(n_rollout_procs)
        logging.info("Warming up pool")
        self.proc_pool.map(float, range(n_rollout_procs))
        logging.info("Done warming up pool")


    def get_orders(self, game, power) -> List[str]:
        plausible_orders = self.get_plausible_orders(game, power)
        logging.info("Plausible orders: {}".format(plausible_orders))
        game_json = to_saved_game_format(game)

        # divide up the rollouts among the processes
        procs_per_order = max(1, self.n_rollout_procs // len(plausible_orders))
        logging.info(f"num_plausible_orders= {len(plausible_orders)} , procs_per_order= {procs_per_order}")
        batch_sizes = [len(x) for x in torch.arange(self.rollouts_per_plausible_order).chunk(procs_per_order) if len(x) > 0]
        logging.info(f"procs_per_order= {procs_per_order} , batch_sizes= {batch_sizes}")
        results = self.proc_pool.map(call,
                [partial(self.do_rollout,
                    game_json=game_json,
                    set_orders_dict={power: orders},
                    hostport=self.hostports[i % self.n_server_procs],
                    max_rollout_length=self.max_rollout_length,
                    batch_size=batch_size,
                 )
            for orders in plausible_orders
            for i, batch_size in enumerate(batch_sizes)
        ])
        results = [(order_dict[power], scores) for result in results for (order_dict, scores) in result]

        return self.best_order_from_results(results, power)

    @classmethod
    def best_order_from_results(cls, results, power) -> List[str]:
        """Given a set of rollout results, choose the move to play

        Arguments:
        - results: Tuple[orders, all_scores], where
            -> orders: a complete set of orders, e.g. ("A ROM H", "F NAP - ION", "A VEN H")
            -> all_scores: Dict[power, supply count], e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        - power: the power making the orders, e.g. "ITALY"

        Returns:
        - the "orders" with the highest average score
        """
        order_scores = Counter()
        order_counts = Counter()

        for orders, all_scores in results:
            order_scores[orders] += all_scores[power]
            order_counts[orders] += 1

        order_avg_score = {
            orders: order_scores[orders] / order_counts[orders] for orders in order_scores
        }
        logging.info("order_avg_score: {}".format(order_avg_score))
        return max(order_avg_score.items(), key=lambda kv: kv[1])[0]

    @classmethod
    def do_model_request(cls, client, x, temperature=1.0) -> List[Tuple[str]]:
        """Synchronous request to model server

        Arguments:
        - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature

        Returns:
        - a list (len = batch size) of order-sets
        """
        temp_array = torch.zeros(x[0].shape[0], 1).fill_(temperature)
        req = (*x, temp_array)
        if req[0].shape[0] > 30:
            logging.info(f"Req batch size: {req[0].shape[0]}")
        try:
            order_idxs, = client.evaluate(req)
        except:
            e = sys.exc_info()[0]
            print("EXCEPTION", e)
            print([(x.shape, x.dtype) for x in req])
            raise

        # order_idxs, = model_client.synchronous_request([*x, temp_array])
        return [
            tuple(ORDER_VOCABULARY[idx] for idx in order_idxs[i, :])
            for i in range(order_idxs.shape[0])
        ]

    def get_plausible_orders(self, game, power, n=100, temperature=0.5) -> Set[Tuple[str]]:
        generated_orders = []
        # FIXME FIXME FIXME
        x = [encode_inputs(game, power)] * n
        for x_chunk in [x[i:i+10] for i in range(0, len(x), 10)]:
            batch_inputs, _seq_lens = cat_pad_inputs(x_chunk)
            # print(batch_inputs)
            # x = [t.repeat([n] + ([1] * (len(t.shape) - 1))) for t in x]
            generated_orders += self.do_model_request(self.client, batch_inputs, temperature)
        unique_orders = set(generated_orders)
        logging.debug(f"Generated {len(generated_orders)} sets of orders; found {len(unique_orders)} unique sets.")
        return unique_orders

    @classmethod
    def do_rollout(
        cls,
        game_json,
        hostport,
        set_orders_dict={},
        temperature=0.05,
        max_rollout_length=40,
        batch_size=1,
    ) -> Dict[str, float]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
        - set_orders_dict: Dict[power, orders] to set for current turn
        - temperature: model softmax temperature for rollout policy
        - model_server_port: port on which the batching model server is listening

        Returns a Dict[power, final supply count]
        """
        assert batch_size <= 8
        client = torchbeast.Client(hostport)
        client.connect(3)
        timings = defaultdict(float)
        tic = time.time()
        logging.info("new proc {} -- {}".format(set_orders_dict, os.getpid()))
        faulthandler.register(signal.SIGUSR1)
        torch.set_num_threads(1)

        games = [from_saved_game_format(game_json) for _ in range(batch_size)]
        # model_client = ModelClient(port=model_server_port)

        # set orders if specified
        for power, orders in set_orders_dict.items():
            for game in games:
                game.set_orders(power, list(orders))
        all_powers = list(games[0].powers.keys())  # assumption: this is constant

        turn_idx = 0
        other_powers = [p for p in all_powers if p not in set_orders_dict]
        timings['setup'] = time.time() - tic; tic = time.time()
        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()
        while not all(game.is_game_done for game in games) and turn_idx < max_rollout_length:
            timings['prep0'] += time.time() - tic; tic = time.time()

            batch_data = []
            for game in games:
                if game.is_game_done:
                    continue
                all_possible_orders = game.get_all_possible_orders()  # this is expensive
                game_state = encode_state(game)
                batch_data += [(game, p, encode_inputs(game, p, all_possible_orders=all_possible_orders, game_state=game_state))
                               for p in other_powers]

            xs: List[Tuple] = [b[2] for b in batch_data]
            batch_inputs, seq_lens = cat_pad_inputs(xs)

            timings['prep'] += time.time() - tic; tic = time.time()


            # get orders
            batch_orders = cls.do_model_request(client, batch_inputs, temperature)
            timings['model'] += time.time() - tic; tic = time.time()

            # process turn
            assert len(batch_data) == len(batch_orders), f"{len(batch_data)} != {len(batch_orders)}"
            for (game, other_power, _), orders, seq_len in zip(batch_data, batch_orders, seq_lens):
                game.set_orders(other_power, list(orders[:seq_len]))

            for game in games:
                if not game.is_game_done:
                    game.process()
            timings['env'] += time.time() - tic; tic = time.time()

            turn_idx += 1
            other_powers = all_powers  # no set orders on ssubsequent turns

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        # return avg supply counts for each power
        result = [(set_orders_dict, {k: len(v) for k, v in game.get_state()["centers"].items()}) for game in games]
        logging.info(f"end do_rollout pid {os.getpid()} for {batch_size} games in {turn_idx} turns. timings: "
                     f"{ {k : float('{:.3}'.format(v)) for k, v in timings.items()} }.")
        return result


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
    # FIXME: move to utils
    batch = list(zip(*xs))
    padded_loc_idxs_seqs, _ = cat_pad_sequences(
        batch[-2], pad_value=-1, pad_to_len=MAX_SEQ_LEN
    )
    padded_mask_seqs, seq_lens = cat_pad_sequences(
        batch[-1], pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN
    )

    batch_inputs = [torch.cat(ts) for ts in batch[:-2]] + [
        padded_loc_idxs_seqs,
        padded_mask_seqs,
    ]
    return batch_inputs, seq_lens


def model_output_transform(y):
    # return only order_idxs, not order_scores
    return y[:1]


if __name__ == "__main__":
    import diplomacy
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)
    logging.info("PID: {}".format(os.getpid()))

    MODEL_PTH = "/checkpoint/jsgray/dipnet.pth"
    game = diplomacy.Game()

    mp.set_start_method("spawn")
    agent = SimpleSearchDipnetAgent(MODEL_PTH)
    logging.info("Constructed agent")
    logging.info("Warmup: {}".format(agent.get_orders(game, "ITALY")))

    tic = time.time()
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))

    logging.info(f"Performed all rollouts for four searches in {time.time() - tic} s")
