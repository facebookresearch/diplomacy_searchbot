import faulthandler
import logging
import os
import signal
import time
from collections import Counter
from functools import partial
from typing import List, Tuple, Set, Dict, Union, Sequence

import diplomacy
import numpy as np
import postman
import torch
import torch.multiprocessing as mp

from fairdiplomacy.game import Game
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
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state

if os.path.exists(diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH):
    try:
        os.remove(diplomacy.utils.convoy_paths.EXTERNAL_CACHE_PATH)
    except FileNotFoundError:
        pass


def make_server_process(*, model_path, device, max_batch_size, wait_till_full):
    q = mp.SimpleQueue()
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
            wait_till_full=wait_till_full,
            model_path=model_path,  # debugging only
        ),
        daemon=True,
    )
    server.start()
    port = q.get()
    hostport = f"127.0.0.1:{port}"
    return server, q, hostport


class BaseSearchAgent(BaseAgent):
    def __init__(
        self,
        *,
        model_path,
        max_batch_size,
        max_rollout_length,
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
                )
            else:
                self.value_hostport = None

        self.client = postman.Client(self.hostports[0])
        logging.info(f"Connecting to {self.hostports[0]} [{os.uname().nodename}]")
        self.client.connect(20)
        logging.info(f"Connected to {self.hostports[0]}")

        self.proc_pool = mp.Pool(n_rollout_procs)
        logging.info("Warming up pool")
        self.proc_pool.map(float, range(n_rollout_procs))
        logging.info("Done warming up pool")

    @classmethod
    def do_model_request(
        cls, client, x, temperature: float, top_p: float
    ) -> Tuple[List[List[Tuple[str]]], np.ndarray]:
        """Synchronous request to model server

        Arguments:
        - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature
        - top_p: probability mass to samples from

        Returns:
        - a list (len = batch size) of lists (len=7) of order-tuples
        - [7] float32 array of estimated final scores
        """
        temp_array = torch.zeros(x[0].shape[0], 1).fill_(temperature)
        top_p_array = torch.zeros(x[0].shape[0], 1).fill_(top_p)
        req = (*x, temp_array, top_p_array)
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
        self,
        game,
        *,
        n=1000,
        temperature=0.6,
        limit: Union[int, Sequence[int]],  # limit, or list of limits per power
        batch_size=500,
        top_p=1.0,
    ) -> Dict[str, Set[Tuple[str]]]:
        assert n % batch_size == 0, f"{n}, {batch_size}"

        # limits is a list of 7 limits
        limits = [limit] * 7 if type(limit) == int else limit
        assert len(limits) == 7
        del limit

        # trivial return case: all powers have at most `limit` actions
        orderable_locs = game.get_orderable_locations()
        if max(map(len, orderable_locs.values())) <= 1:
            all_orders = game.get_all_possible_orders()
            pow_orders = {
                p: all_orders[orderable_locs[p][0]] if orderable_locs[p] else [] for p in POWERS
            }
            if all(len(pow_orders[p]) <= limit for p, limit in zip(POWERS, limits)):
                return {p: set((x,) for x in orders) for p, orders in pow_orders.items()}

        # non-trivial return case: query model
        counters = {p: Counter() for p in POWERS}
        x = [encode_inputs(game)] * n
        for x_chunk in [x[i : i + batch_size] for i in range(0, n, batch_size)]:
            batch_inputs, seq_lens = cat_pad_inputs(x_chunk)
            batch_orders, _ = self.do_model_request(self.client, batch_inputs, temperature, top_p)
            batch_orders = list(zip(*batch_orders))
            for p, power in enumerate(POWERS):
                counters[power].update(batch_orders[p])

        logging.info(
            "get_plausible_orders(n={}, t={}) found {} unique sets, choosing top {}".format(
                n, temperature, list(map(len, counters.values())), limits
            )
        )

        # filter out badly-coordinated actions
        counters = {
            power: (
                filter_keys(counter, are_supports_coordinated, log_warn=True)
                if len(counter) > limit
                else counter
            )
            for (power, counter), limit in zip(counters.items(), limits)
        }

        # choose most common
        most_common = {
            power: counter.most_common(limit)
            for (power, counter), limit in zip(counters.items(), limits)
        }

        try:
            logging.info(
                "get_plausible_orders filtered down to {} unique sets, n_0={}, n_cut={}".format(
                    list(map(len, counters.values())),
                    [safe_idx(most_common[p], 0, default=(None, None))[1] for p in POWERS],
                    [
                        safe_idx(most_common[p], limit - 1, default=(None, None))[1]
                        for (p, limit) in zip(POWERS, limits)
                    ],
                )
            )
        except:
            # TODO: remove this if not seen in production
            logging.warning("error in get_plausible_orders logging")

        return {
            power: set([orders for orders, _ in orders_and_counts])
            for power, orders_and_counts in most_common.items()
        }

    def distribute_rollouts(
        self, game, set_orders_dicts: List[Dict], average_n_rollouts=1
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
        game_json = game.to_saved_game_format()

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
                    )
                    for i, d in enumerate(set_orders_dicts)
                ],
            )
        )
        logging.info(
            "Timings[avg.do_rollout, n={}*{}, len={}] {}".format(
                len(set_orders_dicts),
                average_n_rollouts,
                self.max_rollout_length,
                sum(all_timings) / len(all_timings),
            )
        )
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
    ) -> Tuple[Tuple[Dict, List[Dict]], TimingCtx]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
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

            games = [Game.from_saved_game_format(game_json) for _ in range(batch_size)]
            for i in range(len(games)):
                games[i].game_id += f"_{i}"
            est_final_scores = {}

            # set orders if specified
            for power, orders in set_orders_dict.items():
                for game in games:
                    game.set_orders(power, list(orders))

            other_powers = [p for p in POWERS if p not in set_orders_dict]

        for turn_idx in range(max_rollout_length + 1):
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
                if turn_idx == max_rollout_length:
                    _, final_scores = cls.do_model_request(
                        value_client, batch_inputs, temperature, top_p
                    )
                    for game in games:
                        if not game.is_game_done:
                            est_final_scores[
                                game.game_id
                            ] = final_scores  # FIXME: should be final_scores[i]!
                    # don't step the environment on the turn that you're grabbing the value
                    break

                batch_orders, _ = cls.do_model_request(client, batch_inputs, temperature, top_p)

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
                (
                    {p: v.square_score for p, v in current_scores.items()}
                    if game.is_game_done or not use_predicted_final_scores
                    else dict(zip(POWERS, est_final_scores[game.game_id]))
                )
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
    except Exception as e:
        logging.exception("Caught exception in the server (%s)", e)
        raise
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
    ckpt_sync_every=0,
    wait_till_full=False,
    empty_cache=True,
    model_path=None,  # for debugging only
):

    if device > 0:
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info(f"Server {os.getpid()} loaded model, device={device}, seed={seed}")

    if seed is not None:
        torch.manual_seed(seed)

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

                    # print(f"GOT {len(inputs)} BATCH FOR {model_path}")

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
                logging.info("TimeoutError: %s", e)

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


def average_score_dicts(score_dicts: List[Dict]) -> Dict:
    return {p: sum(d.get(p, 0) for d in score_dicts) / len(score_dicts) for p in POWERS}


def filter_keys(d, fn, log_warn=False):
    """Return a copy of a dict-like input containing the subset of keys where fn(k) is truthy"""
    r = type(d)()
    for k, v in d.items():
        if fn(k):
            r[k] = v
        elif log_warn:
            logging.warning(f"filtered bad key: {k}")
    return r


def are_supports_coordinated(orders: List[str]) -> bool:
    """Return False if any supports or convoys are not properly coordinated

    e.g. if "F BLA S A SEV - RUM", return False if "A SEV" is not ordered "A SEV - RUM"
             0  1  2 3  4  5  6
    """
    required = {}
    ordered = {}

    for order in orders:
        split = order.split()
        ordered[split[1]] = split  # save by location
        if split[2] in ("S", "C"):
            if split[4] in required and required[split[4]] != split[3:]:
                # an order is already required of this unit, but it contradicts this one
                return False
            else:
                required[split[4]] = split[3:]

    for req_loc, req_order in required.items():
        if req_loc not in ordered:
            # supporting a foreign unit is always allowed, since we can't
            # control the coordination
            continue

        actual_order = ordered[req_loc]

        if len(req_order) == 2 and actual_order[2] == "-":
            # we supported a hold, but it tried to move
            return False
        elif (
            len(req_order) > 2
            and req_order[2] == "-"
            and (actual_order[2] != "-" or actual_order[3] != req_order[3])
        ):
            # we supported a move, but the order given was (1) not a move, or
            # (2) a move to the wrong destination
            return False

    # checks passed, return True
    return True


def safe_idx(seq, idx, default=None):
    try:
        return seq[idx]
    except IndexError:
        return default
