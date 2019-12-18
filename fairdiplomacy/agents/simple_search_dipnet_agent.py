import copy
import faulthandler
import itertools
import logging
import os
import random
import signal
import torch
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from multiprocessing import Process, set_start_method, get_context
from typing import List, Tuple, Set, Dict

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import encode_inputs, ORDER_VOCABULARY
from fairdiplomacy.models.consts import MAX_SEQ_LEN
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import EOS_IDX
from fairdiplomacy.serving import ModelServer, ModelClient


class SimpleSearchDipnetAgent(BaseAgent):
    def __init__(
        self,
        model_path,
        n_rollout_procs=80,
        n_server_procs=4,
        max_batch_size=1000,
        max_batch_latency=0.005,
        rollouts_per_plausible_order=10,
    ):
        super().__init__()

        self.rollouts_per_plausible_order = rollouts_per_plausible_order
        self.server_ports = list(
            range(ModelServer.DEFAULT_PORT, ModelServer.DEFAULT_PORT + n_server_procs)
        )

        # load model and launch server in separate process
        def load_model():
            return load_dipnet_model(model_path, map_location="cuda", eval=True).cuda()

        for port in self.server_ports:
            logging.info("Launching server on port {}".format(port))
            get_context("fork").Process(
                target=ModelServer,
                kwargs=dict(
                    load_model_fn=load_model,
                    max_batch_size=max_batch_size,
                    max_batch_latency=max_batch_latency,
                    port=port,
                    output_transform=model_output_transform,
                    seed=0,
                    start=True,
                ),
                daemon=True,
            ).start()
            logging.info("Launched server on port {}".format(port))

        self.proc_pool = ProcessPoolExecutor(n_rollout_procs, mp_context=get_context("spawn"))
        self.model_client = ModelClient()

    def get_orders(self, game, power) -> List[str]:
        plausible_orders = self.get_plausible_orders(game, power)
        logging.info("Plausible orders: {}".format(plausible_orders))

        game_json = to_saved_game_format(game)
        port_iterator = itertools.cycle(self.server_ports)
        futures = [
            (
                orders,
                self.proc_pool.submit(
                    self.do_rollout,
                    game_json,
                    {power: orders},
                    model_server_port=next(port_iterator),  # round robin server assignment
                    early_exit_after_no_change=5,
                ),
            )
            for orders in plausible_orders
            for _ in range(self.rollouts_per_plausible_order)
        ]
        results = [(orders, f.result()) for (orders, f) in futures]

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
    def do_model_request(cls, model_client, x, temperature=1.0) -> List[Tuple[str]]:
        """Synchronous request to model server

        Arguments:
        - x: a Tuple of Tensors, where each tensor's dim=0 is the batch dim
        - temperature: model softmax temperature

        Returns:
        - a list (len = batch size) of order-sets
        """
        temp_array = torch.zeros(x[0].shape[0], 1).fill_(temperature)
        order_idxs, = model_client.synchronous_request([*x, temp_array])
        return [
            tuple(ORDER_VOCABULARY[idx] for idx in order_idxs[i, :])
            for i in range(order_idxs.shape[0])
        ]

    def get_plausible_orders(self, game, power, n=100, temperature=0.25) -> Set[Tuple[str]]:
        x = encode_inputs(game, power)
        x = [t.repeat([n] + ([1] * (len(t.shape) - 1))) for t in x]
        return set(self.do_model_request(self.model_client, x, temperature))

    @classmethod
    def do_rollout(
        cls,
        game_json,
        set_orders_dict={},
        temperature=0.05,
        model_server_port=ModelServer.DEFAULT_PORT,
        early_exit_after_no_change=None,
    ) -> Dict[str, int]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
        - set_orders_dict: Dict[power, orders] to set for current turn
        - temperature: model softmax temperature for rollout policy
        - model_server_port: port on which the batching model server is listening
        - early_exit_after_no_change: int, end rollout early if game state is unchanged after # turns

        Returns a Dict[power, final supply count]
        """
        logging.info("hi mom new proc {} -- {}".format(set_orders_dict, os.getpid()))
        faulthandler.register(signal.SIGUSR1)
        torch.set_num_threads(1)

        game = from_saved_game_format(game_json)

        model_client = ModelClient(port=model_server_port)
        orig_order = [v for v in set_orders_dict.values()][0]

        # keep track of state to see if we should exit early
        if early_exit_after_no_change is not None:
            last_units = copy.deepcopy(game.get_state()["units"])
            last_units_n_turns = 0

        # set orders if specified
        for power, orders in set_orders_dict.items():
            game.set_orders(power, list(orders))

        while not game.is_game_done:
            # prepare inputs
            all_possible_orders = game.get_all_possible_orders()
            other_powers = [p for p in game.powers if p not in set_orders_dict]
            xs: List[Tuple] = [encode_inputs(game, p, all_possible_orders) for p in other_powers]
            padded_loc_idxs_seqs, _ = cat_pad_sequences(
                [x[-2] for x in xs], pad_value=-1, pad_to_len=MAX_SEQ_LEN
            )
            padded_mask_seqs, seq_lens = cat_pad_sequences(
                [x[-1] for x in xs], pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN
            )
            batch_inputs = [torch.cat(ts) for ts in list(zip(*xs))[:-2]] + [
                padded_loc_idxs_seqs,
                padded_mask_seqs,
            ]

            # get orders
            batch_orders = cls.do_model_request(model_client, batch_inputs, temperature)

            # process turn
            for other_power, other_orders, seq_len in zip(other_powers, batch_orders, seq_lens):
                game.set_orders(other_power, list(other_orders[:seq_len]))
            game.process()

            # should we exit early?
            if early_exit_after_no_change is not None:
                if game.get_state()["units"] == last_units:
                    last_units_n_turns += 1
                    if last_units_n_turns >= early_exit_after_no_change:
                        logging.info("Early exiting rollout in phase {}".format(game.phase))
                        break
                else:
                    last_units = copy.deepcopy(game.get_state()["units"])
                    last_units_n_turns = 0

            # clear set_orders_dict
            set_orders_dict = {}

        # return supply counts for each power
        result = {k: len(v) for k, v in game.get_state()["centers"].items()}
        logging.info("hi mom pid {} result {}".format(os.getpid(), result))
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


def model_output_transform(y):
    # return only order_idxs, not order_scores
    return y[:1]


if __name__ == "__main__":
    import diplomacy

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)
    logging.info("PID: {}".format(os.getpid()))

    MODEL_PTH = "/checkpoint/jsgray/dipnet.pth"
    game = diplomacy.Game()

    agent = SimpleSearchDipnetAgent(MODEL_PTH)
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))
