import faulthandler
import logging
import os
import signal
import torch
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from multiprocessing import Process, set_start_method
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
        n_procs=40,
        get_orders_timeout=5,
        max_batch_size=1000,
        max_batch_latency=0.05,
    ):
        super().__init__()

        set_start_method("forkserver")

        # load model and launch server in separate process
        # model = load_dipnet_model(model_path, map_location="cuda", eval=True)
        # Process(
        #     target=ModelServer(
        #         model, max_batch_size, max_batch_latency, output_transform=lambda y: y[:1]
        #     ).start,
        #     daemon=True,
        # ).start()

        self.model_client = ModelClient()
        self.proc_pool = ProcessPoolExecutor(n_procs)

    def get_orders(self, game, power) -> List[str]:
        plausible_orders = self.get_plausible_orders(game, power)
        logging.info("Plausible orders: {}".format(plausible_orders))

        ROLLOUTS_PER_ORDER = 5

        game_json = to_saved_game_format(game)
        futures = [
            (orders, self.proc_pool.submit(self.do_rollout, game_json, {power: orders}))
            for orders in plausible_orders
            for _ in range(ROLLOUTS_PER_ORDER)
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
    def do_rollout(cls, game_json, set_orders_dict={}, temperature=0.05) -> Dict[str, int]:
        """Complete game, optionally setting orders for the current turn

        This method can safely be called in a subprocess

        Arguments:
        - game_json: json-formatted game, e.g. output of to_saved_game_format(game)
        - set_orders_dict: Dict[power, orders] to set for current turn
        - temperature: model softmax temperature for rollout policy

        Returns a Dict[power, final supply count]
        """
        logging.info("hi mom new proc {} -- {}".format(set_orders_dict, os.getpid()))
        faulthandler.register(signal.SIGUSR1)
        torch.set_num_threads(1)

        game = from_saved_game_format(game_json)

        model_client = ModelClient()
        orig_order = [v for v in set_orders_dict.values()][0]

        # set orders if specified
        for power, orders in set_orders_dict.items():
            game.set_orders(power, list(orders))

        while not game.is_game_done:
            logging.debug("hi mom rollout {} phase {}".format(orig_order, game.phase))

            # get orders
            all_possible_orders = game.get_all_possible_orders()
            other_powers = [p for p in game.powers if p not in set_orders_dict]
            xs: List[Tuple] = [encode_inputs(game, p, all_possible_orders) for p in other_powers]
            padded_mask_seqs, seq_lens = cat_pad_sequences(
                [x[-1] for x in xs], pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN
            )
            batch_inputs = [torch.cat(ts) for ts in list(zip(*xs))[:-1]] + [padded_mask_seqs]
            batch_orders = cls.do_model_request(model_client, batch_inputs, temperature)

            # process turn
            for other_power, other_orders, seq_len in zip(other_powers, batch_orders, seq_lens):
                game.set_orders(other_power, list(other_orders[:seq_len]))
            game.process()

            # clear set_orders_dict
            set_orders_dict = {}

        # return supply counts for each power
        result = {k: len(v) for k, v in game.get_state()["centers"].items()}
        logging.info("hi mom result {}".format(result))
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


if __name__ == "__main__":
    import diplomacy

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)
    logging.info("PID: {}".format(os.getpid()))

    MODEL_PTH = "/checkpoint/jsgray/dipnet.20103672.pth"
    game = diplomacy.Game()

    agent = SimpleSearchDipnetAgent(MODEL_PTH)
    logging.info("Chose orders: {}".format(agent.get_orders(game, "ITALY")))
