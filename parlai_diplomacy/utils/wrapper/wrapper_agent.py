"""
Module to wrap ParlAI agent to produce orders given a game JSON.
"""
from abc import ABC, abstractmethod
import json
import pathlib
from typing import Dict, List, Optional, Set

from parlai.core.agents import create_agent_from_model_file
import parlai_diplomacy.utils.game_loading as game_load
import parlai_diplomacy.utils.game_to_sequence_formatting as formatter


TESTCASE_PATH = str(pathlib.Path(__file__).parent.parent.parent.parent / "test_situations.json")


class BaseWrapper(ABC):
    OPT_OVERRIDE = {
        "interactive_mode": True,
        "skip_generation": False,
        "gpu": 0,  # NOTE: loading on gpu
        "datapath": "/private/home/wyshi/ParlAI/data",  # TODO: get rid of this dependency
    }

    def __init__(self, model_path):
        self.parlai_agent = create_agent_from_model_file(model_path, self.override_opts())

    def override_opts(self):
        return self.OPT_OVERRIDE

    def get_model_pred(self, input_seq):
        """
        Return the model's prediction for an input sequence
        """
        ex = {"text": input_seq, "episode_done": True}
        self.parlai_agent.observe(ex)
        response = self.parlai_agent.act()["text"]
        return response

    def produce_order(
        self,
        game_json: Dict,
        possible_orders: List[List[str]],
        power: str,
        max_orders: Optional[int] = None,
    ) -> List[str]:
        """
        Queries an agent to select most probable orders from the possible ones.

        Args:
            game_json: game json file as returned by Game.to_saved_format().
            possible_orders: list of possibilities for each location. E.g.,
              the first location may contain all orders that are possible for
              “A MAR”.
            power: name of the power for which orders are queried.
            max_orders: only available for build stage, max number of orders
              to produce.

        Returns:
            orders: list of orders for each position or list of build/disband orders for chosen positions.

            If `max_orders is None`, than `len(orders) == len(possible_orders)`
            and `orders in product(*possible_orders)`.

            If `max_orders is not None`, then `len(orders) <= max_orders`
            and `any(set(seq).issuperset(orders) for seq in product(*possible_orders))`.
        """
        game_json = game_load.organize_game_by_phase(game_json)
        seqs = self.format_input_seq(game_json)
        seq = list(seqs.values())[-1][power]
        raw_pred = self.get_model_pred(seq)
        return self.format_output_seq(raw_pred, power)

    @abstractmethod
    def format_input_seq(self, game_json: Dict) -> Dict[str, str]:
        """
        Given a game json, return a dictionary of formatted input sequences for each
        power to feed directly to the model

        Args:
        game_json: game json file as returned by Game.to_saved_format().

        Returns:
        A dict of `power` -> `input sequence for model`
        """
        pass

    @abstractmethod
    def format_output_seq(self, output_seq: str, power: str) -> Set[str]:
        """
        Given a text sequence returned by a model and a power, return the set
        of orders predicted for a given model

        Args:
        output_seq: output sequence of text returned by the model

        Returns:
        A set of orders
        """
        pass


class ParlAISingleOrderWrapper(BaseWrapper):
    def format_input_seq(self, game_json):
        format_sequences = formatter.SequenceFormatHelper.change_format(game_json, "state_order")
        return format_sequences

    def format_output_seq(self, output_seq, power):
        preds = formatter.order_seq_to_fairdip(output_seq)
        return preds


class ParlAIAllOrderWrapper(BaseWrapper):
    def format_input_seq(self, game_json):
        format_sequences = formatter.SequenceFormatHelper.change_format(
            game_json, "shortstate_order"
        )
        return format_sequences

    def format_output_seq(self, output_seq, power):
        orders_dct = formatter.all_orders_seq_to_dct(output_seq)
        power_preds = orders_dct[power.capitalize()]
        return power_preds


def test_load_single():
    """
    Test load and predictions for a ParlAI transformer trained to produce a single order at a time
    """
    with open(TESTCASE_PATH, "r") as f:
        data = json.load(f)

    agent = ParlAISingleOrderWrapper(
        model_path="/checkpoint/wyshi/20200827/resume_newdata_state_order_chunk_bart_diplomacy/18e/model"
    )

    for _, info in data.items():
        game_json_path = info["game_path"]
        game = game_load.load_viz_to_dipcc_format(game_json_path)
        orders = agent.produce_order(game_json=game, possible_orders=[], max_orders=None)
        print(orders)
        break


def test_load_all():
    """
    Test load and predictions for a ParlAI transformer trained to predict all orders at once
    """
    with open(TESTCASE_PATH, "r") as f:
        data = json.load(f)

    agent = ParlAIAllOrderWrapper(
        model_path="/checkpoint/wyshi/20200827/resume_allorder_shortstate_bart_diplomacy/18e/model"
    )

    for _, info in data.items():
        game_json_path = info["game_path"]
        game = game_load.load_viz_to_dipcc_format(game_json_path)
        orders = agent.produce_order(game_json=game, possible_orders=[], max_orders=None)
        print(orders)
        break


if __name__ == "__main__":
    test_load_single()
    test_load_all()
