import asyncio
import threading
import queue

import diplomacy
from tornado import gen
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
from diplomacy_research.models.datasets.grpc_dataset import GRPCDataset
from diplomacy_research.players.benchmarks import sl_neurips2019
from diplomacy_research.players.model_based_player import ModelBasedPlayer
from diplomacy_research.utils.cluster import start_io_loop

from fairdiplomacy.agents.base_agent import BaseAgent


class MilaSLAgent(BaseAgent):
    def __init__(self, temperature=0.1):
        self.q_in = queue.Queue()
        self.q_out = queue.Queue()
        self.thread = threading.Thread(
            target=thread_main, args=(self.q_in, self.q_out, temperature), daemon=True
        )
        self.thread.start()

    def get_orders(self, game, power):
        self.q_in.put((game, power))
        orders = self.q_out.get()
        return orders


def thread_main(q_in, q_out, temperature):
    @gen.coroutine
    def coroutine():
        player = Player()
        while True:
            game, power = q_in.get()
            orders = yield player.get_orders(game, power, temperature=temperature)
            q_out.put(orders)

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
    start_io_loop(coroutine)


class Player(ModelBasedPlayer):
    def __init__(self, temperature=0.1, host="devfair0304", port=9501):
        grpc_dataset = GRPCDataset(
            hostname=host,
            port=port,
            model_name="player",
            signature=sl_neurips2019.PolicyAdapter.get_signature(),
            dataset_builder=sl_neurips2019.BaseDatasetBuilder(),
        )
        policy_adapter = sl_neurips2019.PolicyAdapter(grpc_dataset)

        # Building benchmark model
        super().__init__(policy_adapter=policy_adapter, temperature=temperature, use_beam=False)

