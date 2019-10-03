import asyncio
import os
import random
import sys
import threading
import queue

import diplomacy
from tornado import gen
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
from diplomacy_research.players.benchmark_player import DipNetSLPlayer
from diplomacy_research.utils.cluster import start_io_loop

from fairdiplomacy.agents.base_agent import BaseAgent


class MilaSLAgent(BaseAgent):
    def __init__(self):
        self.q_in = queue.Queue()
        self.q_out = queue.Queue()
        self.thread = threading.Thread(
            target=thread_main, args=(self.q_in, self.q_out), daemon=True
        )
        self.thread.start()

    def get_orders(self, game, power):
        self.q_in.put((game, power))
        orders = self.q_out.get()
        return orders


def thread_main(q_in, q_out):
    @gen.coroutine
    def coroutine():
        player = DipNetSLPlayer()
        while True:
            game, power = q_in.get()
            orders = yield player.get_orders(game, power)
            q_out.put(orders)

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
    start_io_loop(coroutine)


if __name__ == "__main__":
    agent = MilaSLAgent()
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY")
    print(orders)
