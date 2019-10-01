import random

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def get_orders(self, state, possible_orders):
        return [random.choice(orders) for orders in possible_orders.values()]
