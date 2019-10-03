class BaseAgent:
    def get_orders(self, game, power):
        raise NotImplementedError("Subclasses must implement")
