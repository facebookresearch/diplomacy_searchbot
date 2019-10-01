class BaseAgent:
    def get_orders(self, state, possible_orders):
        raise NotImplementedError("Subclasses must implement")
