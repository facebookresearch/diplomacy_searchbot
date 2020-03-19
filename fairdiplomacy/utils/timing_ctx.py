from collections import Counter
import time


class TimingCtx:
    def __init__(self, *, init=None):
        self.timings = init if init is not None else Counter()
        self.arg = None
        self.last_clear = time.time()

    def clear(self):
        self.timings.clear()
        self.last_clear = time.time()

    def __call__(self, arg):
        self.arg = arg
        return self

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, *args):
        self.timings[self.arg] += time.time() - self.tic

    def __repr__(self):
        return dict(**self.timings, total=time.time() - self.last_clear).__repr__()

    def __add__(self, other):
        if isinstance(other, TimingCtx):
            return TimingCtx(init=self.timings + other.timings)
        else:
            raise ValueError(f"__add__ called with type {type(other)}")

    def __truediv__(self, other):
        return {k: v / other for k, v in self.timings.items()}

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def items(self):
        return self.timings.items()
