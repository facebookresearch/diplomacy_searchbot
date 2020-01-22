from collections import defaultdict
import time


class TimingCtx:
    def __init__(self):
        self.timings = defaultdict(float)
        self.arg = None

    def clear(self):
        self.timings.clear()

    def __call__(self, arg):
        self.arg = arg
        return self

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, *args):
        self.timings[self.arg] += time.time() - self.tic

    def __repr__(self):
        return dict(self.timings).__repr__()
