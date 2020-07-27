from collections import Counter
import time


class TimingCtx:
    def __init__(self, init=None, init_ns=None):
        self.timings = init if init is not None else Counter()
        self.ns = init_ns if init_ns is not None else Counter()
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
        self.ns[self.arg] += 1

    def __repr__(self):
        return dict(
            total=time.time() - self.last_clear,
            **dict(sorted(self.timings.items(), key=lambda kv: kv[1], reverse=True)),
        ).__repr__()

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

    @classmethod
    def pprint_multi(cls, timings, log_fn):
        data = []
        for k in timings[0].timings.keys():
            t_mean = sum(t.timings[k] for t in timings) / len(timings)
            t_max = max(t.timings[k] for t in timings)
            n = sum(t.ns[k] for t in timings) / len(timings)
            data.append((t_mean, t_max, k, n))

        log_fn(f"TimingCtx summary of {len(timings)} timings:")
        ind_log_fn = lambda x: log_fn("  " + x)
        ind_log_fn(
            "{:^24}|{:^8}|{:^18}|{:^18}".format("Key", "N_mean", "t_mean (ms)", "t_max (ms)")
        )
        ind_log_fn("-" * (24 + 8 + 18 + 18))
        for t_mean, t_max, k, n in sorted(data, reverse=True):
            ind_log_fn(
                "{:^24}|{:^8.1f}|{:^18.1f}|{:^18.1f}".format(k, n, t_mean * 1e3, t_max * 1e3)
            )
        ind_log_fn("-" * (24 + 8 + 18 + 18))

        sum_t_mean = sum(t_mean for t_mean, _, _, _ in data)
        ind_log_fn("{:^24}|{:^8}|{:^18.1f}|{:^18}".format("Total", "", sum_t_mean * 1e3, ""))

class DummyCtx:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def __call__(self, *args):
        return self
