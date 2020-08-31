from collections import Counter
import time


class TimingCtx:
    def __init__(self, init=None, init_ns=None, first_tic=None):
        self.timings = init if init is not None else Counter()
        self.ns = init_ns if init_ns is not None else Counter()
        self.arg = None
        self.last_clear = time.time()
        self.first_tic = first_tic

    def clear(self):
        self.timings.clear()
        self.last_clear = time.time()

    def start(self, arg):
        if self.arg is not None:
            self.__exit__()
        self.__call__(arg)
        self.__enter__()

    def stop(self):
        self.start(None)

    def __call__(self, arg):
        self.arg = arg
        return self

    def __enter__(self):
        self.tic = time.time()
        if self.first_tic is None:
            self.first_tic = self.tic

    def __exit__(self, *args):
        self.timings[self.arg] += time.time() - self.tic
        self.ns[self.arg] += 1
        self.arg = None

    def __repr__(self):
        return dict(
            total=time.time() - self.last_clear,
            **dict(sorted(self.timings.items(), key=lambda kv: kv[1], reverse=True)),
        ).__repr__()

    def __truediv__(self, other):
        return {k: v / other for k, v in self.timings.items()}

    def __iadd__(self, other):
        if other != 0:
            self.timings += other.timings
            self.ns += other.ns
        return self

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

        log_fn(f"TimingCtx summary of {len(timings)} timings (mean/max shown over this group):")
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

    def pprint(self, log_fn):
        data = []
        for k in self.timings.keys():
            n = self.ns[k]
            t_total = self.timings[k]
            t_mean = t_total / n
            data.append((t_total, t_mean, k, n))

        log_fn(f"TimingCtx summary:")
        ind_log_fn = lambda x: log_fn("  " + x)
        ind_log_fn("{:^24}|{:^8}|{:^18}|{:^18}".format("Key", "N", "t_total (ms)", "t_mean (ms)"))
        ind_log_fn("-" * (24 + 8 + 18 + 18))
        for t_total, t_mean, k, n in sorted(data, reverse=True):
            ind_log_fn(
                "{:^24}|{:^8.1f}|{:^18.1f}|{:^18.1f}".format(k, n, t_total * 1e3, t_mean * 1e3)
            )
        ind_log_fn("-" * (24 + 8 + 18 + 18))

        elapsed = time.time() - self.first_tic
        sum_t_total = sum(t_total for t_total, _, _, _ in data)
        ind_log_fn(
            "{:^24}|{:^8}|{:^18.1f}|{:^18}".format("Lost", "", (elapsed - sum_t_total) * 1e3, "")
        )
        ind_log_fn("{:^24}|{:^8}|{:^18.1f}|{:^18}".format("Total", "", sum_t_total * 1e3, ""))


class DummyCtx:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def __call__(self, *args):
        return self
