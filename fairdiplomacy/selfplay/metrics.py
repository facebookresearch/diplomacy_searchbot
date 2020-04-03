from typing import Dict, Optional
import collections
import importlib
import logging
import pathlib
import subprocess
import time

import torch


def _sanitize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


def rec_map(callable, dict_seq_nest):
    """Recursive map that goes into dics, lists, and tuples.
 
    This function tries to preserve named tuples and custom dics. It won't
    work with non-materialized iterators.
    """
    if isinstance(dict_seq_nest, list):
        return type(dict_seq_nest)(rec_map(callable, x) for x in dict_seq_nest)
    if isinstance(dict_seq_nest, tuple):
        return type(dict_seq_nest)(*[rec_map(callable, x) for x in dict_seq_nest])
    if isinstance(dict_seq_nest, dict):
        return type(dict_seq_nest)((k, rec_map(callable, v)) for k, v in dict_seq_nest.items())
    return callable(dict_seq_nest)


def recursive_tensor_item(tensor_nest):
    return rec_map(_sanitize, tensor_nest)


def flatten_dict(tensor_dict):
    if not isinstance(tensor_dict, dict):
        return tensor_dict
    dd = {}
    for k, v in tensor_dict.items():
        v = flatten_dict(v)
        if isinstance(v, dict):
            for subkey, subvaule in v.items():
                dd[f"{k}/{subkey}"] = subvaule
        else:
            dd[k] = v
    dd = dict(sorted(dd.items()))
    return dd


class StopWatchTimer:
    """Time something with ability to pause."""

    def __init__(self, auto_start=True):
        self._elapsed: float = 0
        self._start: Optional[float] = None
        if auto_start:
            self.start()

    def start(self) -> None:
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + time.time() - self._start
        else:
            return self._elapsed

    def pause(self) -> None:
        self._elapsed = self.elapsed
        self._start = None


class MultiStopWatchTimer:
    """Time several stages that go one after another."""

    def __init__(self):
        self._start: Optional[float] = None
        self._name = None
        self._timings = collections.defaultdict(float)

    def start(self, name) -> None:
        now = time.time()
        if self._name is not None:
            self._timings[self._name] += now - self._start
        self._start = now
        self._name = name

    @property
    def timings(self) -> Dict[str, float]:
        if self._name is not None:
            self.start(self._name)
        return self._timings


class FractionCounter:
    def __init__(self):
        self.numerator = self.denominator = 0

    def update(self, top, bottom=1.0):
        self.numerator += _sanitize(top)
        self.denominator += _sanitize(bottom)

    def value(self):
        return self.numerator / max(self.denominator, 1e-6)


class MaxCounter:
    def __init__(self, default=0):
        self._value = default

    def update(self, value):
        self._value = max(value, self._value)

    def value(self):
        return self._value
