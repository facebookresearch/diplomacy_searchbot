# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Dict, Optional, TypeVar

X = TypeVar("X")


def sample_p_dict(d: Dict[X, float], *, rng: Optional[np.random.RandomState] = None) -> X:
    if abs(sum(d.values()) - 1) > 0.001:
        raise ValueError(f"Values sum to {sum(d.values())}")

    if rng is None:
        rng = np.random

    xs, ps = zip(*d.items())
    sumps = sum(ps)
    ps = [p / sumps for p in ps]
    idx = rng.choice(range(len(ps)), p=ps)
    return xs[idx]
