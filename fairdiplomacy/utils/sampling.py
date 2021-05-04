# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Dict, Any

X = Any


def sample_p_dict(d: Dict[X, float]) -> X:
    if abs(sum(d.values()) - 1) > 0.001:
        raise ValueError(f"Values sum to {sum(d.values())}")

    xs, ps = zip(*d.items())
    idx = np.random.choice(range(len(ps)), p=ps)
    return xs[idx]
