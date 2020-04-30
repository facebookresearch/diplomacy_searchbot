import numpy as np
from typing import Dict, Any

X = Any


def sample_p_dict(d: Dict[X, float]) -> X:
    if abs(sum(d.values()) - 1) > 0.001:
        raise ValueError(f"Values sum to {sum(d.values())}")

    xs, ps = zip(*d.items())
    idx = np.random.choice(range(len(ps)), p=ps)
    return xs[idx]
