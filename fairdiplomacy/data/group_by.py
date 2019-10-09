from collections import defaultdict


def group_by(collection, fn):
    """Group the elements of a collection by the results of passing them through fn

    Returns a dict, {k_0: list[e_0, e_1, ...]} where e are elements of `collection` and
    f(e_0) = f(e_1) = k_0
    """
    r = defaultdict(list)
    for e in collection:
        k = fn(e)
        r[k].append(e)
    return r
