import numpy as np


def build_adjacency_matrix(MAP, LOCS):
    """Return an 81x81 matrix of loc graph adjacencies

    - Coasts are adjacent to their parents (e.g. STP/SC is adjacent to STP)
    - locs are not adjacent to themselves
    """

    adj = np.zeros((len(LOCS), len(LOCS)))

    for i, loc in enumerate(LOCS):
        abuts = MAP.abut_list(loc)
        assert len(abuts) > 0, "Something wrong with {}".format(loc)
        for abut in abuts:
            if abut == "SWI":
                # SWI is impassible, so not really a node in the graph. Why is it here..?
                continue
            j = LOCS.index(abut.upper())
            adj[i, j] = 1

    return adj
