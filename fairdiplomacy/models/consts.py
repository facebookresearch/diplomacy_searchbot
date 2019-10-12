import diplomacy

from .utils import build_adjacency_matrix

MAP = diplomacy.engine.map.Map()
LOCS = sorted(l.upper() for l in MAP.locs if l != "SWI")
LOC_TYPES = {k.upper(): v for (k, v) in MAP.loc_type.items()}
POWERS = sorted(MAP.powers)
SEASONS = ["SPRING", "FALL", "WINTER"]
ADJACENCY_MATRIX = build_adjacency_matrix(MAP, LOCS)
