import diplomacy

from .utils import build_adjacency_matrix

MAP = diplomacy.engine.map.Map()
LOCS = sorted(l.upper() for l in MAP.locs if l != "SWI")
LOC_TYPES = {k.upper(): v for (k, v) in MAP.loc_type.items()}
POWERS = sorted(MAP.powers)
SEASONS = ["SPRING", "FALL", "WINTER"]
MAX_SEQ_LEN = 17  # can't have 18 orders in one phase or you've already won
ADJACENCY_MATRIX = build_adjacency_matrix(MAP, LOCS)
