import diplomacy

from diplomacy_research.models.state_space import get_adjacency_matrix, get_sorted_locs

from .preprocess_adjacency import preprocess_adjacency

MAP = diplomacy.engine.map.Map()
LOCS = get_sorted_locs(MAP)
LOC_TYPES = {k.upper(): v for (k, v) in MAP.loc_type.items()}
POWERS = sorted(MAP.powers)
SEASONS = ["SPRING", "FALL", "WINTER"]
MAX_SEQ_LEN = 17  # can't have 18 orders in one phase or you've already won
ADJACENCY_MATRIX = preprocess_adjacency(get_adjacency_matrix())
