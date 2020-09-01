import diplomacy
import numpy as np
from diplomacy_research.models.state_space import (
    get_adjacency_matrix,
    get_sorted_locs,
    get_board_alignments,
)

from .preprocess_adjacency import preprocess_adjacency

MAP = diplomacy.engine.map.Map()
LOCS = get_sorted_locs(MAP)
LOC_TYPES = {k.upper(): v for (k, v) in MAP.loc_type.items()}
POWERS = sorted(MAP.powers)
POWER2IDX = {v: k for k, v in enumerate(POWERS)}
SEASONS = ["SPRING", "FALL", "WINTER"]
MAX_SEQ_LEN = 17  # can't have 18 orders in one phase or you've already won
N_SCS = 34  # number of supply centers
ADJACENCY_MATRIX = preprocess_adjacency(get_adjacency_matrix())
MASTER_ALIGNMENTS = np.stack(get_board_alignments(LOCS, False, 1, 81))
COASTAL_HOME_SCS = [
    "TRI",
    "EDI",
    "LVP",
    "LON",
    "BRE",
    "MAR",
    "BER",
    "KIE",
    "NAP",
    "ROM",
    "VEN",
    "SEV",
    "STP",
    "STP/NC",
    "STP/SC",
    "ANK",
    "CON",
    "SMY",
]
LOGIT_MASK_VAL = -1e8
