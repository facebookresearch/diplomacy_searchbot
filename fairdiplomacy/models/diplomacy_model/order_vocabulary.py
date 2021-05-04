# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from fairdiplomacy.models.consts import COASTAL_HOME_SCS
from itertools import combinations, product
from typing import Dict, Set

from .order_vocabulary_consts import (
    ORDER_VOCABULARY as __ORDER_VOCABULARY,
    ORDER_VOCABULARY_BY_UNIT as __ORDER_VOCABULARY_BY_UNIT,
)

EOS_IDX = -1
_ORDER_VOCABULARY = None
_ORDER_VOCABULARY_BY_UNIT = None
_ORDER_VOCABULARY_IDXS_BY_UNIT = None
_ORDER_VOCABULARY_IDXS_LEN = None
_ORDER_VOCABULARY_INCOMPATIBLE_BUILD_IDXS = None


def get_order_vocabulary():
    global _ORDER_VOCABULARY, _ORDER_VOCABULARY_BY_UNIT, _ORDER_VOCABULARY_IDXS_BY_UNIT, _ORDER_VOCABULARY_IDXS_LEN

    if _ORDER_VOCABULARY is not None:
        return _ORDER_VOCABULARY

    _ORDER_VOCABULARY, _ORDER_VOCABULARY_BY_UNIT = __ORDER_VOCABULARY, __ORDER_VOCABULARY_BY_UNIT
    order_vocabulary_idxs = {order: i for i, order in enumerate(_ORDER_VOCABULARY)}

    _ORDER_VOCABULARY_IDXS_BY_UNIT = {
        unit: [order_vocabulary_idxs[order] for order in orders]
        for unit, orders in _ORDER_VOCABULARY_BY_UNIT.items()
    }

    _ORDER_VOCABULARY_IDXS_LEN = max(len(o) for o in _ORDER_VOCABULARY_IDXS_BY_UNIT.values())

    return _ORDER_VOCABULARY


def get_order_vocabulary_by_unit():
    get_order_vocabulary()
    return _ORDER_VOCABULARY_BY_UNIT


def get_order_vocabulary_idxs_len():
    get_order_vocabulary()
    return _ORDER_VOCABULARY_IDXS_LEN


def get_order_vocabulary_idxs_by_unit():
    get_order_vocabulary()

    return _ORDER_VOCABULARY_IDXS_BY_UNIT


def get_build_order_sets() -> Dict[str, Set[str]]:
    RAW_ORDERS = {
        "AUSTRIA": [["F TRI B", "A TRI B"], ["A BUD B"], ["A VIE B"]],
        "ENGLAND": [["F LON B", "A LON B"], ["A EDI B", "F EDI B"], ["F LVP B", "A LVP B"]],
        "FRANCE": [["F BRE B", "A BRE B"], ["A PAR B"], ["F MAR B", "A MAR B"]],
        "GERMANY": [["F KIE B", "A KIE B"], ["A BER B", "F BER B"], ["A MUN B"]],
        "ITALY": [["F ROM B", "A ROM B"], ["A NAP B", "F NAP B"], ["F VEN B", "A VEN B"]],
        "RUSSIA": [
            ["A WAR B"],
            ["A MOS B"],
            ["F SEV B", "A SEV B"],
            ["F STP/NC B", "F STP/SC B", "A STP B"],
        ],
        "TURKEY": [["F ANK B", "A ANK B"], ["A SMY B", "F SMY B"], ["F CON B", "A CON B"]],
    }

    return {
        power: [
            set(x)
            for n_build in range(1, len(v) + 1)
            for c in combinations(v, n_build)
            for x in product(*c)
        ]
        for power, v in RAW_ORDERS.items()
    }
