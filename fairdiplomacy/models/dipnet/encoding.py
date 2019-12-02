import numpy as np

import diplomacy
from diplomacy_research.models import state_space as ss

from fairdiplomacy.models.consts import LOCS, LOC_TYPES, POWERS, MAP


def board_state_to_np(state):
    """Encode the current game state as an 81x35 np array

    See section 4.1 and Figure 2 of the MILA paper for an explanation.
    """
    state_proto = ss.dict_to_proto(state, ss.StateProto)
    return ss.proto_to_board_state(state_proto, MAP).astype('float32')


def prev_orders_to_np(phase):
    """Encode the previous orders as an 81x40 np array

    phase: a game phase, e.g. from game.get_phase_history(), of the last movement phase

    See section 4.1 and Figure 2 of the MILA paper for an explanation.
    """
    phase_proto = ss.dict_to_proto(phase.to_dict(), ss.PhaseHistoryProto)
    return ss.proto_to_prev_orders_state(phase_proto, MAP).astype('float32')


def get_power_at_loc(state, loc):
    """Return the power with a unit at loc, or owning the supply at loc, or None"""
    # check for units
    for power, units in state["units"].items():
        if "A " + loc in units or "F " + loc in units:
            return power

    # supply owner, or None
    return get_supply_center_power(state, loc)


def get_supply_center_power(state, loc):
    """Return the owner of the supply center at loc, or None if not a supply"""
    for power, centers in state["centers"].items():
        if loc.split("/")[0] in centers:
            return power
    return None
