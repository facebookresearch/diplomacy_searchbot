import logging
import numpy as np

import diplomacy

from fairdiplomacy.models.consts import LOCS, LOC_TYPES, POWERS


def board_state_to_np(state):
    """Encode the current game state as an 81x35 np array

    See section 4.1 and Figure 2 of the MILA paper for an explanation.
    """
    enc = np.zeros((len(LOCS), 35), dtype=np.float32)

    # encode unit type (0-2) and unit power (3-10)
    enc[:, 2] = 1  # set all to None type
    enc[:, 10] = 1  # set all to None power
    for power, units in state["units"].items():
        for unit in units:
            # unit type (0-2)
            unit_type, loc = unit.split(" ")
            type_idx = ["A", "F"].index(unit_type.replace("*", ""))
            loc_idx = LOCS.index(loc)
            enc[loc_idx, type_idx] = 1

            # unit power (3-10)
            power_idx = 3 + POWERS.index(power)
            enc[loc_idx, power_idx] = 1

            # unset None bits (2, 10)
            enc[loc_idx, 2] = 0
            enc[loc_idx, 10] = 0

    # encode None unit type (2) and None unit power (10) for empty locs
    for i in range(len(LOCS)):
        if enc[i, 0] == 0 and enc[i, 1] == 0:
            enc[i, 2] = 1
            enc[i, 10] = 1

    # encode buildable (11)
    for builds in state["builds"].values():
        if builds["count"] > 0:
            for loc in builds["homes"]:
                enc[LOCS.index(loc), 11] = 1

    # encode removable (12)
    for power, builds in state["builds"].items():
        if builds["count"] < 0:
            for unit in state["units"][power]:
                _, loc = unit.split()
                enc[LOCS.index(loc), 12] = 1

    # encode dislodged unit type (13-15) and power (16-23)
    enc[:, 15] = 1  # set all to None type
    enc[:, 23] = 1  # set all to None power
    for power, units in state["retreats"].items():
        for unit in units.keys():
            # unit type (13-14)
            unit_type, loc = unit.split(" ")
            type_idx = 13 + ["A", "F"].index(unit_type.replace("*", ""))
            loc_idx = LOCS.index(loc)
            enc[loc_idx, type_idx] = 1

            # unit power (3-9)
            power_idx = 16 + POWERS.index(power)
            enc[loc_idx, power_idx] = 1

            # unset None bits (15, 23)
            enc[loc_idx, 15] = 1
            enc[loc_idx, 23] = 1

    # encode area type, land (24), water (25), or coast (26)
    for i, loc in enumerate(LOCS):
        idx = 24 + ["LAND", "WATER", "COAST"].index(LOC_TYPES[loc])
        enc[i, idx] = 1

    # encode supply center owner (27-34)
    enc[:, 34] = 1  # set all to None
    for power, centers in state["centers"].items():
        power_idx = 27 + POWERS.index(power)
        for loc in centers:
            enc[LOCS.index(loc), 34] = 0  # unset None bit (34)
            enc[LOCS.index(loc), power_idx] = 1
    return enc


def prev_orders_to_np(state, orders_by_power):
    """Encode the previous orders as an 81x40 np array

    state: the output of game.get_state()
    orders_by_power: a map of power -> List[order str]

    See section 4.1 and Figure 2 of the MILA paper for an explanation.
    """
    enc = np.zeros((len(LOCS), 40), dtype=np.float32)

    # pre-set None previous order unit type (2), power (10), order type (15),
    # source power (23), destination power (31), and supply owner (39)
    enc[:, [2, 10, 15, 23, 31, 39]] = 1

    # encode phase previous orders
    for power, orders in orders_by_power.items():
        for order in orders:
            try:
                unit_type, loc, order_type, *rest = order.split()
                loc_idx = LOCS.index(loc)

                # set unit type (0-2)
                unit_type_idx = 0 + ["A", "F"].index(unit_type.replace("*", ""))
                enc[loc_idx, unit_type_idx] = 1
                enc[loc_idx, 2] = 0

                # set issuing power (3-10)
                power_idx = 3 + POWERS.index(power)
                enc[loc_idx, power_idx] = 1
                enc[loc_idx, 10] = 0

                # set order type (10-15)
                order_type_idx = 10 + ["H", "-", "S", "C"].index(order_type)
                enc[loc_idx, order_type_idx] = 1
                enc[loc_idx, 15] = 0

                # set source location power (16-23) for support holds, support moves, and convoys
                if order_type == "S" or order_type == "C":
                    source_loc = rest[1]
                    source_loc_power = get_power_at_loc(state, source_loc)
                    if source_loc_power is not None:
                        power_idx = 16 + POWERS.index(source_loc_power)
                        enc[loc_idx, power_idx] = 1
                        enc[loc_idx, 23] = 0

                # set destination location power (24-31) and supply center power
                # (32-39) for moves, support moves, and convoys
                if (
                    order_type == "-"
                    or order_type == "C"
                    or (order_type == "S" and len(rest) > 2 and rest[2] == "-")
                ):  # support move

                    dest_loc = rest[-1]

                    # set destination location power (24-31)
                    dest_loc_power = get_power_at_loc(state, dest_loc)
                    if dest_loc_power is not None:
                        power_idx = 24 + POWERS.index(dest_loc_power)
                        enc[loc_idx, power_idx] = 1
                        enc[loc_idx, 31] = 0

                    # set destination supply center owner (32-39)
                    dest_supply_power = get_supply_center_power(state, dest_loc)
                    if dest_supply_power is not None:
                        power_idx = 32 + POWERS.index(dest_loc_power)
                        enc[loc_idx, power_idx] = 1
                        enc[loc_idx, 39] = 0

            except Exception:
                logging.exception("Can't encode bad order: {}".format(order))
                continue

    return enc


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


if __name__ == "__main__":
    game = diplomacy.Game()
    print(board_state_to_np(game.get_state()))
    print(prev_orders_to_np(game.get_state(), {}))

    game.set_orders("ITALY", ["A ROM H", "F NAP - ION", "A VEN - TRI"])
    game.process()
    print(board_state_to_np(game.get_state()))
    print(prev_orders_to_np(game.get_state(), game.order_history.last_value()))
