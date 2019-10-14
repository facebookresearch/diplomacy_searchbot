"""
https://github.com/diplomacy/research/blob/master/diplomacy_research/models/state_space.py
"""
from diplomacy import Map


def get_order_vocabulary():
    """ Computes the list of all valid orders on the standard map
        :return: A sorted list of all valid orders on the standard map
    """
    # pylint: disable=too-many-nested-blocks,too-many-branches
    categories = [
        "H",
        "D",
        "B",
        "-",
        "R",
        "SH",
        "S-",
        "-1",
        "S1",
        "C1",  # Move, Support, Convoy (using 1 fleet)
        "-2",
        "S2",
        "C2",  # Move, Support, Convoy (using 2 fleets)
        "-3",
        "S3",
        "C3",  # Move, Support, Convoy (using 3 fleets)
        "-4",
        "S4",
        "C4",
    ]  # Move, Support, Convoy (using 4 fleets)
    orders = {category: set() for category in categories}
    map_object = Map()
    locs = sorted([loc.upper() for loc in map_object.locs])

    # All holds, builds, and disbands orders
    for loc in locs:
        for unit_type in ["A", "F"]:
            if map_object.is_valid_unit("%s %s" % (unit_type, loc)):
                orders["H"].add("%s %s H" % (unit_type, loc))
                orders["D"].add("%s %s D" % (unit_type, loc))

                # Allowing builds in all SCs (even though only homes will likely be used)
                if loc[:3] in map_object.scs:
                    orders["B"].add("%s %s B" % (unit_type, loc))

    # Moves, Retreats, Support Holds
    for unit_loc in locs:
        for dest in [loc.upper() for loc in map_object.abut_list(unit_loc, incl_no_coast=True)]:
            for unit_type in ["A", "F"]:
                if not map_object.is_valid_unit("%s %s" % (unit_type, unit_loc)):
                    continue

                if map_object.abuts(unit_type, unit_loc, "-", dest):
                    orders["-"].add("%s %s - %s" % (unit_type, unit_loc, dest))
                    orders["R"].add("%s %s R %s" % (unit_type, unit_loc, dest))

                # Making sure we can support destination
                if not (
                    map_object.abuts(unit_type, unit_loc, "S", dest)
                    or map_object.abuts(unit_type, unit_loc, "S", dest[:3])
                ):
                    continue

                # Support Hold
                for dest_unit_type in ["A", "F"]:
                    for coast in ["", "/NC", "/SC", "/EC", "/WC"]:
                        if map_object.is_valid_unit("%s %s%s" % (dest_unit_type, dest, coast)):
                            orders["SH"].add(
                                "%s %s S %s %s%s"
                                % (unit_type, unit_loc, dest_unit_type, dest, coast)
                            )

    # Convoys, Move Via
    for nb_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if nb_fleets > 4:
            continue

        for start, fleets, dests in map_object.convoy_paths[nb_fleets]:
            for end in dests:
                orders["-%d" % nb_fleets].add("A %s - %s VIA" % (start, end))
                orders["-%d" % nb_fleets].add("A %s - %s VIA" % (end, start))
                for fleet_loc in fleets:
                    orders["C%d" % nb_fleets].add("F %s C A %s - %s" % (fleet_loc, start, end))
                    orders["C%d" % nb_fleets].add("F %s C A %s - %s" % (fleet_loc, end, start))

    # Support Move (Non-Convoyed)
    for start_loc in locs:
        for dest_loc in [
            loc.upper() for loc in map_object.abut_list(start_loc, incl_no_coast=True)
        ]:
            for support_loc in map_object.abut_list(
                dest_loc, incl_no_coast=True
            ) + map_object.abut_list(dest_loc[:3], incl_no_coast=True):
                support_loc = support_loc.upper()

                # A unit cannot support itself
                if support_loc[:3] == start_loc[:3]:
                    continue

                # Making sure the src unit can move to dest
                # and the support unit can also support to dest
                for src_unit_type in ["A", "F"]:
                    for support_unit_type in ["A", "F"]:
                        if (
                            map_object.abuts(src_unit_type, start_loc, "-", dest_loc)
                            and map_object.abuts(support_unit_type, support_loc, "S", dest_loc[:3])
                            and map_object.is_valid_unit("%s %s" % (src_unit_type, start_loc))
                            and map_object.is_valid_unit(
                                "%s %s" % (support_unit_type, support_loc)
                            )
                        ):
                            orders["S-"].add(
                                "%s %s S %s %s - %s"
                                % (
                                    support_unit_type,
                                    support_loc,
                                    src_unit_type,
                                    start_loc,
                                    dest_loc[:3],
                                )
                            )

    # Support Move (Convoyed)
    for nb_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if nb_fleets > 4:
            continue

        for start_loc, fleets, ends in map_object.convoy_paths[nb_fleets]:
            for dest_loc in ends:
                for support_loc in map_object.abut_list(dest_loc, incl_no_coast=True):
                    support_loc = support_loc.upper()

                    # A unit cannot support itself
                    if support_loc[:3] == start_loc[:3]:
                        continue

                    # A fleet cannot support if it convoys
                    if support_loc in fleets:
                        continue

                    # Making sure the support unit can also support to dest
                    # And that the support unit is not convoying
                    for support_unit_type in ["A", "F"]:
                        if map_object.abuts(
                            support_unit_type, support_loc, "S", dest_loc
                        ) and map_object.is_valid_unit("%s %s" % (support_unit_type, support_loc)):
                            orders["S%d" % nb_fleets].add(
                                "%s %s S A %s - %s"
                                % (support_unit_type, support_loc, start_loc, dest_loc[:3])
                            )

    # Sorting each category
    final_orders = []
    for category in categories:
        category_orders = [order for order in orders[category] if order not in final_orders]
        final_orders += list(
            sorted(category_orders, key=lambda value: (value.split()[1], value))  # Sorting by loc
        )  # Then alphabetically
    return final_orders


if __name__ == "__main__":
    for order in get_order_vocabulary():
        print(order)