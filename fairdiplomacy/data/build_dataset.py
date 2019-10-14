from pprint import pformat
import argparse
import logging
import os
import sqlite3

import diplomacy
import joblib

from consts import PROPER_START_TERR_IDS, COUNTRY_ID_TO_POWER, TERR_ID_TO_LOC
from group_by import group_by

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s")


def find_good_games(db):
    """Yields game ids for games that meet QA criteria"""
    for (game_id,) in db.execute("SELECT id FROM wD_Games WHERE variantID=1"):
        moves = db.execute(
            """SELECT turn, terrID, type
               FROM wD_MovesArchive
               WHERE gameID=?
            """,
            (game_id,),
        ).fetchall()

        # Criterion: has proper starting units
        if {terr_id for (turn, terr_id, _) in moves if turn == 0} != set(PROPER_START_TERR_IDS):
            continue

        # Criterion: is at least 5 turns long
        turns = {turn for (turn, _, _) in moves}
        if not all(t in turns for t in range(5)):
            continue

        # Criterion: has non-hold moves in turn 0
        if all(typ == "Hold" for (turn, _, typ) in moves if turn == 0):
            continue

        # Meets all criteria: good game
        yield game_id


def move_row_to_order_str(row):
    """Convert a db row from wD_MovesArchive into an order string

    Returns a 3-tuple:
      - turn
      - power string, e.g. "AUSTRIA"
      - order string, e.g. "A RUH - BEL"

    N.B. Some order strings are incomplete: supports, convoys, and destroys are
    missing the unit type, since it is not stored in the db and must be
    determined from the game context.

    e.g. this function will return "A PAR S BRE - PIC" instead of the proper
    "A PAR S A BRE - PIC", since it is unknown which unit type resides at BRE

    Similarly, this function may return "X BRE X" instead of "A BRE D" or "F BRE D"
    """

    gameID, turn, terrID, countryID, unitType, success, dislodged, typ, toTerrID, fromTerrID, viaConvoy = (
        row
    )

    power = COUNTRY_ID_TO_POWER[countryID]  # e.g. "ITALY"
    loc = TERR_ID_TO_LOC[terrID]  # short code, e.g. "TRI"

    if typ == "Build Army":
        return turn, power, "A {} B".format(loc)
    if typ == "Build Fleet":
        return turn, power, "F {} B".format(loc)
    if typ == "Destroy":
        return turn, power, "X {} X".format(loc)

    unit_type = unitType[0].upper()  # "A" or "F"

    if typ == "Hold":
        return turn, power, "{} {} H".format(unit_type, loc)
    elif typ == "Move":
        to_loc = TERR_ID_TO_LOC[toTerrID]
        via_suffix = " VIA" if viaConvoy == "Yes" else ""
        return turn, power, "{} {} - {}{}".format(unit_type, loc, to_loc, via_suffix)
    elif typ == "Support hold":
        to_loc = TERR_ID_TO_LOC[toTerrID]
        return turn, power, "{} {} S {}".format(unit_type, loc, to_loc)
    elif typ == "Support move":
        from_loc = TERR_ID_TO_LOC[fromTerrID]
        to_loc = TERR_ID_TO_LOC[toTerrID]
        return (turn, power, "{} {} S {} - {}".format(unit_type, loc, from_loc, to_loc))
    elif typ == "Convoy":
        from_loc = TERR_ID_TO_LOC[fromTerrID]
        to_loc = TERR_ID_TO_LOC[toTerrID]
        return (turn, power, "{} {} C {} - {}".format(unit_type, loc, from_loc, to_loc))
    elif typ == "Retreat":
        to_loc = TERR_ID_TO_LOC[toTerrID]
        return turn, power, "{} {} R {}".format(unit_type, loc, to_loc)
    elif typ == "Disband":
        return turn, power, "{} {} D".format(unit_type, loc)
    else:
        raise ValueError(
            "Unexpected move type = {} in gameID = {}, turn = {}, terrID = {}".format(
                typ, gameID, turn, terrID
            )
        )


def get_game_orders(db, game_id):
    """Return a dict mapping turn -> list of (turn, power, order) tuples

    i.e. return type is Dict[int, List[Tuple[int, str, str]]]
    """
    # gather orders
    turn_power_orders = [
        move_row_to_order_str(row)
        for row in db.execute("SELECT * FROM wD_MovesArchive WHERE gameID=?", (game_id,))
    ]
    orders_by_turn = group_by(turn_power_orders, lambda tpo: tpo[0])

    # major weirdness in the db: if the game ends in a draw, the orders from
    # the final turn are repeated, resulting in a bunch of invalid orders.
    game_over, last_turn = db.execute(
        "SELECT gameOver,turn FROM wD_Games WHERE id=?", (game_id,)
    ).fetchone()
    if game_over == "Drawn":
        last_orders = {(power, order) for (_, power, order) in orders_by_turn[last_turn]}
        penult_orders = {(power, order) for (_, power, order) in orders_by_turn[last_turn - 1]}
        if last_orders == penult_orders:
            # fix this weirdness by removing the duplicate orders
            del orders_by_turn[last_turn]

    return orders_by_turn


def process_game(db, game_id, log_path=None):
    """Search db for moves from `game_id` and process them through a Game()

    Return a Game object with all moves processed
    """
    if log_path is not None:
        logging.getLogger("p").propagate = False
        logging.getLogger("p").setLevel(logging.DEBUG)
        logging.getLogger("p").handlers = [logging.FileHandler(log_path)]

    # gather orders
    orders_by_turn = get_game_orders(db, game_id)

    # run them through a diplomacy.Game
    game = diplomacy.Game()
    for turn in range(len(orders_by_turn)):

        logging.getLogger("p").info("=======================> TURN {}".format(turn))
        logging.getLogger("p").debug(
            "Turn orders from db: {}".format(pformat(orders_by_turn[turn]))
        )

        # separate orders into one of {"MOVEMENT", "RETREATS", "DISBANDS", "BUILDS"}
        orders_by_category = group_by(orders_by_turn[turn], lambda tpo: get_order_category(tpo[2]))

        # process movements
        set_phase_orders(game, orders_by_category["MOVEMENT"])
        logging.getLogger("p").info("process {}".format(game.phase))
        game.process()
        logging.getLogger("p").debug(
            "post-process units: {}".format(pformat(game.get_state()["units"]))
        )

        # process all retreats
        if game.phase.split()[-1] == "RETREATS":
            set_phase_orders(game, orders_by_category["RETREATS"])

            # which locs require a move?
            orderable_locs = {
                loc for locs in game.get_orderable_locations().values() for loc in locs
            }

            # which locs have been ordered already?
            ordered_locs = {
                order.split()[1] for orders in game.get_orders().values() for order in orders
            }

            # which locs are missing a move?
            missing_locs = orderable_locs - ordered_locs
            logging.getLogger("p").debug("retreat phase missing locs: {}".format(missing_locs))

            # if there is a disband for this loc, process it in the retreat phase
            for loc in missing_locs:
                power, order = pop_order_at_loc(orders_by_category["DISBANDS"], loc)
                if order is not None:
                    logging.getLogger("p").info("set order {} {}".format(power, [order]))
                    game.set_orders(power, [order])

            logging.getLogger("p").info("process {}".format(game.phase))
            game.process()
            logging.getLogger("p").debug(
                "post-process units: {}".format(pformat(game.get_state()["units"]))
            )

        # process builds, remaining disbands
        if game.phase.split()[-1] == "ADJUSTMENTS":
            set_phase_orders(game, orders_by_category["BUILDS"] + orders_by_category["DISBANDS"])
            logging.getLogger("p").info("process {}".format(game.phase))
            game.process()
            logging.getLogger("p").debug(
                "post-process units: {}".format(pformat(game.get_state()["units"]))
            )

    return game


def set_phase_orders(game, phase_orders):
    logging.getLogger("p").info("set_phase_orders start {}".format(game.phase))

    # map of loc -> (power, "A/F")
    unit_at_loc = {}
    for power, unit_list in game.get_state()["units"].items():
        for unit in unit_list:
            unit_type, loc = unit.split()
            unit_at_loc[loc] = (power, unit_type)
            unit_at_loc[loc.split("/")[0]] = (power, unit_type)

    orders_by_power = group_by(phase_orders, lambda tpo: tpo[1])
    for power, tpos in orders_by_power.items():

        # compile orders, adding in missing unit type info for supports/convoys/destroys
        orders = []
        for _, _, order in tpos:
            split = order.split()

            # fill in unit type for supports / convoys
            if split[2] in ("S", "C"):
                loc = split[3]
                _, unit_type = unit_at_loc[loc]
                split = split[:3] + [unit_type] + split[3:]

            # fill in unit type for destroys
            elif split[0] == "X":
                loc = split[1]
                _, unit_type = unit_at_loc[loc]
                split = [unit_type, loc, "D"]

            possible_orders = set(game.get_all_possible_orders()[split[1]])
            if " ".join(split) in possible_orders:
                orders.append(" ".join(split))
            else:
                # if order is not valid, try location coastal variants, since
                # some orders are coming out of the db without the proper
                # coast.
                variant_split = get_valid_coastal_variant(split, possible_orders)
                if variant_split is not None:
                    orders.append(" ".join(variant_split))
                else:
                    # if there are no valid coastal variants, check if this is
                    # a disband that has already been processed. This sometimes
                    # happens when a unit is dislodged and has nowhere to
                    # retreat -- there will be a disband in the db, but the
                    # Game object disbands it automatically.
                    if split[2] == "D" and unit_at_loc.get(split[1], (None, None))[0] != power:
                        logging.getLogger("p").warning(
                            'Skipping disband: {} "{}"'.format(power, order)
                        )
                    else:
                        error_msg = 'Bad order: {} "{}", possible_orders={}'.format(
                            power, order, possible_orders
                        )
                        logging.getLogger("p").error(error_msg)
                        err = ValueError(error_msg)
                        err.partial_game = game
                        raise err

        # ensure that each order is valid
        for order in orders:
            loc = order.split()[1]
            assert order in game.get_all_possible_orders()[loc], (
                game.phase,
                (power, loc, order),
                game.get_all_possible_orders()[loc],
            )

        logging.getLogger("p").info('set_phase_orders -> {} "{}"'.format(power, orders))
        game.set_orders(power, orders)


def get_valid_coastal_variant(split, possible_orders):
    """Find a variation on the `split` order that is in `possible_orders`

    Args:
        - split: a list of order string components,
                 e.g. ["F", "AEG", "S", "F", "BUL", "-", "GRE"]
        - possible_orders: a list of order strings,
                e.g. ["F AEG S F BUL/SC - GRE", "F AEG H", "F AEG - GRE", ...]

    This function tries variations (e.g. "BUL", "BUL/SC", etc.) of the `split`
    order until one is found in `possible_orders`.

    Returns a split order, or None if none is found

    e.g. for the example inputs above, this function returns:
            ["F", "AEG", "S", "F", "BUL/SC", "-", "GRE"]
    """
    for idx in [1, 4, 6]:  # try loc, from_loc, and to_loc
        if len(split) <= idx:
            continue
        for variant in [split[idx].split("/")[0] + x for x in ["", "/NC", "/EC", "/SC", "/WC"]]:
            try_split = split[:idx] + [variant] + split[(idx + 1) :]
            if " ".join(try_split) in possible_orders:
                return try_split
    return None


def pop_order_at_loc(tpos, loc):
    """If there is an order at loc in tpos, remove and return it

    tpos: A list of (turn, power, order) tuples

    Returns: (power, order) if found, else (None, None)
    """
    for i, (_, power, order) in enumerate(tpos):
        order_loc = order.split()[1]
        if loc.split("/")[0] == order_loc.split("/")[0]:
            del tpos[i]
            return (power, order)
    return None, None


def get_order_category(order):
    """Given an order string, return the category type, one of:

    {"MOVEMENT, "RETREATS", "DISBANDS", "BUILDS"}
    """
    order_type = order.split()[2]
    if order_type in ("X", "D"):
        return "DISBANDS"
    elif order_type == "B":
        return "DISBANDS"
    elif order_type == "R":
        return "RETREATS"
    else:
        return "MOVEMENT"


def process_and_save_game(game_id, db_path):
    db = sqlite3.connect(db_path)

    save_path = os.path.join(args.out_dir, "game_{}.json".format(game_id))
    log_path = os.path.join(args.out_dir, "game_{}.log".format(game_id))
    if not args.overwrite and os.path.isfile(save_path):
        logging.info("Skipping game {}".format(game_id))
        return

    logging.info("About to process game {}".format(game_id))
    try:
        game = process_game(db, game_id, log_path=log_path)
    except Exception as err:
        logging.exception("Exception processing game {}".format(game_id))
        if hasattr(err, "partial_game"):
            partial_save_path = save_path + ".partial"
            diplomacy.utils.export.to_saved_game_format(err.partial_game, partial_save_path)
            logging.info("Saved partial game to {}".format(partial_save_path))
        return

    # Save json file
    diplomacy.utils.export.to_saved_game_format(game, save_path)
    logging.info("Saved to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Dump game.json files to this dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite existing game.json files",
    )
    parser.add_argument(
        "--db-path",
        default="webdiplomacy-movesdata-20190707.sqlite3.db",
        help="Path to SQLITE db file",
    )
    args = parser.parse_args()

    logging.info("Ensuring out dir exists: {}".format(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    db = sqlite3.connect(args.db_path)
    # good_game_ids = list(find_good_games(db))
    good_game_ids = [3232, 3587, 3533, 3269, 3614, 3402, 3198, 3589, 3606]
    logging.info("Found {} good game ids".format(len(good_game_ids)))

    joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(process_and_save_game)(game_id, args.db_path) for game_id in good_game_ids
    )
