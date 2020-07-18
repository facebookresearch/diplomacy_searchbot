import requests
import argparse
import time
import json
from pprint import pprint, pformat
import logging
from collections import defaultdict
from typing import List

from fairdiplomacy.game import Game
from fairdiplomacy.data.build_dataset import (
    TERR_ID_TO_LOC,
    COUNTRY_ID_TO_POWER,
    COUNTRY_POWER_TO_ID,
    get_valid_coastal_variant,
)
from fairdiplomacy.models.consts import POWERS

from diplomacy.integration.webdiplomacy_net.orders import Order as WebdipOrder
from diplomacy.integration.webdiplomacy_net.game import state_dict_to_game_and_power

API_URL = "http://webdiplomacy.net/api.php"

STATUS_ROUTE = "game/status"
MISSING_ORDERS_ROUTE = "players/missing_orders"
POST_ORDERS_ROUTE = "game/orders"

logger = logging.getLogger("webdip")


LOC_TO_TERR_ID = {v: k for k, v in TERR_ID_TO_LOC.items()}


COAST_ID_TO_LOC_ID = {
    76: 8,
    77: 8,
    78: 32,
    79: 32,
    80: 20,
    81: 20,
}


def webdip_state_to_game(webdip_state_json, stop_at_phase=None):
    game = Game()
    for phase in webdip_state_json["phases"]:
        # print(phase)
        terr_to_unit = {}
        for j in phase["units"]:
            if j["unitType"] == "":
                continue
            terr_to_unit[TERR_ID_TO_LOC[j["terrID"]]] = j["unitType"][0]

        orders = phase["orders"]
        power_to_orders = defaultdict(list)
        for order_json in phase["orders"]:
            if game.phase == stop_at_phase:
                break

            # 1. extract the data
            power = COUNTRY_ID_TO_POWER[order_json["countryID"]]
            loc = TERR_ID_TO_LOC[order_json["terrID"]]
            from_loc = TERR_ID_TO_LOC[order_json["fromTerrID"]]
            to_loc = TERR_ID_TO_LOC[order_json["toTerrID"]]
            unit = order_json["unitType"][:1]
            if unit == "":
                unit = terr_to_unit.get(loc, "F")  # default to Fleet in case we missed a coast
            order_type = order_json["type"][0]
            if order_type == "M":
                order_type = "-"
            if order_type == "B":
                unit = order_json["type"].split()[1][0]  # e.g. type="Build Army"
            # if order_type == "D":

            via = "VIA" if order_json["viaConvoy"] == "Yes" else ""

            # 2. build the order string
            if from_loc != "":
                order_str = f"{unit} {loc} {order_type} {terr_to_unit.get(from_loc, 'F')} {from_loc} - {to_loc}"
            else:
                # note: default to Fleet in secondary location because sometimes
                # we get confused with NC / SC
                secondary_unit = terr_to_unit.get(to_loc, "F") + " " if order_type == "S" else ""
                order_str = f"{unit} {loc} {order_type} {secondary_unit}{to_loc} {via}".strip()

            possible_orders = game.get_all_possible_orders()[loc]
            # check if this is
            # a disband that has already been processed. This sometimes
            # happens when a unit is dislodged and has nowhere to
            # retreat -- there will be a disband in the db, but the
            # Game object disbands it automatically.
            if (
                phase["phase"] == "Retreats"
                and order_str.split()[-1] == "D"
                and (order_str not in possible_orders or not game.phase.endswith("RETREATS"))
            ):  # two cases: retreat phase with this order skipped; or retreat phase skipped entirely
                continue

            if order_str not in possible_orders:
                # if order is not valid, try location coastal variants, since
                # some orders are coming out of the db without the proper
                # coast.
                variant_split = get_valid_coastal_variant(order_str.split(), possible_orders)
                if variant_split is not None:
                    order_str = " ".join(variant_split)

            assert order_str in possible_orders, (
                game.phase,
                (power, loc, order_str),
                possible_orders,
                order_json,
            )

            logger.debug('set_phase_orders -> {} "{}"'.format(power, order_str))
            power_to_orders[power].append(order_str)

        for power, orders in power_to_orders.items():
            logger.debug(f"Set {power} {orders}")
            game.set_orders(power, orders)

        if power_to_orders:
            game.process()
            logger.debug(f"Process after: {game.phase}")

    return game


# def get_order_type(order: List[str], phase: str):
#     abbrev_to_type = {
#         "H": "Hold",
#         "-": "Move",
#         "S": "Support",
#         "C": "Convoy",
#         "R": "Retreat",
#         "D": "Destroy" if phase.endswith("ADJUSTMENTS") else "Disband",  # OMG
#         "B": "Build",
#     }
#     if order[0] == "WAIVE":
#         return "Wait"
#     order_type = abbrev_to_type[order[2]]

#     if order_type == "Support":
#         order_type += " hold" if len(order) == 5 else " move"
#     if order_type == "Build":
#         order_type += " Army" if order[0] == "A" else " Fleet"

#     return order_type


# def order_to_json(order: List[str], phase: str):

#     order_type = get_order_type(order, phase)
#     if order[-1] == "VIA":
#         via_convoy = "Yes"
#         order = order[:-1]
#     else:
#         via_convoy = "No"

#     terr_id = LOC_TO_TERR_ID[order[1]]
#     if order_type in ("Move", "Support hold", "Support move", "Convoy", "Retreat"):
#         to_terr_id = LOC_TO_TERR_ID[order[-1]]
#     elif order_type in ("Build Army", "Build Fleet", "Destroy"):
#         to_terr_id = terr_id
#     else:
#         to_terr_id = 0

#     if order_type in ("Support move", "Convoy"):
#         from_terr_id = LOC_TO_TERR_ID[order[4]]
#     else:
#         from_terr_id = 0

#     j = {
#         "type": order_type,
#         "terrID": terr_id,
#         "fromTerrID": from_terr_id,
#         "toTerrID": to_terr_id,
#         "viaConvoy": via_convoy,
#         "unitType": "Fleet" if order[0] == "F" else "Army",
#     }

#     return j


def play_webdip(
    api_key: str,
    game_id=0,
    agent=None,
    check_phase=None,
    json_out=None,
    force=False,
    force_power="ENGLAND",
):
    api_header = {"Authorization": f"Bearer {api_key}"}

    while True:
        logging.info("========================================================================")
        if check_phase or force or json_out:
            next_game = {"gameID": game_id, "countryID": COUNTRY_POWER_TO_ID[force_power]}
        else:
            missing_orders_resp = requests.get(
                API_URL, params={"route": MISSING_ORDERS_ROUTE}, headers=api_header,
            )
            missing_orders_json = json.loads(missing_orders_resp.content)
            logger.info(missing_orders_json)
            if game_id != 0:
                missing_orders_json = [x for x in missing_orders_json if x["gameID"] == game_id]
            logger.info(missing_orders_json)
            if len(missing_orders_json) == 0:
                logger.info("No games to provide orders. Sleeping for 5 seconds.")
                time.sleep(5)
                continue

            next_game = missing_orders_json[0]

        power = COUNTRY_ID_TO_POWER[next_game["countryID"]]
        status_resp = requests.get(
            API_URL,
            params={
                "route": STATUS_ROUTE,
                "gameID": next_game["gameID"],
                "countryID": next_game["countryID"],
            },
            headers=api_header,
        )
        status_json = json.loads(status_resp.content)

        # I can probably use the diplomacy API for this, but it wasn't
        # working out ofthebox for some reason
        # game, power = state_dict_to_game_and_power(status_json, next_game["countryID"])#, max_phases=max_phases)
        # game = Game.clone_from(game)
        game = webdip_state_to_game(status_json, stop_at_phase=check_phase)

        if json_out:
            game_json = game.to_saved_game_format()
            with open(json_out, "w") as jf:
                json.dump(game_json, jf)
            return

        if agent is None:
            return

        logger.info(f"Orderable locations: {game.get_orderable_locations(power)}")

        try:
            agent_orders = agent.get_orders(game, power)
        except:
            tmp_path = "game_exception.%s.json" % next_game["gameID"]
            logging.error(
                f"Got exception while trying to get actions for {power}."
                f" Saving game to {tmp_path}"
            )
            game_json = game.to_saved_game_format()
            with open(tmp_path, "w") as jf:
                json.dump(game_json, jf)
            raise

        logging.info(f"Power: {power}, Orders: {agent_orders}")

        if check_phase:
            break

        agent_orders_json = {
            "gameID": next_game["gameID"],
            "turn": status_json["phases"][-1]["turn"]
            if status_json["phases"]
            else 0,  # len(game.get_phase_history()) - 1, # status_json["phases"]),
            "phase": "Builds"
            if game.phase.startswith("W")
            else "Retreats"
            if game.phase.endswith("RETREATS")
            else "Diplomacy",
            "countryID": next_game["countryID"],
            "orders": [
                WebdipOrder(order, game=game).to_dict() for order in agent_orders
            ],  # [order_to_json(order.split(), game.phase) for order in agent_orders],
            "ready": "Yes",
        }

        for order in agent_orders_json["orders"]:
            if order["fromTerrID"] in COAST_ID_TO_LOC_ID:
                order["fromTerrID"] = COAST_ID_TO_LOC_ID[order["fromTerrID"]]

        print(game.phase)
        # print([p.name for p in game.get_phase_history()])
        # print([p['turn'] for p in status_json['phases']])
        logging.info(f"JSON: {pformat(agent_orders_json)}")
        orders_resp = requests.post(
            API_URL,
            params={"route": POST_ORDERS_ROUTE},
            headers=api_header,
            json=agent_orders_json,
        )

        ###############################################################################
        # After this it's all sanity checks and corner cases
        ###############################################################################

        if (
            orders_resp.status_code == 400
            and orders_resp.content.startswith(b"Invalid phase, expected `Retreats`, got ")
            and len(game.get_state()["units"][power])
            < len(game.get_phase_history()[-1].state["units"][power])
            and game.get_phase_history()[-1].name.endswith("M")
        ):
            # This sometimes happens when a unit is dislodged and has nowhere to
            # retreat -- webdip expects a disband, but the
            # Game object disbands it automatically.
            logging.info("Detected auto-disband... re-issuing after auto-disband.")
            retreat_json = {k: v for k, v in agent_orders_json.items()}
            retreat_json.update(
                phase="Retreats",
                orders=[],  # just provide empty orders; webdip will process the disband automatically
            )
            orders_resp = requests.post(
                API_URL, params={"route": POST_ORDERS_ROUTE}, headers=api_header, json=retreat_json
            )
            continue
            # orders_resp = requests.post(
            #     API_URL,
            #     params={"route": POST_ORDERS_ROUTE},
            #     headers=api_header,
            #     json=agent_orders_json,
            # )
            # if orders_resp.status_code == 400 and "Finished" in str(orders_resp.content):
            #     continue

        if orders_resp.status_code != 200:
            # logging.error(f"Error {orders_resp.status_code}; Response: {orders_resp.content}")
            raise RuntimeError(
                f"Error {orders_resp.status_code}; Response: {str(orders_resp.content)}"
            )

        # logger.info(orders_resp.content)
        try:
            orders_resp_json = json.loads(orders_resp.content)
        except json.decoder.JSONDecodeError as e:
            logger.info("GOT ERROR DECODING ORDER RESPONSE!")
            logger.info(orders_resp.content)
            logger.info(e)
            continue

        logger.info(f"Response: {pformat(orders_resp_json)}")

        # sanity check that the orders were processed correctly
        order_req_by_unit = {
            x["terrID"] if x["terrID"] != "" else x["toTerrID"]: x
            for x in agent_orders_json["orders"]
        }
        order_resp_by_unit = {
            x["terrID"] if x["terrID"] is not None else x["toTerrID"]: x for x in orders_resp_json
        }
        if len(order_req_by_unit) > len(order_resp_by_unit):
            raise RuntimeError(f"{order_req_by_unit} != {order_resp_by_unit}")
        if len(order_req_by_unit) < len(order_resp_by_unit) and game.phase.endswith("MOVEMENT"):
            raise RuntimeError(f"{order_req_by_unit} != {order_resp_by_unit}")
        for terr in order_req_by_unit:
            if order_req_by_unit[terr]["type"] != order_resp_by_unit[terr]["type"]:
                if (
                    order_req_by_unit[terr]["type"] == "Destroy"
                    and order_resp_by_unit[terr]["type"] == "Retreat"
                ):
                    continue
                raise RuntimeError(f"{order_req_by_unit[terr]} != {order_resp_by_unit[terr]}")

        if force:
            break

        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="webdip API key")
    parser.add_argument(
        "--game-id", type=int, help="webdip game ID. If not specified, looks for any game to play."
    )
    parser.add_argument("--country", default="AUSTRIA", help="country of power to control")

    args = parser.parse_args()

    play_webdip(args.api_key, args.game_id, args.country)
