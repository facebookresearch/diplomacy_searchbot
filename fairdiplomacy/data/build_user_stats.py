from pprint import pformat, pprint
from collections import defaultdict, namedtuple
import argparse
import logging
import os
import sqlite3
import time
import json

import diplomacy
import joblib

from fairdiplomacy.data.build_dataset import find_good_games

from trueskill import Rating, rate

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s")

TABLE_GAMES = "redacted_games"
TABLE_MEMBERS = "redacted_members"

STATUS_TO_RANK = {
    "Won": 0,
    "Drawn": 1,
    "Survived": 2,
    "Defeated": 3,
    "Resigned": 3,
}

COUNTRY_ID_TO_POWER = {
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}

# GameRow = namedtuple('GameRow', 'id pressType')
MemberRow = namedtuple("MemberRow", "user game country status")
# class UserStats:
#     def __init__(self):
#         self.total, self.won, self.draw, self.survived, self.lost = 0, 0, 0, 0, 0


def make_ratings_table(db):
    member_rows = [
        MemberRow(*row)
        for row in db.execute(
            f"SELECT hashed_userID, hashed_gameID, countryID, status FROM {TABLE_MEMBERS}"
        ).fetchall()
    ]
    press_type = {game_id: press_type for game_id, press_type in db.execute(
        f"SELECT hashed_id, pressType from redacted_games"
    ).fetchall()}
    user_ids = list(set(r.user for r in member_rows))

    max_userid = max(user_ids)
    ratings = [Rating() for _ in range(max_userid + 1)]  # FIXME: prior?
    user_stats = [defaultdict(float) for _ in range(max_userid + 1)]

    member_dict = {(r.game, r.country): r for r in member_rows}
    good_games = list(find_good_games(db))
    # print(good_games)

    print(f"Found {len(good_games)} good games.")

    # randomly shuffle the rows
    # shuffled_games = [shuffled_games[i] for i in torch.randperm(len(shuffled_games))]

    def make_stats_dict(user_id):
        return {
            **user_stats[u],
            "trueskill_mean": ratings[u].mu,
            "trueskill_std": ratings[u].sigma,
        }

    game_stats = {}
    for ii, game_id in enumerate(good_games):
        if ii & (ii + 1) == 0:
            print(f"Done {ii} / {len(good_games)} games")
        # FIXME: correct for power?
        user_ids, ranks = [], []
        for country_id in range(1, 7 + 1):
            k = (game_id, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                this_user_stats = user_stats[member_row.user]
                this_user_stats["total"] += 1
                if member_row.status not in STATUS_TO_RANK:
                    continue
                this_user_stats[member_row.status] += 1

                user_ids.append(member_dict[k].user)
                ranks.append(STATUS_TO_RANK[member_dict[k].status])

        # each user is on their own team
        if len(user_ids) > 1:
            new_ratings = rate([(ratings[u],) for u in user_ids], ranks=ranks)
            # print('new_ratings', new_ratings)
            # print('ranks', ranks)
            for u, new_rating in zip(user_ids, new_ratings):
                ratings[u] = new_rating[0]

        this_game_stats = {
            'id': game_id,
            'press_type': press_type[game_id]
        }
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_id, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                u = member_row.user
                this_game_stats[pwr] = {
                    "id": u,
                    "cur": make_stats_dict(u),
                    "status": member_row.status,
                }
            else:
                this_game_stats[pwr] = None
        # pprint(this_game_stats)
        game_stats[game_id] = this_game_stats

    print("Adding final stats")
    for ii, game_id in enumerate(good_games):
        this_game_stats = game_stats[game_id]
        if this_game_stats is None:
            continue
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_id, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                u = member_row.user
                this_game_stats[pwr]['final'] = make_stats_dict(u)

    return game_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Dump output pickle to this file")
    parser.add_argument(
        "--db-path",
        default="webdiplomacy-movesdata-20190707.sqlite3.db",
        help="Path to SQLITE db file",
    )
    args = parser.parse_args()

    db = sqlite3.connect(args.db_path)

    game_stats = make_ratings_table(db)

    with open(args.out, "w") as f:
        json.dump(game_stats, f)
