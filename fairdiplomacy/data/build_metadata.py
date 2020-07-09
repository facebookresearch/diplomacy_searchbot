from collections import defaultdict, namedtuple
import argparse
import logging
import os
import sqlite3
import json

import torch
import torch.nn as nn
import torch.optim as optim

from fairdiplomacy.data.build_dataset import find_good_games

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


MemberRow = namedtuple("MemberRow", "user game country status")


def compute_logit_ratings(game_stats, wd=0.1):
    # 1. construct dataset

    # 1a. find all u1 > u2 pairs from the game stats
    dataset = []
    POWERS = COUNTRY_ID_TO_POWER.values()
    for game in game_stats.values():
        for pwr0 in POWERS:
            p0 = game[pwr0]["points"]
            id0 = game[pwr0]["id"]
            for pwr1 in POWERS:
                p1 = game[pwr1]["points"]
                id1 = game[pwr1]["id"]
                if pwr0 == pwr1:
                    continue
                if p0 > p1:
                    dataset.append((id0, id1))
                if p0 < p1:
                    dataset.append((id1, id0))

    # 1b. shuffle
    dataset = torch.tensor(dataset, dtype=torch.long)
    dataset = dataset[torch.randperm(len(dataset))]

    # 1c. split into train and val
    N_val = int(len(dataset) * 0.05)
    val_dataset = dataset[:N_val]
    train_dataset = dataset[N_val:]

    num_users = dataset.max() + 1
    user_scores = nn.Parameter(torch.zeros(num_users))
    optimizer = optim.Adagrad([user_scores], lr=1e0)

    # cross entropy loss where P(win) = softmax(score0, score1)
    def L(dataset):
        return -user_scores[dataset].log_softmax(-1)[:, 0].mean()

    # run gradient descent to optimize the loss
    for epoch in range(100):
        optimizer.zero_grad()
        train_loss = L(train_dataset) + wd * (user_scores ** 2).mean()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = L(val_dataset)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.5f}  Val Loss: {val_loss:.5f} Mean: {user_scores.abs().mean():.5f} ( {user_scores.min():.5f} - {user_scores.max():.5f} ) "
            )

    return user_scores.tolist()


def make_ratings_table(db):
    member_rows = [
        MemberRow(*row)
        for row in db.execute(
            f"SELECT hashed_userID, hashed_gameID, countryID, status FROM {TABLE_MEMBERS}"
        ).fetchall()
    ]
    press_type = {
        game_id: press_type
        for game_id, press_type in db.execute(
            f"SELECT hashed_id, pressType from redacted_games"
        ).fetchall()
    }

    user_ids = list(set(r.user for r in member_rows))

    max_userid = max(user_ids)
    user_stats = [defaultdict(float) for _ in range(max_userid + 1)]

    member_dict = {(r.game, r.country): r for r in member_rows}
    good_games = list(find_good_games(db))

    print(f"Found {len(good_games)} good games.")

    WIN_STATI = ("Won", "Drawn")
    game_stats = {}
    for ii, game_id in enumerate(good_games):
        if ii & (ii - 1) == 0:
            print(f"Done {ii} / {len(good_games)} games")

        user_ids, winners = [], []
        for country_id in range(1, 7 + 1):
            k = (game_id, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                this_user_stats = user_stats[member_row.user]
                this_user_stats["total"] += 1
                if member_row.status not in STATUS_TO_RANK:
                    continue
                this_user_stats[member_row.status] += 1
                if member_row.status in WIN_STATI:
                    winners.append(this_user_stats)
                user_ids.append(member_dict[k].user)

        # allot points to winners
        for winner in winners:
            winner["total_points"] += 1.0 / len(winners)

        this_game_stats = {"id": game_id, "press_type": press_type[game_id]}
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_id, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                u = member_row.user
                this_game_stats[pwr] = {
                    "id": u,
                    "points": 1.0 / len(winners) if member_row.status in WIN_STATI else 0,
                    "status": member_row.status,
                }
            else:
                this_game_stats[pwr] = None

        game_stats[game_id] = this_game_stats

    print("Computing logit scores")
    ratings = compute_logit_ratings(game_stats)
    for i in range(len(user_stats)):
        user_stats[i]["logit_rating"] = ratings[i]

    print("Adding final stats")
    for ii, game_id in enumerate(good_games):
        this_game_stats = game_stats[game_id]
        if this_game_stats is None:
            continue
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_id, country_id)
            if k in member_dict:
                this_game_stats[pwr].update(**user_stats[member_dict[k].user])

    user_stats = [
        {**user_stats[u], "id": u} for u in range(max_userid) if user_stats[u]["total"] > 0
    ]
    return game_stats, user_stats


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

    game_stats, user_stats = make_ratings_table(db)

    with open(args.out, "w") as f:
        json.dump(game_stats, f)

    with open(os.path.dirname(args.out) + "/user_stats.json", "w") as f:
        json.dump(user_stats, f)
        # keys = list(user_stats[0].keys())
        # f.write(" ".join(keys) + "\n")
        # for user in user_stats:
        #     f.write(" ".join(str(user[k]) if k in user else '0' for k in keys) + "\n")
