#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Last Run: July 7, 2020
Simple script that preprocesses redacted_messages jsons and adds 
fields from the the master sqlite table
"""
import sqlite3
import json
import os
import argparse
from tqdm import tqdm
from glob import glob

DB2JSON_FIELD_TUPLE = ("hashed_id", "hashed_gameID")


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[str(col[0])] = str(row[idx])
    return d


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("field", type=str)
    parser.add_argument("table", type=str)
    parser.add_argument(
        "--datapath",
        type=str,
        default="/checkpoint/fairdiplomacy/chat_messages_json_runthree/"
        "large_jsons_nostream/redacted_messages_*.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/checkpoint/fairdiplomacy/processed_chat_jsons/variantID",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="/checkpoint/fairdiplomacy/facebook_notext.sqlite3",
    )
    args = parser.parse_args()

    print("Connecting to sqlite db...")
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = dict_factory
    cur = conn.cursor()
    cur.execute(f"select {DB2JSON_FIELD_TUPLE[0]}, {args.field} from {args.table}")
    hashedID2field_list = cur.fetchall()
    hashedID2field_dict = {
        d[DB2JSON_FIELD_TUPLE[0]]: d[args.field] for d in hashedID2field_list
    }
    print("Fields fetched...")
    conn.close()

    tot = len(glob(args.datapath))
    for i, fle in enumerate(glob(args.datapath)):
        print(f"[ Processing data from path {i} / {tot}: {fle} ... ]")
        with open(fle, "r") as f:
            json_data = json.load(f)

        print("Processing json...")
        for json_entry in tqdm(json_data[2]["data"]):
            json_entry[f"{args.field}"] = str(
                hashedID2field_dict[json_entry[DB2JSON_FIELD_TUPLE[1]]]
            )

        save_file = os.path.join(args.save_dir, os.path.basename(fle))
        with open(save_file, "w") as f:
            json.dump(json_data, f)

        print(f"Updated json saved to {save_file}")

    print("Done!")
