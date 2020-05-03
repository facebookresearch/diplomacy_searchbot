#!/usr/bin/env python

import argparse
import glob
import math
import random
import torch
from fairdiplomacy.data.dataset import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-pct", type=float, default=0.01)
    args = parser.parse_args()

    datasets = [torch.load(d) for g in args.glob for d in glob.glob(g)]
    random.shuffle(datasets)
    n_val = math.ceil(args.val_pct * len(datasets))
    val_ds, train_ds = datasets[:n_val], datasets[n_val:]

    val_d = Dataset.from_merge(val_ds)
    train_d = Dataset.from_merge(train_ds)

    torch.save((train_d, val_d), args.out)
