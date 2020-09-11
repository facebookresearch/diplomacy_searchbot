#!/usr/bin/env python

import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("min", type=int)
parser.add_argument("max", type=int)
parser.add_argument("power")
args = parser.parse_args()

g = glob.glob(f"{args.dir}/game_{args.power[:3]}.*.json")
nums = {int(x.split(".")[-2]) for x in g}
missing = set(range(args.min, args.max + 1)) - nums

print(",".join(map(str, missing)))
