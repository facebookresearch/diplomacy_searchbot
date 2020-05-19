#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}



MAX_PER_DIR=50

# SRC_DIR=/tmp/blah/6hgames/
SRC_DIR=/checkpoint/jsgray/diplomacy/mila_dataset/data/

GAMES=$(ls $SRC_DIR \
    | grep -v game_96000_json \
    | grep -v game_97600_json \
    | shuf -n $MAX_PER_DIR --random-source=<(get_seeded_random 42) \
    | grep -v game_130375_json \
    | grep -v game_129670_json \
    | grep -v game_13196_json \
    | grep -v game_134282_json \
    | grep -v game_7484_json \
)

for x in $GAMES
do
    python render_equivalence_test.py $SRC_DIR/$x \
        | grep -v "The following errors were encountered" \
        | grep -v "^--" \
        > ../dipcc/tests/test_eq_$(echo $x | sed 's/.json//').cc
done
