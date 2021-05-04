# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def sort_phase_key(phase):
    if phase == "COMPLETED":
        return (1e6,)
    else:
        return (
            int(phase[1:5]),
            {"S": 0, "F": 1, "W": 2}[phase[0]],
            {"M": 0, "R": 1, "A": 2}[phase[5]],
        )
