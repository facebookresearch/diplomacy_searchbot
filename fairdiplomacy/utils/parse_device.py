# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def parse_device(device):
    try:
        if device.startswith("cuda") or device.startswith("cpu"):
            return device
    except AttributeError:
        pass

    return f"cuda:{device}"
