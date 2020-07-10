#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


### how to select the conversation messages used in the input sequence #################################
SPEAKER_MSG_ONLY = "speaker_msg_only"
PARTNER_MSG_ONLY = "partner_msg_only"
ALL_MSG = "all_msg_are_selected"

HOW_TO_SELECT_MSG = SPEAKER_MSG_ONLY


### overwrite the joined json ###########################################################
OVERWRITE_JOINED_JSON = False


### the input sequence content ###########################################################
STATE_ONLY = "state_only"
MSG_ONLY = "msg_only"
STATE_AND_MSG = "state_and_msg"

INPUT_SEQ = STATE_AND_MSG
