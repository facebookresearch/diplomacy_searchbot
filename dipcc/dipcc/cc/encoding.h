/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/power.h"

#define BOARD_STATE_ENC_WIDTH 35
#define PREV_ORDERS_ENC_WIDTH 40
#define PREV_ORDERS_CAPACITY 100

namespace py = pybind11;

namespace dipcc {

void encode_board_state(GameState &state, float *r);
void encode_prev_orders(PhaseData &phase_data, float *r);

} // namespace dipcc
