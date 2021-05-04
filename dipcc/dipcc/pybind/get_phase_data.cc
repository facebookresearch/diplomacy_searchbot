/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/game.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

PhaseData Game::get_phase_data() {
  // Does NOT return the current staged orders, nor the current phase's messages.
  // This is intentional, to match the old diplomacy research python implementation,
  // even though this is a little weird and inconsistent.
  return PhaseData(*state_, std::unordered_map<Power, std::vector<Order>>());
}

} // namespace dipcc
