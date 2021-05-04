/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <string>
#include <vector>

#include "../cc/game.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

vector<PhaseData> Game::get_phase_history() {
  vector<PhaseData> r;
  r.reserve(state_history_.size());

  for (auto &it : state_history_) {
    string name = it.first.to_string();
    r.push_back(PhaseData(*it.second, order_history_[name], message_history_[name]));
  }

  return r;
}

} // namespace dipcc
