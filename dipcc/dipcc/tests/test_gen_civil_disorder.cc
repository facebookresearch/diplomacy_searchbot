/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <queue>
#include <unordered_set>
#include <utility>

#include "../cc/adjacencies.h"
#include "../cc/game.h"
#include "../cc/loc.h"
#include "../cc/thirdparty/nlohmann/json.hpp"
#include "../cc/util.h"
#include "consts.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;

namespace dipcc {

class TestGenCivilDisorder : public ::testing::Test {};

int get_civil_disorder_distance(Loc init, const vector<Loc> &targets,
                                bool is_army) {
  queue<pair<Loc, int>> todo;
  todo.push(make_pair(init, 0));
  set<Loc> visited{};

  while (todo.size() > 0) {
    pair<Loc, int> cur = todo.front();
    todo.pop();

    Loc loc = cur.first;
    int dist = cur.second;

    if (set_contains(visited, loc)) {
      continue;
    }
    visited.insert(loc);

    if (vec_contains(targets, root_loc(loc))) {
      return dist;
    }

    if (is_army) {
      // armies can travel over water for the purpose of civil disorder
      // distance calculations
      for (Loc loc_var : expand_coasts(loc)) {
        for (Loc adj_loc : ADJ_A[static_cast<int>(loc_var)]) {
          todo.push(make_pair(adj_loc, dist + 1));
        }
        for (Loc adj_loc : ADJ_F[static_cast<int>(loc_var)]) {
          todo.push(make_pair(adj_loc, dist + 1));
        }
      }
    } else {
      for (Loc adj_loc : ADJ_F[static_cast<int>(loc)]) {
        todo.push(make_pair(adj_loc, dist + 1));
      }
    }
  }

  return -1;
}

TEST_F(TestGenCivilDisorder, TestGenCivilDisorder) {
  for (Power power : POWERS) {
    vector<int> dists_a(81, -1);
    vector<int> dists_f(81, -1);

    for (Loc loc : LOCS) {
      dists_a[static_cast<int>(loc) - 1] =
          get_civil_disorder_distance(loc, home_centers(power), true);
      dists_f[static_cast<int>(loc) - 1] =
          get_civil_disorder_distance(loc, home_centers(power), false);
    }

    std::cout << power_str(power) << " ARMY {";
    for (int d : dists_a) {
      std::cout << d << ",";
    }
    std::cout << "}\n";

    std::cout << power_str(power) << " FLEET {";
    for (int d : dists_f) {
      std::cout << d << ",";
    }
    std::cout << "}\n";
  }
}

} // namespace dipcc
