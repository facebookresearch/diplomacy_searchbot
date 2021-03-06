/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <unordered_set>

#include "../cc/game.h"
#include "../cc/thirdparty/nlohmann/json.hpp"
#include "consts.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;

namespace dipcc {

class PhaseTest : public ::testing::Test {};

TEST_F(PhaseTest, TestPhaseNE) {
  ASSERT_TRUE(!(Phase("S1902R") == Phase("S1902M")));
  ASSERT_TRUE(Phase("S1902R") < Phase("S1902M") ||
              Phase("S1902M") < Phase("S1902R"));
}

} // namespace dipcc
