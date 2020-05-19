#include <unordered_set>

#include "../cc/game.h"
#include "../cc/thirdparty/nlohmann/json.hpp"
#include "consts.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;

namespace dipcc {

void expect_json_any(const json &j, const function<bool(const json &)> f) {
  for (const json &x : j) {
    if (f(x)) {
      return;
    }
  }
  FAIL();
}

template <class T> void expect_json_contains(const json &j, const T &x) {
  expect_json_any(j, [x](const json &y) { return y.get<T>() == x; });
}

class GameTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

  // Declare member variables here
};

TEST_F(GameTest, TestInitialStateJson) {
  Game game;
  json j = game.get_state();
  LOG(INFO) << "Initial state: " << j;

  // Phase
  EXPECT_EQ(j["phase"].get<string>(), "S1901M");

  // Centers
  EXPECT_EQ(j["centers"]["RUSSIA"].size(), 4);
  expect_json_contains<string>(j["centers"]["RUSSIA"], "STP");

  // Units
  expect_json_contains<string>(j["units"]["RUSSIA"], "F STP/SC");
}

TEST_F(GameTest, TestGetOrderableLocations) {
  Game game;
  auto orderable_locs = game.get_orderable_locations();

  EXPECT_EQ(orderable_locs.size(), 7);
  EXPECT_EQ(orderable_locs[Power::ITALY].size(), 3);
  EXPECT_THAT(orderable_locs[Power::ITALY], testing::Contains(Loc::NAP));
  EXPECT_THAT(orderable_locs[Power::ITALY],
              testing::Not(testing::Contains(Loc::STP)));
}

TEST_F(GameTest, TestOrderParsing1) {
  Order order = Order("F SEV H");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::SEV);
  EXPECT_EQ(order.get_type(), OrderType::H);
}

TEST_F(GameTest, TestOrderParsing2) {
  Order order = Order("F BUL/EC - BLA");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BUL_EC);
  EXPECT_EQ(order.get_type(), OrderType::M);
  EXPECT_EQ(order.get_dest(), Loc::BLA);
}

TEST_F(GameTest, TestOrderParsing3) {
  Order order = Order("F BUL/EC S F BLA");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BUL_EC);
  EXPECT_EQ(order.get_type(), OrderType::SH);
  EXPECT_EQ(order.get_target().type, UnitType::FLEET);
  EXPECT_EQ(order.get_target().loc, Loc::BLA);
}

TEST_F(GameTest, TestOrderParsing4) {
  Order order = Order("F BUL/EC S F BLA - RUM");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BUL_EC);
  EXPECT_EQ(order.get_type(), OrderType::SM);
  EXPECT_EQ(order.get_target().type, UnitType::FLEET);
  EXPECT_EQ(order.get_target().loc, Loc::BLA);
  EXPECT_EQ(order.get_dest(), Loc::RUM);
}

TEST_F(GameTest, TestOrderParsing5) {
  Order order = Order("F BLA S F BUL/EC - RUM");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BLA);
  EXPECT_EQ(order.get_type(), OrderType::SM);
  EXPECT_EQ(order.get_target().type, UnitType::FLEET);
  EXPECT_EQ(order.get_target().loc, Loc::BUL_EC);
  EXPECT_EQ(order.get_dest(), Loc::RUM);
}

TEST_F(GameTest, TestOrderParsing6) {
  Order order = Order("F BLA S A BUL - RUM");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BLA);
  EXPECT_EQ(order.get_type(), OrderType::SM);
  EXPECT_EQ(order.get_target().type, UnitType::ARMY);
  EXPECT_EQ(order.get_target().loc, Loc::BUL);
  EXPECT_EQ(order.get_dest(), Loc::RUM);
}

TEST_F(GameTest, TestOrderParsing7) {
  Order order = Order("F BLA C A BUL - RUM");
  EXPECT_EQ(order.get_unit().type, UnitType::FLEET);
  EXPECT_EQ(order.get_unit().loc, Loc::BLA);
  EXPECT_EQ(order.get_type(), OrderType::C);
  EXPECT_EQ(order.get_target().type, UnitType::ARMY);
  EXPECT_EQ(order.get_target().loc, Loc::BUL);
  EXPECT_EQ(order.get_dest(), Loc::RUM);
}

} // namespace dipcc
