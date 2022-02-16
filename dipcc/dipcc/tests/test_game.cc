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

TEST_F(GameTest, TestDrawOnStalemateOff) {
  Game game;
  game.set_orders("RUSSIA", {"F SEV - RUM"});
  game.process(); // S1901M
  game.process(); // F1901M
  game.set_orders("RUSSIA", {"F SEV B"});
  for (int i = 0; i < 10; ++i) {
    game.process();
  }
  EXPECT_EQ(game.is_game_done(), false);
}

TEST_F(GameTest, TestDrawOnStalemateTwo) {
  Game game(2);
  game.set_orders("RUSSIA", {"F SEV - RUM"});
  game.process(); // S1901M
  game.process(); // F1901M
  game.set_orders("RUSSIA", {"F SEV B"});
  while (game.get_state().get_phase().year < 1904) {
    EXPECT_EQ(game.is_game_done(), false);
    game.process();
  }
  EXPECT_EQ(game.is_game_done(), true);
}

TEST_F(GameTest, TestDrawOnStalemateOne) {
  Game game(1);
  game.process(); // S1901M
  game.process(); // F1901M
  EXPECT_EQ(game.is_game_done(), true);
}

TEST_F(GameTest, TestHashSameBounce) {
  Game game;
  game.set_orders("FRANCE", {"A PAR - BUR"});
  game.set_orders("FRANCE", {"A MAR - BUR"});
  game.process();

  Game game2;
  game2.process();
  ASSERT_EQ(game.compute_board_hash(), game2.compute_board_hash());
}

TEST_F(GameTest, TestHashDifferent) {
  Game game;
  game.set_orders("FRANCE", {"A PAR - BUR"});
  game.process();
  Game game2;
  game2.process();
  ASSERT_NE(game.compute_board_hash(), game2.compute_board_hash());
}

TEST_F(GameTest, TestGameRollback) {
  Game game;
  game.set_orders("FRANCE", {"A PAR - BUR"});
  game.process();
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1901M");
  auto units = game.get_state().get_units();
  EXPECT_EQ(units.find(Loc::PAR), units.end());
  EXPECT_NE(units.find(Loc::BUR), units.end());

  game = game.rolled_back_to_phase_start("S1901M"); // preserve_orders=False
  EXPECT_EQ(game.get_state().get_phase().to_string(), "S1901M");
  units = game.get_state().get_units();
  EXPECT_NE(units.find(Loc::PAR), units.end());
  EXPECT_EQ(units.find(Loc::BUR), units.end());

  // check orders not preserved
  game.process();
  units = game.get_state().get_units();
  EXPECT_NE(units.find(Loc::PAR), units.end()); // unit still in PAR
}

TEST_F(GameTest, TestGameRollbackPreserveOrders) {
  Game game;
  game.set_orders("FRANCE", {"A PAR - BUR"});
  game.process();
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1901M");
  auto units = game.get_state().get_units();
  EXPECT_NE(units.find(Loc::BUR), units.end()); // unit moved to BUR

  game = game.rolled_back_to_phase_end("S1901M"); // preserve_orders=True
  EXPECT_EQ(game.get_state().get_phase().to_string(), "S1901M");
  units = game.get_state().get_units();
  EXPECT_NE(units.find(Loc::PAR), units.end()); // unit back in PAR

  // check orders preserved
  game.process();
  units = game.get_state().get_units();
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1901M");
  EXPECT_NE(units.find(Loc::BUR), units.end()); // unit moved to BUR
}

TEST_F(GameTest, TestGameNextPhase) {
  Game game;
  EXPECT_FALSE(game.get_next_phase(Phase("S1901M")).has_value());
  EXPECT_FALSE(game.get_prev_phase(Phase("S1901M")).has_value());

  game.process(); // F1901M
  EXPECT_EQ(game.get_next_phase(Phase("S1901M")).value(), Phase("F1901M"));
  EXPECT_FALSE(game.get_prev_phase(Phase("S1901M")).has_value());
  EXPECT_FALSE(game.get_next_phase(Phase("F1901M")).has_value());
  EXPECT_EQ(game.get_prev_phase(Phase("F1901M")).value(), Phase("S1901M"));

  game.process(); // S1902M
  EXPECT_EQ(game.get_next_phase(Phase("F1901M")).value(), Phase("S1902M"));
  EXPECT_EQ(game.get_prev_phase(Phase("F1901M")).value(), Phase("S1901M"));
}

TEST_F(GameTest, TestGameRollbackMessages) {
  Game game;
  game.process();
  game.add_message(Power::RUSSIA, PowerOrAll::TURKEY, "hi Turkey, it's F1901M", 1111);
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1901M");
  game.process();
  game.add_message(Power::TURKEY, PowerOrAll::RUSSIA, "hi Russia, it's S1902M", 2222);
  EXPECT_EQ(game.get_state().get_phase().to_string(), "S1902M");
  game.process();
  game.process();
  game.process();

  // check phase start
  Game game_S1902M_start = game.rolled_back_to_phase_start("S1902M");
  EXPECT_EQ(game_S1902M_start.get_message_history()[Phase("F1901M")].size(), 1);
  EXPECT_EQ(game_S1902M_start.get_message_history()[Phase("S1902M")].size(), 0);

  // check phase end
  Game game_S1902M_end = game.rolled_back_to_phase_end("S1902M");
  EXPECT_EQ(game_S1902M_end.get_message_history()[Phase("F1901M")].size(), 1);
  EXPECT_EQ(game_S1902M_end.get_message_history()[Phase("S1902M")].size(), 1);

  // check same phase
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1903M");
  game.add_message(Power::AUSTRIA, PowerOrAll::ENGLAND, "hi England, it's F1903M", 9999);

  Game game_same_end = game.rolled_back_to_phase_end("F1903M");
  EXPECT_EQ(game_same_end.get_message_history()[Phase("F1903M")].size(), 1);

  Game game_same_start = game.rolled_back_to_phase_start("F1903M");
  EXPECT_EQ(game_same_start.get_message_history()[Phase("F1903M")].size(), 0);

  // check same phase staged orders not preserved with _start()
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1903M");
  game.set_orders("RUSSIA", {"F SEV - RUM"});
  Game game_no_preserve = game.rolled_back_to_phase_start("F1903M");
  game_no_preserve.process();
  EXPECT_EQ(game_no_preserve.get_state().get_phase().to_string(),
            "S1904M"); // no move so no W phase

  // check same phase staged orders preserved with _end()
  EXPECT_EQ(game.get_state().get_phase().to_string(), "F1903M");
  game.set_orders("RUSSIA", {"F SEV - RUM"});
  Game game_yes_preserve = game.rolled_back_to_phase_end("F1903M");
  game_yes_preserve.process();
  EXPECT_EQ(game_yes_preserve.get_state().get_phase().to_string(),
            "W1903A"); // move preserved so W phase
}

} // namespace dipcc
