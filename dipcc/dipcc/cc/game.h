#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "../pybind/phase_data.h"
#include "enums.h"
#include "game_state.h"
#include "hash.h"
#include "loc.h"
#include "order.h"
#include "phase.h"
#include "power.h"
#include "thirdparty/nlohmann/json.hpp"
#include "unit.h"

using nlohmann::json;

namespace dipcc {

class Game {
public:
  Game();
  Game(const std::string &json_str);

  void set_orders(const std::string &power,
                  const std::vector<std::string> &orders);

  void process();

  GameState &get_state();

  std::unordered_map<Power, std::unordered_set<Loc>> get_orderable_locations();

  const std::unordered_map<Loc, std::set<Order>> &get_all_possible_orders();

  bool is_game_done() const;

  std::string game_id;

  std::string to_json();

  // python

  std::unordered_map<std::string, std::vector<std::string>>
  py_get_all_possible_orders();

  pybind11::dict py_get_state();

  pybind11::dict py_get_orderable_locations();

  std::vector<PhaseData> get_phase_history();

  static Game from_json(const std::string &s) { return Game(s); }

  std::string get_phase_long() { return state_.get_phase().to_string_long(); }

private:
  void crash_dump();

  // Members
  GameState state_;
  std::unordered_map<Power, std::vector<Order>> staged_orders_;
  std::map<Phase, GameState> state_history_;
  std::map<Phase, std::unordered_map<Power, std::vector<Order>>> order_history_;
  std::vector<std::string> rules_ = {"NO_PRESS", "POWER_CHOICE"};
};

} // namespace dipcc
