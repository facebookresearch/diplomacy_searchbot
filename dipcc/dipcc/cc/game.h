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
#include "json.h"
#include "loc.h"
#include "message.h"
#include "order.h"
#include "phase.h"
#include "power.h"
#include "thirdparty/nlohmann/json.hpp"
#include "unit.h"

using nlohmann::json;

namespace dipcc {

class Game {
public:
  Game(int draw_on_stalemate_years = -1);
  Game(const std::string &json_str);

  void set_orders(const std::string &power,
                  const std::vector<std::string> &orders);

  void process();

  GameState &get_state();

  std::unordered_map<Power, std::unordered_set<Loc>> get_orderable_locations();

  const std::unordered_map<Loc, std::set<Order>> &get_all_possible_orders();

  bool is_game_done() const;

  GameState *get_last_movement_phase(); // can return nullptr

  std::string game_id;

  std::string to_json();

  void rollback_to_phase(const std::string &phase_s);

  std::map<Phase, std::shared_ptr<GameState>> &get_state_history() {
    return state_history_;
  }
  std::map<Phase, std::unordered_map<Power, std::vector<Order>>> &
  get_order_history() {
    return order_history_;
  }

  std::vector<float> get_square_scores() const {
    return state_->get_square_scores();
  }

  void clear_old_all_possible_orders();

  void set_exception_on_convoy_paradox() {
    exception_on_convoy_paradox_ = true;
  }

  void set_draw_on_stalemate_years(int year) {
    draw_on_stalemate_years_ = year;
  }

  // press

  std::map<Phase, std::vector<Message>> &get_message_history() {
    return message_history_;
  }

  void add_message(Power sender, Power recipient, const std::string &body);

  // python

  std::unordered_map<std::string, std::vector<std::string>>
  py_get_all_possible_orders();
  pybind11::dict py_get_state();
  pybind11::dict py_get_orderable_locations();
  std::vector<PhaseData> get_phase_history();
  PhaseData get_phase_data();
  pybind11::dict py_get_message_history();

  static Game from_json(const std::string &s) { return Game(s); }

  std::string get_phase_long() { return state_->get_phase().to_string_long(); }
  std::string get_phase_short() { return state_->get_phase().to_string(); }
  void py_add_message(const std::string &sender, const std::string &recipient,
                      const std::string &body) {
    add_message(power_from_str(sender), power_from_str(recipient), body);
  }

  // mila compat

  std::string map_name() { return "standard"; }
  char phase_type() { return state_->get_phase().phase_type; }

private:
  void crash_dump();
  void maybe_early_exit();

  // Members
  std::shared_ptr<GameState> state_;
  std::unordered_map<Power, std::vector<Order>> staged_orders_;
  std::map<Phase, std::shared_ptr<GameState>> state_history_;
  std::map<Phase, std::unordered_map<Power, std::vector<Order>>> order_history_;
  std::map<Phase, std::vector<Message>> message_history_;
  std::vector<std::string> rules_ = {"NO_PRESS", "POWER_CHOICE"};
  int draw_on_stalemate_years_ = -1;
  bool exception_on_convoy_paradox_ = false;
};

} // namespace dipcc
