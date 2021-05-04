/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <chrono>
#include <exception>
#include <glog/logging.h>

#include "adjacencies.h"
#include "checks.h"
#include "exceptions.h"
#include "game.h"
#include "game_id.h"
#include "json.h"
#include "thirdparty/nlohmann/json.hpp"
#include "util.h"

using nlohmann::json;
using namespace std;

namespace dipcc {

void Game::set_orders(const std::string &power_str,
                      const std::vector<std::string> &order_strs) {
  Power power = power_from_str(power_str);
  auto &staged_orders = staged_orders_[power];

  for (const std::string &order_str : order_strs) {
    if (order_str == "WAIVE") {
      continue;
    }
    Order order(order_str);
    bool overwritten = false;
    for (int i = 0; i < staged_orders.size(); ++i) {
      if (staged_orders[i].get_unit().loc == order.get_unit().loc) {
        // overwrite existing order
        staged_orders[i] = order;
        overwritten = true;
        break;
      }
    }
    if (!overwritten) {
      staged_orders.push_back(order);
    }
  }
}

void Game::process() {
  state_history_[state_->get_phase()] = state_;
  order_history_[state_->get_phase()] = staged_orders_;

  try {
    state_ = std::make_shared<GameState>(
        state_->process(staged_orders_, exception_on_convoy_paradox_));
    maybe_early_exit();
  } catch (const ConvoyParadoxException &e) {
    throw e;
  } catch (const std::exception &e) {
    this->crash_dump();
    LOG(ERROR) << "Exception: " << e.what();
    throw e;
  } catch (...) {
    this->crash_dump();
    LOG(ERROR) << "Unknown exception";
    exit(1);
  }
  staged_orders_.clear();
}

GameState &Game::get_state() { return *state_; }

std::unordered_map<Power, std::vector<Loc>> Game::get_orderable_locations() {
  return state_->get_orderable_locations();
}

const std::unordered_map<Loc, std::set<Order>> &
Game::get_all_possible_orders() {
  return state_->get_all_possible_orders();
}

Game::Game(int draw_on_stalemate_years)
    : state_(std::make_shared<GameState>()),
      draw_on_stalemate_years_(draw_on_stalemate_years) {

  // game id
  this->game_id = gen_game_id();

  // units
  state_->set_unit(Power::AUSTRIA, UnitType::ARMY, Loc::BUD);
  state_->set_unit(Power::AUSTRIA, UnitType::ARMY, Loc::VIE);
  state_->set_unit(Power::AUSTRIA, UnitType::FLEET, Loc::TRI);
  state_->set_unit(Power::ENGLAND, UnitType::FLEET, Loc::EDI);
  state_->set_unit(Power::ENGLAND, UnitType::FLEET, Loc::LON);
  state_->set_unit(Power::ENGLAND, UnitType::ARMY, Loc::LVP);
  state_->set_unit(Power::FRANCE, UnitType::FLEET, Loc::BRE);
  state_->set_unit(Power::FRANCE, UnitType::ARMY, Loc::MAR);
  state_->set_unit(Power::FRANCE, UnitType::ARMY, Loc::PAR);
  state_->set_unit(Power::GERMANY, UnitType::FLEET, Loc::KIE);
  state_->set_unit(Power::GERMANY, UnitType::ARMY, Loc::BER);
  state_->set_unit(Power::GERMANY, UnitType::ARMY, Loc::MUN);
  state_->set_unit(Power::ITALY, UnitType::FLEET, Loc::NAP);
  state_->set_unit(Power::ITALY, UnitType::ARMY, Loc::ROM);
  state_->set_unit(Power::ITALY, UnitType::ARMY, Loc::VEN);
  state_->set_unit(Power::RUSSIA, UnitType::ARMY, Loc::WAR);
  state_->set_unit(Power::RUSSIA, UnitType::ARMY, Loc::MOS);
  state_->set_unit(Power::RUSSIA, UnitType::FLEET, Loc::SEV);
  state_->set_unit(Power::RUSSIA, UnitType::FLEET, Loc::STP_SC);
  state_->set_unit(Power::TURKEY, UnitType::FLEET, Loc::ANK);
  state_->set_unit(Power::TURKEY, UnitType::ARMY, Loc::CON);
  state_->set_unit(Power::TURKEY, UnitType::ARMY, Loc::SMY);

  // centers
  state_->set_center(Loc::BUD, Power::AUSTRIA);
  state_->set_center(Loc::TRI, Power::AUSTRIA);
  state_->set_center(Loc::VIE, Power::AUSTRIA);
  state_->set_center(Loc::EDI, Power::ENGLAND);
  state_->set_center(Loc::LON, Power::ENGLAND);
  state_->set_center(Loc::LVP, Power::ENGLAND);
  state_->set_center(Loc::BRE, Power::FRANCE);
  state_->set_center(Loc::MAR, Power::FRANCE);
  state_->set_center(Loc::PAR, Power::FRANCE);
  state_->set_center(Loc::BER, Power::GERMANY);
  state_->set_center(Loc::KIE, Power::GERMANY);
  state_->set_center(Loc::MUN, Power::GERMANY);
  state_->set_center(Loc::NAP, Power::ITALY);
  state_->set_center(Loc::ROM, Power::ITALY);
  state_->set_center(Loc::VEN, Power::ITALY);
  state_->set_center(Loc::MOS, Power::RUSSIA);
  state_->set_center(Loc::SEV, Power::RUSSIA);
  state_->set_center(Loc::STP, Power::RUSSIA);
  state_->set_center(Loc::WAR, Power::RUSSIA);
  state_->set_center(Loc::ANK, Power::TURKEY);
  state_->set_center(Loc::CON, Power::TURKEY);
  state_->set_center(Loc::SMY, Power::TURKEY);
}

unordered_map<string, vector<string>> Game::py_get_all_possible_orders() {
  const auto &all_possible_orders(this->get_all_possible_orders());
  unordered_map<string, vector<string>> r;

  r.reserve(LOCS.size());
  for (Loc loc : LOCS) {
    r[loc_str(loc)] = std::vector<string>();
  }

  for (const auto &p : all_possible_orders) {
    std::string loc = loc_str(p.first);
    std::string rloc = loc_str(root_loc(p.first));
    r[loc].reserve(p.second.size());
    r[rloc].reserve(p.second.size());
    for (const Order &order : p.second) {
      r[loc].push_back(order.to_string());
      if (loc != rloc) {
        r[rloc].push_back(order.to_string());
      }
    }
  }
  return r;
}

bool Game::is_game_done() const {
  return state_->get_phase().phase_type == 'C';
}

string Game::to_json() {
  json j;

  j["id"] = this->game_id;
  j["map"] = "standard";

  for (auto &rule : rules_) {
    j["rules"].push_back(rule);
  }

  for (auto &q : state_history_) {
    GameState &state = *q.second;

    json phase;
    phase["name"] = state.get_phase().to_string();
    phase["state"] = state.to_json();
    for (auto &p : POWERS) {
      phase["orders"][power_str(p)] = vector<string>();
    }
    for (auto &p : order_history_[state.get_phase()]) {
      string power = power_str(p.first);
      for (Order &order : p.second) {
        phase["orders"][power].push_back(order.to_string());
      }
    }
    phase["messages"] = json::value_type::array(); // mila compat
    for (auto & [ time_sent, msg ] : message_history_[state.get_phase()]) {
      phase["messages"].push_back(msg);
    }

    phase["results"] = json::value_type::object(); // mila compat

    phase["logs"] = json::value_type::array();
    for (auto &data : logs_[state.get_phase()]) {
      phase["logs"].push_back(data);
    }

    j["phases"].push_back(phase);
  }

  // current phase
  json current;
  current["name"] = state_->get_phase().to_string();
  current["state"] = state_->to_json();
  current["orders"] = json::value_type::object();  // mila compat
  current["results"] = json::value_type::object(); // mila compat
  current["messages"] = json::value_type::array(); // mila compat
  for (auto & [ time_sent, msg ] : message_history_[state_->get_phase()]) {
    current["messages"].push_back(msg);
  }
  current["logs"] = json::value_type::array();
  for (auto &data : logs_[state_->get_phase()]) {
    current["logs"].push_back(data);
  }
  for (auto & [ power, orders ] : staged_orders_) {
    for (auto &order : orders) {
      current["orders"][power_str(power)].push_back(order.to_string());
    }
  }
  j["phases"].push_back(current);

  return j.dump();
}

Game::Game(const string &json_str) {
  auto j = json::parse(json_str);

  if (j.find("id") != j.end() && !j["id"].is_null()) {
    this->game_id = j["id"];
  } else if (j.find("game_id") != j.end() && !j["game_id"].is_null()) {
    this->game_id = j["game_id"];
  } else {
    this->game_id = gen_game_id();
  }

  if (!j["rules"].empty()) {
    this->rules_.clear();
    for (string rule : j["rules"]) {
      this->rules_.push_back(rule);
    }
  }

  if (j.find("phases") != j.end()) {
    string phase_str;
    for (auto &j_phase : j["phases"]) {
      phase_str = j_phase["name"];
      state_history_[phase_str] = std::make_shared<GameState>(j_phase["state"]);

      for (auto &it : j_phase["orders"].items()) {
        Power power = power_from_str(it.key());
        for (auto &j_order : it.value()) {
          order_history_[phase_str][power].push_back(Order(j_order));
        }
      }

      if (j_phase.find("messages") != j_phase.end()) {
        for (auto &j_msg : j_phase["messages"]) {
          JCHECK(message_history_[phase_str].find(j_msg["time_sent"]) ==
                     message_history_[phase_str].end(),
                 "from_json duplicate message timestamps not allowed");
          message_history_[phase_str][j_msg["time_sent"]] = j_msg;
        }
      }
      if (j_phase.find("logs") != j_phase.end()) {
        for (auto &data : j_phase["logs"]) {
          logs_[phase_str].push_back(data);
        }
      }
    }
  } else {
    string phase_str;
    for (auto &j_state : j["state_history"].items()) {
      phase_str = j_state.key();
      state_history_[phase_str] = std::make_shared<GameState>(j_state.value());

      if (j["order_history"].find(phase_str) != j["order_history"].end()) {
        for (auto &it : j["order_history"][phase_str].items()) {
          Power power = power_from_str(it.key());
          for (auto &j_order : it.value()) {
            order_history_[phase_str][power].push_back(Order(j_order));
          }
        }
      }

      if (j.find("message_history") == j.end() ||
          j["message_history"].find(phase_str) == j["message_history"].end()) {
        continue;
      }
      for (auto &j_msg : j["message_history"][phase_str]) {
        JCHECK(message_history_[phase_str].find(j_msg["time_sent"]) ==
                   message_history_[phase_str].end(),
               "from_json duplicate message timestamps not allowed");
        message_history_[phase_str][j_msg["time_sent"]] = j_msg;
      }
    }
  }

  // Pop last state as current state
  auto it = state_history_.rbegin();
  Phase current_phase = it->first;
  state_ = it->second;
  state_history_.erase(current_phase);
  if (order_history_.find(current_phase) != order_history_.end()) {
    staged_orders_ = order_history_[current_phase];
    order_history_.erase(current_phase);
  }
}

void Game::crash_dump() {
  json j_orders;
  for (auto &it : staged_orders_) {
    for (auto &order : it.second) {
      j_orders[power_str(it.first)].push_back(order.to_string());
    }
  }

  LOG(ERROR) << "CRASH DUMP";
  std::cerr << this->to_json() << "\n";
  LOG(ERROR) << "ORDERS:";
  std::cerr << j_orders.dump() << "\n";
}

Game Game::rolled_back_to_phase_start(const std::string &phase_s) {
  Game new_game(*this);
  new_game.rollback_to_phase(phase_s, false, false, false);
  return new_game;
}

Game Game::rolled_back_to_phase_end(const std::string &phase_s) {
  Game new_game(*this);
  new_game.rollback_to_phase(phase_s, true, true, true);
  return new_game;
}

void Game::rollback_to_phase(const std::string &phase_s,
                             bool preserve_phase_messages,
                             bool preserve_phase_orders,
                             bool preserve_phase_logs) {
  Phase phase(phase_s);

  // delete message_history_ including (?) and after phase
  auto m_it = message_history_.find(phase);
  if (preserve_phase_messages && m_it != message_history_.end()) {
    ++m_it;
  }
  while (m_it != message_history_.end()) {
    m_it = message_history_.erase(m_it);
  }

  // delete logs_ including (?) and after phase
  auto l_it = logs_.find(phase);
  if (preserve_phase_logs && l_it != logs_.end()) {
    ++l_it;
  }
  while (l_it != logs_.end()) {
    l_it = logs_.erase(l_it);
  }

  // set current state
  if (!preserve_phase_orders) {
    staged_orders_.clear();
  }
  if (state_->get_phase() == phase) {
    return;
  }
  auto s_it = state_history_.find(phase);
  JCHECK(s_it != state_history_.end(), "rollback_to_phase phase not found");
  state_ = s_it->second;

  // delete state_history_ including and after phase
  while (s_it != state_history_.end()) {
    s_it = state_history_.erase(s_it);
  }

  // delete order_history_ including and after phase
  auto o_it = order_history_.find(phase);
  if (preserve_phase_orders) {
    staged_orders_ = o_it->second;
  }
  while (o_it != order_history_.end()) {
    o_it = order_history_.erase(o_it);
  }
}

void Game::rollback_messages_to_timestamp(const uint64_t timestamp) {
  for (auto & [ phase, messages ] : message_history_) {
    (void)phase;
    for (auto it = messages.begin(); it != messages.end(); ++it) {
      if (it->first > timestamp) {
        messages.erase(it, messages.end());
        break;
      }
    }
  }
}

GameState *Game::get_last_movement_phase() {
  for (auto it = state_history_.rbegin(); it != state_history_.rend(); ++it) {
    if (it->first.phase_type == 'M') {
      return state_history_[it->first].get();
    }
  }

  // no previous move phases
  return nullptr;
}

void Game::clear_old_all_possible_orders() {
  for (auto &p : state_history_) {
    p.second->clear_all_possible_orders();
  }
}

void Game::maybe_early_exit() {
  Phase phase = state_->get_phase();

  if (draw_on_stalemate_years_ < 1 ||
      phase.year - 1901 < draw_on_stalemate_years_ || phase.season != 'S' ||
      phase.phase_type != 'M') {
    return;
  }

  for (int i = 1; i <= draw_on_stalemate_years_; ++i) {
    if (state_->get_centers() !=
        state_history_.at(Phase('S', phase.year - i, 'M'))->get_centers()) {
      // no stalemate
      return;
    }
  }

  // we have a stalemate
  DLOG(INFO) << "Game over! Stalemate after " << draw_on_stalemate_years_
             << " years";

  state_->set_phase(phase.completed());
}

void Game::add_message(Power sender, Power recipient, const std::string &body,
                       uint64_t time_sent) {
  if (time_sent == 0) {
    time_sent = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
  }
  auto &phase_messages = message_history_[state_->get_phase()];
  if (phase_messages.find(time_sent) != phase_messages.end()) {
    JFAIL("add_message duplicate timestamps not currently allowed");
  }
  phase_messages[time_sent] =
      Message{sender, recipient, state_->get_phase(), body, time_sent};
}

void Game::add_log(const std::string &data) {
  logs_[state_->get_phase()].push_back(data);
}

std::optional<Phase> Game::get_next_phase(Phase from) {
  if (from == state_->get_phase()) {
    return {};
  }
  auto it = state_history_.find(from);
  if (it == state_history_.end()) {
    JFAIL("get_next_phase phase not found");
  }
  if (from == state_history_.rbegin()->first) {
    return state_->get_phase();
  }
  ++it;
  return it->first;
}

std::optional<Phase> Game::get_prev_phase(Phase from) {
  if (from == Phase("S1901M")) {
    return {};
  }
  if (from == state_->get_phase()) {
    return state_history_.rbegin()->first;
  }
  auto it = state_history_.find(from);
  if (it == state_history_.end()) {
    JFAIL("get_prev_phase phase not found");
  }
  return std::prev(it, 1)->first;
}

} // namespace dipcc
