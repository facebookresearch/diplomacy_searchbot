#pragma once

#include <unordered_map>

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/power.h"
#include "py_dict.h"

namespace py = pybind11;

namespace dipcc {

// forward declares
py::dict
py_orders_to_dict(std::unordered_map<Power, std::vector<Order>> &orders);
py::dict py_state_to_dict(GameState &state);
// !forward declares

class PhaseData {
public:
  PhaseData(const GameState &state,
            const std::unordered_map<Power, std::vector<Order>> &orders) {
    name_ = state.get_phase().to_string();
    state_ = state;
    orders_ = orders;
  }

  py::dict py_get_state() { return py_state_to_dict(state_); }

  py::dict py_get_orders() { return py_orders_to_dict(orders_); }

  py::dict to_dict() {
    py::dict d;
    d["name"] = name_;
    d["state"] = py_get_state();
    d["orders"] = py_get_orders();
    return d;
  }

  const std::string get_name() const { return name_; }

private:
  // Members
  std::string name_;
  GameState state_;
  std::unordered_map<Power, std::vector<Order>> orders_;
};

} // namespace dipcc
