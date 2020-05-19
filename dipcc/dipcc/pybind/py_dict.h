#pragma once

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/order.h"
#include "../cc/power.h"

namespace py = pybind11;

namespace dipcc {

py::dict
py_orders_to_dict(std::unordered_map<Power, std::vector<Order>> &orders);

py::dict py_state_to_dict(GameState &state);

}; // namespace dipcc
