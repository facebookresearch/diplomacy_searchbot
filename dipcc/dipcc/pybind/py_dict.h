#pragma once

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/message.h"
#include "../cc/order.h"
#include "../cc/power.h"

namespace py = pybind11;

namespace dipcc {

py::dict
py_orders_to_dict(std::unordered_map<Power, std::vector<Order>> &orders);

py::dict py_state_to_dict(GameState &state);

py::dict py_message_to_dict(const Message &message);

py::list py_messages_to_list(const std::vector<Message> &messages);

py::dict py_message_history_to_dict(
    const std::map<Phase, std::vector<Message>> &message_history);

}; // namespace dipcc
