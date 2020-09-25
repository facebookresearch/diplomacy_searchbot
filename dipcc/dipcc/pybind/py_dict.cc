#include "py_dict.h"
#include "../cc/checks.h"
#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/power.h"
#include <glog/logging.h>

using namespace std;
namespace py = pybind11;

namespace dipcc {

py::dict py_orders_to_dict(unordered_map<Power, vector<Order>> &orders) {
  py::dict d;

  for (auto &it : orders) {
    auto py_power = py::cast<string>(power_str(it.first));
    auto list = py::list();
    for (Order &order : it.second) {
      list.append(py::cast<string>(order.to_string()));
    }
    d[py_power] = list;
  }

  return d;
}

py::dict py_state_to_dict(GameState &state) {
  py::dict d;

  auto &all_possible_orders(state.get_all_possible_orders());

  // builds
  d["builds"] = py::dict();
  for (Power power : POWERS) {
    auto power_py_str = py::cast<std::string>(power_str(power));
    d["builds"][power_py_str] = py::dict();
    py::list homes_list;
    if (state.get_phase().phase_type == 'A') {
      for (Loc center : home_centers(power)) {
        auto it = all_possible_orders.find(center);
        if (it != all_possible_orders.end() && it->second.size() > 0) {
          homes_list.append(loc_str(center));
        }
      }
      d["builds"][power_py_str]["count"] = std::min(
          state.get_n_builds(power), static_cast<int>(homes_list.size()));
    } else {
      d["builds"][power_py_str]["count"] = 0;
    }
    d["builds"][power_py_str]["homes"] = homes_list;
  }

  // centers
  d["centers"] = py::dict();
  for (Power power : POWERS) {
    d["centers"][py::cast<std::string>(power_str(power))] = py::list();
  }
  for (const auto &p : state.get_centers()) {
    static_cast<py::list>(
        d["centers"][py::cast<std::string>(power_str(p.second))])
        .append(loc_str(p.first));
  }

  // homes
  d["homes"] = py::dict();
  for (Power power : POWERS) {
    py::list homes;
    for (Loc center : home_centers(power)) {
      auto owner_it = state.get_centers().find(center);
      if (owner_it != state.get_centers().end() && owner_it->second == power) {
        homes.append(loc_str(center));
      }
    }
    d["homes"][py::cast<std::string>(power_str(power))] = homes;
  }

  // name
  d["name"] = py::cast<std::string>(state.get_phase().to_string());

  // retreats
  d["retreats"] = py::dict();
  for (Power power : POWERS) {
    d["retreats"][py::cast<std::string>(power_str(power))] = py::dict();
  }
  if (state.get_phase().phase_type == 'R') {
    for (OwnedUnit &unit : state.get_dislodged_units()) {
      auto key = py::cast<std::string>(unit.unowned().to_string());
      py::list retreats;
      auto orders_it = all_possible_orders.find(unit.loc);
      JCHECK(orders_it != all_possible_orders.end(),
             "Dislodged unit has no retreat orders: " + loc_str(unit.loc));
      for (const Order &order : orders_it->second) {
        if (order.get_type() == OrderType::R) {
          retreats.append(loc_str(order.get_dest()));
        }
      }
      d["retreats"][py::cast<std::string>(power_str(unit.power))][key] =
          retreats;
    }
  }

  // units
  d["units"] = py::dict();
  for (Power power : POWERS) {
    d["units"][py::cast<std::string>(power_str(power))] = py::list();
  }
  for (auto &p : state.get_units()) {
    OwnedUnit unit = p.second;
    if (unit.type == UnitType::NONE) {
      LOG(WARNING) << "UnitType::NONE in py_state_to_dict units, loc="
                   << (unit.loc == Loc::NONE ? "NONE" : loc_str(unit.loc));
    }
    static_cast<py::list>(
        d["units"][py::cast<std::string>(power_str(unit.power))])
        .append(unit.unowned().to_string());
  }
  for (OwnedUnit unit : state.get_dislodged_units()) {
    if (unit.type == UnitType::NONE) {
      LOG(WARNING) << "UnitType::NONE in py_state_to_dict dislodged_units, loc="
                   << (unit.loc == Loc::NONE ? "NONE" : loc_str(unit.loc));
    }
    static_cast<py::list>(
        d["units"][py::cast<std::string>(power_str(unit.power))])
        .append("*" + unit.unowned().to_string());
  }

  return d;
}

py::dict Game::py_get_state() {
  auto d(py_state_to_dict(this->get_state()));

  // DEBUGGING
  if (this->get_state().get_phase().phase_type == 'R') {
    for (auto &p : this->get_state().get_all_possible_orders()) {
      if (this->get_state().get_unit_rooted(p.first).type == UnitType::NONE) {
        LOG(WARNING) << "Found weird case, logging crash dump: " << p.first;
        this->crash_dump();
      }
    }
  }
  // !DEBUGGING

  return d;
}

py::dict py_message_to_dict(const Message &message) {
  py::dict d;
  d["sender"] = power_str(message.sender);
  d["recipient"] = power_str(message.recipient);
  d["phase"] = message.phase.to_string();
  d["body"] = message.body;
  return d;
}

py::list py_messages_to_list(const std::vector<Message> &messages) {
  py::list lst;
  for (const auto &m : messages) {
    lst.append(py_message_to_dict(m));
  }
  return lst;
}

py::dict py_message_history_to_dict(
    const std::map<Phase, std::vector<Message>> &message_history) {
  py::dict d;
  for (auto & [ phase, messages ] : message_history) {
    d[py::cast<std::string>(phase.to_string())] = py_messages_to_list(messages);
  }
  return d;
}

}; // namespace dipcc
