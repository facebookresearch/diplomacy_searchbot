#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/loc.h"
#include "../cc/power.h"
#include "../cc/thirdparty/nlohmann/json.hpp"

#define S_ARMY 0
#define S_FLEET 1
#define S_UNIT_NONE 2
#define S_AUS 3
#define S_ENG 4
#define S_FRA 5
#define S_ITA 6
#define S_GER 7
#define S_RUS 8
#define S_TUR 9
#define S_POW_NONE 10
#define S_BUILDABLE 11
#define S_REMOVABLE 12
#define S_DIS_ARMY 13
#define S_DIS_FLEET 14
#define S_DIS_UNIT_NONE 15
#define S_DIS_AUS 16
#define S_DIS_ENG 17
#define S_DIS_FRA 18
#define S_DIS_ITA 19
#define S_DIS_GER 20
#define S_DIS_RUS 21
#define S_DIS_TUR 22
#define S_DIS_POW_NONE 23
#define S_LAND 24
#define S_WATER 25
#define S_COAST 26
#define S_SC_AUS 27
#define S_SC_ENG 28
#define S_SC_FRA 29
#define S_SC_ITA 30
#define S_SC_GER 31
#define S_SC_RUS 32
#define S_SC_TUR 33
#define S_SC_POW_NONE 34

#define O_ARMY 0
#define O_FLEET 1
#define O_UNIT_NONE 2
#define O_AUS 3
#define O_ENG 4
#define O_FRA 5
#define O_ITA 6
#define O_GER 7
#define O_RUS 8
#define O_TUR 9
#define O_POW_NONE 10
#define O_HOLD 11
#define O_MOVE 12
#define O_SUPPORT 13
#define O_CONVOY 14
#define O_ORDER_NONE 15
#define O_SRC_AUS 16
#define O_SRC_ENG 17
#define O_SRC_FRA 18
#define O_SRC_ITA 19
#define O_SRC_GER 20
#define O_SRC_RUS 21
#define O_SRC_TUR 22
#define O_SRC_NONE 23
#define O_DEST_AUS 24
#define O_DEST_ENG 25
#define O_DEST_FRA 26
#define O_DEST_ITA 27
#define O_DEST_GER 28
#define O_DEST_RUS 29
#define O_DEST_TUR 30
#define O_DEST_NONE 31
#define O_SC_AUS 32
#define O_SC_ENG 33
#define O_SC_FRA 34
#define O_SC_ITA 35
#define O_SC_GER 36
#define O_SC_RUS 37
#define O_SC_TUR 38
#define O_SC_NONE 39

namespace py = pybind11;

namespace dipcc {

py::array_t<float> encode_board_state(GameState &state) {
  py::array_t<float> r({81, 35});
  memset(r.mutable_data(0, 0), 0, 81 * 35 * sizeof(float));

  //////////////////////////////////////
  // unit type, unit power, removable //
  //////////////////////////////////////

  std::vector<bool> filled(81, false);

  for (auto p : state.get_units()) {
    OwnedUnit unit = p.second;
    JCHECK(unit.type != UnitType::NONE, "UnitType::NONE");
    JCHECK(unit.loc != Loc::NONE, "Loc::NONE");
    JCHECK(unit.power != Power::NONE, "Power::NONE");

    bool removable =
        state.get_phase().season == 'W' && state.get_n_builds(unit.power) < 0;

    size_t loc_i = static_cast<int>(unit.loc) - 1;
    *r.mutable_data(loc_i, unit.type == UnitType::ARMY ? S_ARMY : S_FLEET) = 1;
    *r.mutable_data(loc_i, S_AUS + static_cast<int>(unit.power) - 1) = 1;
    *r.mutable_data(loc_i, S_REMOVABLE) = static_cast<float>(removable);
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      *r.mutable_data(rloc_i, unit.type == UnitType::ARMY ? S_ARMY : S_FLEET) =
          1;
      *r.mutable_data(rloc_i, S_AUS + static_cast<int>(unit.power) - 1) = 1;
      *r.mutable_data(rloc_i, S_REMOVABLE) = static_cast<float>(removable);
      filled[rloc_i] = true;
    }
  }

  // Set locs with no units
  for (int i = 0; i < 81; ++i) {
    if (!filled[i]) {
      *r.mutable_data(i, S_UNIT_NONE) = 1;
      *r.mutable_data(i, S_POW_NONE) = 1;
    }
  }

  ///////////////
  // buildable //
  ///////////////

  if (state.get_phase().phase_type == 'A') {
    for (auto &p : state.get_all_possible_orders()) {
      auto order = p.second.begin();
      if (order->get_type() == OrderType::B) {
        Loc loc = order->get_unit().loc;
        size_t loc_i = static_cast<int>(loc) - 1;
        *r.mutable_data(loc_i, S_BUILDABLE) = 1;
      }
    }
  }

  /////////////////////
  // dislodged units //
  /////////////////////

  std::fill(filled.begin(), filled.end(), false);
  for (OwnedUnit unit : state.get_dislodged_units()) {
    size_t loc_i = static_cast<int>(unit.loc) - 1;
    *r.mutable_data(loc_i,
                    unit.type == UnitType::ARMY ? S_DIS_ARMY : S_DIS_FLEET) = 1;
    *r.mutable_data(loc_i, S_DIS_AUS + static_cast<int>(unit.power) - 1) = 1;
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      *r.mutable_data(rloc_i, unit.type == UnitType::ARMY ? S_DIS_ARMY
                                                          : S_DIS_FLEET) = 1;
      *r.mutable_data(rloc_i, S_DIS_AUS + static_cast<int>(unit.power) - 1) = 1;
      filled[rloc_i] = true;
    }
  }

  // Set locs with no dislodged units
  for (int i = 0; i < 81; ++i) {
    if (!filled[i]) {
      *r.mutable_data(i, S_DIS_UNIT_NONE) = 1;
      *r.mutable_data(i, S_DIS_POW_NONE) = 1;
    }
  }

  ///////////////
  // Area type //
  ///////////////

  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (is_water(loc)) {
      *r.mutable_data(i, S_WATER) = 1;
    } else if (is_coast(loc)) {
      *r.mutable_data(i, S_COAST) = 1;
    } else {
      *r.mutable_data(i, S_LAND) = 1;
    }
  }

  ///////////////////
  // supply center //
  ///////////////////

  auto centers = state.get_centers();
  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (!is_center(loc) || loc != root_loc(loc)) {
      continue;
    }

    auto it = centers.find(loc);
    Power power = it == centers.end() ? Power::NONE : it->second;
    int off = power == Power::NONE ? 7 : static_cast<int>(power) - 1;

    for (Loc cloc : expand_coasts(loc)) {
      int cloc_i = static_cast<int>(cloc) - 1;
      *r.mutable_data(cloc_i, S_SC_AUS + off) = 1;
    }
  }

  return r;
} // encode_board_state

py::array_t<float> encode_prev_orders(PhaseData &phase_data) {
  JCHECK(phase_data.get_state().get_phase().phase_type == 'M',
         "encode_prev_orders called on non-movement phase");

  py::array_t<float> r({81, 40});
  memset(r.mutable_data(0, 0), 0, 81 * 40 * sizeof(float));

  // Store owner of each loc: unit owner if there is a unit, otherwise SC
  // owner, otherwise none (7)
  std::vector<int> loc_owner(81, 7);

  ////////////////////
  // supply centers //
  ////////////////////

  std::vector<bool> filled(81, false);
  for (auto &p : phase_data.get_state().get_centers()) {
    Loc loc = p.first;
    Power power = p.second;
    int power_i = static_cast<int>(power) - 1;
    for (Loc cloc : expand_coasts(loc)) {
      int cloc_i = static_cast<int>(cloc) - 1;

      loc_owner[cloc_i] = power_i;
      *r.mutable_data(cloc_i, O_SC_AUS + power_i) = 1;
      filled[cloc_i] = true;
    }
  }

  // Set unowned SC
  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (!filled[i] && is_center(root_loc(loc))) {
      *r.mutable_data(i, O_SC_NONE) = 1;
    }
  }

  // set owner: units
  for (auto &p : phase_data.get_state().get_units()) {
    OwnedUnit unit = p.second;
    JCHECK(unit.type != UnitType::NONE, "UnitType::NONE");
    JCHECK(unit.loc != Loc::NONE, "Loc::NONE");
    JCHECK(unit.power != Power::NONE, "Power::NONE");

    int loc_i = static_cast<int>(unit.loc) - 1;
    int rloc_i = static_cast<int>(root_loc(unit.loc)) - 1;
    int power_i = static_cast<int>(unit.power) - 1;

    for (Loc cloc : expand_coasts(unit.loc)) {
      int cloc_i = static_cast<int>(cloc) - 1;
      loc_owner[cloc_i] = power_i;
    }
  }

  ////////////
  // orders //
  ////////////

  std::fill(filled.begin(), filled.end(), false);
  for (auto &it : phase_data.get_orders()) {
    Power power = it.first;
    for (auto &order : it.second) {
      Unit unit = order.get_unit();
      int loc_i = static_cast<int>(unit.loc) - 1;
      int rloc_i = static_cast<int>(root_loc(unit.loc)) - 1;

      if (phase_data.get_state().get_unit(unit.loc) != unit.owned_by(power)) {
        // ignore bad order
        continue;
      }

      // Unit type, power
      *r.mutable_data(loc_i, unit.type == UnitType::ARMY ? O_ARMY : O_FLEET) =
          1;
      *r.mutable_data(rloc_i, unit.type == UnitType::ARMY ? O_ARMY : O_FLEET) =
          1;
      *r.mutable_data(loc_i, O_AUS + static_cast<int>(power) - 1) = 1;
      *r.mutable_data(rloc_i, O_AUS + static_cast<int>(power) - 1) = 1;

      // Order Type
      OrderType order_type = order.get_type();
      int order_type_idx;
      if (order_type == OrderType::H) {
        order_type_idx = O_HOLD;
      } else if (order_type == OrderType::M) {
        order_type_idx = O_MOVE;
      } else if (order_type == OrderType::C) {
        order_type_idx = O_CONVOY;
      } else if (order_type == OrderType::SM || order_type == OrderType::SH) {
        order_type_idx = O_SUPPORT;
      }
      *r.mutable_data(loc_i, order_type_idx) = 1;
      *r.mutable_data(rloc_i, order_type_idx) = 1;

      // Src power
      if (order_type == OrderType::SH || order_type == OrderType::SM ||
          order_type == OrderType::C) {
        int src_idx =
            O_SRC_AUS + loc_owner[static_cast<int>(order.get_target().loc) - 1];
        *r.mutable_data(loc_i, src_idx) = 1;
        *r.mutable_data(rloc_i, src_idx) = 1;
      } else {
        *r.mutable_data(loc_i, O_SRC_NONE) = 1;
        *r.mutable_data(rloc_i, O_SRC_NONE) = 1;
      }

      // Dest power
      if (order_type == OrderType::M || order_type == OrderType::SM ||
          order_type == OrderType::C) {
        int dest_idx =
            O_DEST_AUS + loc_owner[static_cast<int>(order.get_dest()) - 1];
        *r.mutable_data(loc_i, dest_idx) = 1;
        *r.mutable_data(rloc_i, dest_idx) = 1;
      } else {
        *r.mutable_data(loc_i, O_DEST_NONE) = 1;
        *r.mutable_data(rloc_i, O_DEST_NONE) = 1;
      }

      filled[loc_i] = true;
      filled[rloc_i] = true;
    } // for (order : orders)
  }   // for (power : powers)

  // Fill locations with no orders
  for (int i = 0; i < 81; ++i) {
    if (filled[i]) {
      continue;
    }

    *r.mutable_data(i, O_UNIT_NONE) = 1;
    *r.mutable_data(i, O_POW_NONE) = 1;
    *r.mutable_data(i, O_ORDER_NONE) = 1;
    *r.mutable_data(i, O_SRC_NONE) = 1;
    *r.mutable_data(i, O_DEST_NONE) = 1;
  }

  return r;
} // encode_prev_orders

py::array_t<float> encode_board_state_from_json(const std::string &json_str) {
  auto j = json::parse(json_str);
  GameState state(j);
  return encode_board_state(state);
}

py::array_t<float> encode_board_state_from_phase(PhaseData &phase) {
  return encode_board_state(phase.get_state());
}

} // namespace dipcc
