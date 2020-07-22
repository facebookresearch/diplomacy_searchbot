#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/encoding.h"
#include "../cc/thirdparty/nlohmann/json.hpp"

namespace py = pybind11;

namespace dipcc {

py::array_t<float> py_encode_board_state(GameState &state) {
  py::array_t<float> r({81, 35});
  encode_board_state(state, r.mutable_data(0, 0));
  return r;
}

py::array_t<float> py_encode_prev_orders(PhaseData &phase_data) {
  py::array_t<float> r({81, 40});
  encode_prev_orders(phase_data, r.mutable_data(0, 0));
  return r;
}

py::array_t<float>
py_encode_board_state_from_json(const std::string &json_str) {
  auto j = json::parse(json_str);
  GameState state(j);
  return py_encode_board_state(state);
}

py::array_t<float> py_encode_board_state_from_phase(PhaseData &phase) {
  return py_encode_board_state(phase.get_state());
}

} // namespace dipcc
