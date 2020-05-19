#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "../cc/game_state.h"

namespace py = pybind11;

// py::array_t<float> encode_board_state(const dipcc::GameState &state) {
py::array_t<float> encode_board_state() {
  py::array_t<float> r({81, 35});
  // *r.mutable_data(5, 10) = 42;

  return r;
}
