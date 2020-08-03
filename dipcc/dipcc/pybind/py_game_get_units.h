#pragma once

#include "../cc/game.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dipcc {

py::dict py_game_get_units(Game *game) { return game->py_get_state()["units"]; }
};
