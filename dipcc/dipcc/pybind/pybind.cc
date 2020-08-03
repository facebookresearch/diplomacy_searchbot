#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../cc/game.h"
#include "encoding.h"
#include "py_game_get_units.h"

namespace py = pybind11;
using namespace dipcc;

PYBIND11_MODULE(pydipcc, m) {
  py::class_<Game>(m, "Game")
      .def(py::init<>())
      .def("process", &Game::process)
      .def("set_orders", &Game::set_orders)
      .def("get_state", &Game::py_get_state, py::return_value_policy::move)
      .def("get_all_possible_orders", &Game::py_get_all_possible_orders)
      .def("get_orderable_locations", &Game::py_get_orderable_locations)
      .def("to_json", &Game::to_json)
      .def("from_json", &Game::from_json)
      .def("get_phase_history", &Game::get_phase_history,
           py::return_value_policy::move)
      .def("get_phase_data", &Game::get_phase_data,
           py::return_value_policy::move)
      .def("rollback_to_phase", &Game::rollback_to_phase)
      .def_property_readonly("is_game_done", &Game::is_game_done)
      .def_property_readonly("phase", &Game::get_phase_long)
      .def_property_readonly("current_short_phase", &Game::get_phase_short)
      .def_readwrite("game_id", &Game::game_id)
      .def("get_current_phase", &Game::get_phase_short)       // mila compat
      .def_property_readonly("map_name", &Game::map_name)     // mila compat
      .def_property_readonly("phase_type", &Game::phase_type) // mila compat
      .def("get_units", &py_game_get_units,
           py::return_value_policy::move) // mila compat
      ;

  py::class_<PhaseData>(m, "PhaseData")
      .def_property_readonly("name", &PhaseData::get_name)
      .def_property_readonly("state", &PhaseData::py_get_state)
      .def_property_readonly("orders", &PhaseData::py_get_orders)
      .def("to_dict", &PhaseData::to_dict);

  m.def("encode_board_state", &encode_board_state,
        py::return_value_policy::move);
  m.def("encode_board_state_from_json", &encode_board_state_from_json,
        py::return_value_policy::move);
  m.def("encode_board_state_from_phase", &encode_board_state_from_phase,
        py::return_value_policy::move);
  m.def("encode_prev_orders", &encode_prev_orders,
        py::return_value_policy::move);
}
