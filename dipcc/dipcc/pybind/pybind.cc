/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../cc/exceptions.h"
#include "../cc/game.h"
#include "../cc/thread_pool.h"
#include "encoding.h"
#include "py_game_get_units.h"
#include "thread_pool.h"

namespace py = pybind11;
using namespace dipcc;

PYBIND11_MODULE(pydipcc, m) {
  // class Game
  py::class_<Game>(m, "Game")
      .def(py::init<int>(), py::arg("draw_on_stalemate_years") = -1)
      .def(py::init<const Game &>())
      .def("process", &Game::process)
      .def("set_orders", &Game::set_orders)
      .def("get_state", &Game::py_get_state, py::return_value_policy::move)
      .def("get_all_possible_orders", &Game::py_get_all_possible_orders)
      // Returns a dict power -> list of orderable locations. The list will be
      // sorted in the same way as x_possible_actions.
      .def("get_orderable_locations", &Game::py_get_orderable_locations)
      .def("to_json", &Game::to_json)
      .def("from_json", &Game::from_json)
      .def("get_phase_history", &Game::get_phase_history,
           py::return_value_policy::move)
      .def("get_phase_data", &Game::get_phase_data,
           py::return_value_policy::move)
      .def_property_readonly("message_history", &Game::py_get_message_history,
                             py::return_value_policy::move)
      .def_property_readonly("messages", &Game::py_get_messages,
                             py::return_value_policy::move)
      .def("get_logs", &Game::py_get_logs, py::return_value_policy::move)
      .def("add_log", &Game::add_log)
      .def("add_message", &Game::py_add_message, py::arg("sender"),
           py::arg("recipient"), py::arg("body"),
           py::arg_v("time_sent", 0,
                     "Time sent, in micros since epoch. Default 0 means use "
                     "current system time"))
      .def("rolled_back_to_phase_start", &Game::rolled_back_to_phase_start)
      .def("rolled_back_to_phase_end", &Game::rolled_back_to_phase_end)
      .def("rollback_messages_to_timestamp",
           &Game::rollback_messages_to_timestamp)
      .def_property_readonly("is_game_done", &Game::is_game_done)
      .def_property_readonly("phase", &Game::get_phase_long)
      .def_property_readonly("current_short_phase", &Game::get_phase_short)
      .def_readwrite("game_id", &Game::game_id)
      .def("get_current_phase", &Game::get_phase_short)       // mila compat
      .def_property_readonly("map_name", &Game::map_name)     // mila compat
      .def_property_readonly("phase_type", &Game::phase_type) // mila compat
      .def("get_orders", &Game::py_get_orders)                // mila compat
      .def("get_units", &py_game_get_units,
           py::return_value_policy::move) // mila compat
      .def("get_square_scores", &Game::get_square_scores)
      .def("clear_old_all_possible_orders",
           &Game::clear_old_all_possible_orders)
      .def("set_exception_on_convoy_paradox",
           &Game::set_exception_on_convoy_paradox)
      .def("compute_board_hash", &Game::compute_board_hash)
      .def("set_draw_on_stalemate_years", &Game::set_draw_on_stalemate_years)
      .def("get_alive_powers",
           [](Game &game) {
             const auto scores = game.get_square_scores();
             std::vector<std::string> powers;
             for (size_t i = 0; i < scores.size(); ++i) {
                  if (scores[i] >= 1e-3) {
                       powers.push_back(POWERS_STR[i]);
                  }
             }
             return powers;
           })
      .def("get_alive_power_ids",
           [](Game &game) {
             const auto scores = game.get_square_scores();
             std::vector<int> powers;
             for (size_t i = 0; i < scores.size(); ++i) {
                  if (scores[i] >= 1e-3) {
                       powers.push_back(i);
                  }
             }
             return powers;
           })
      .def("get_next_phase",
           [](Game &game, const std::string &p) {
             auto x = game.get_next_phase(Phase(p));
             return x ? static_cast<py::object>(py::str(x->to_string()))
                      : py::none();
           })
      .def("get_prev_phase", [](Game &game, const std::string &p) {
        auto x = game.get_prev_phase(Phase(p));
        return x ? static_cast<py::object>(py::str(x->to_string()))
                 : py::none();
      });

  // class PhaseData
  py::class_<PhaseData>(m, "PhaseData")
      .def_property_readonly("name", &PhaseData::get_name)
      .def_property_readonly("state", &PhaseData::py_get_state)
      .def_property_readonly("orders", &PhaseData::py_get_orders)
      .def_property_readonly("messages", &PhaseData::py_get_messages)
      .def("to_dict", &PhaseData::to_dict);

  // class ThreadPool
  py::class_<ThreadPool, std::shared_ptr<ThreadPool>>(m, "ThreadPool")
      .def(py::init<size_t, std::unordered_map<std::string, int>, int>())
      .def("process_multi", &ThreadPool::process_multi)
      .def("encode_inputs_multi", &py_thread_pool_encode_inputs_multi)
      .def("encode_inputs_all_powers_multi",
           &py_thread_pool_encode_inputs_all_powers_multi)
      .def("encode_inputs_state_only_multi",
           &py_thread_pool_encode_inputs_state_only_multi)
      .def("decode_order_idxs", &py_decode_order_idxs);

  // encoding functions
  m.def("encode_board_state", &py_encode_board_state,
        py::return_value_policy::move);
  m.def("encode_board_state_from_json", &py_encode_board_state_from_json,
        py::return_value_policy::move);
  m.def("encode_board_state_from_phase", &py_encode_board_state_from_phase,
        py::return_value_policy::move);
  m.def("encode_prev_orders", &py_encode_prev_orders,
        py::return_value_policy::move);

  // Exceptions
  py::register_exception<ConvoyParadoxException>(m, "ConvoyParadoxException");
}
