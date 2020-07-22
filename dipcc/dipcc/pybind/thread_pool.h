#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/encoding.h"

namespace py = pybind11;

namespace dipcc {

void py_thread_pool_encode_inputs_multi(
    ThreadPool *thread_pool, std::vector<Game *> &games,
    std::vector<py::array_t<float>> &x_board_state,
    std::vector<py::array_t<float>> &x_prev_state,
    std::vector<py::array_t<long>> &x_prev_orders,
    std::vector<py::array_t<float>> &x_season,
    std::vector<py::array_t<float>> &x_in_adj_phase,
    std::vector<py::array_t<float>> &x_build_numbers,
    std::vector<py::array_t<int8_t>> &x_loc_idxs,
    std::vector<py::array_t<int32_t>> &x_possible_actions,
    std::vector<py::array_t<int32_t>> &x_max_seq_len) {

  size_t size = games.size();

  std::vector<float *> p_board_state(size);
  std::vector<float *> p_prev_state(size);
  std::vector<long *> p_prev_orders(size);
  std::vector<float *> p_season(size);
  std::vector<float *> p_in_adj_phase(size);
  std::vector<float *> p_build_numbers(size);
  std::vector<int8_t *> p_loc_idxs(size);
  std::vector<int32_t *> p_possible_actions(size);
  std::vector<int32_t *> p_max_seq_len(size);

  for (int i = 0; i < size; ++i) {
    p_board_state[i] = x_board_state[i].mutable_data(0);
    p_prev_state[i] = x_prev_state[i].mutable_data(0);
    p_prev_orders[i] = x_prev_orders[i].mutable_data(0);
    p_season[i] = x_season[i].mutable_data(0);
    p_in_adj_phase[i] = x_in_adj_phase[i].mutable_data(0);
    p_build_numbers[i] = x_build_numbers[i].mutable_data(0);
    p_loc_idxs[i] = x_loc_idxs[i].mutable_data(0);
    p_possible_actions[i] = x_possible_actions[i].mutable_data(0);
    p_max_seq_len[i] = x_max_seq_len[i].mutable_data(0);
  }

  thread_pool->encode_inputs_multi(games, p_board_state, p_prev_state,
                                   p_prev_orders, p_season, p_in_adj_phase,
                                   p_build_numbers, p_loc_idxs,
                                   p_possible_actions, p_max_seq_len);
} // py_thread_pool_encode_inputs_multi

std::vector<std::vector<std::vector<std::string>>>
py_decode_order_idxs(ThreadPool *thread_pool, py::array_t<long> &order_idxs) {
  return thread_pool->get_orders_encoder().decode_order_idxs(
      order_idxs.mutable_data(0), order_idxs.shape(0));
}

} // namespace dipcc
