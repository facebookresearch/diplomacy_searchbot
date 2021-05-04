/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/data_fields.h"
#include "../cc/encoding.h"

namespace py = pybind11;

namespace dipcc {

TensorDict
py_thread_pool_encode_inputs_state_only_multi(ThreadPool *thread_pool,
                                              std::vector<Game *> &games) {
  return thread_pool->encode_inputs_state_only_multi(games);
}

TensorDict py_thread_pool_encode_inputs_multi(ThreadPool *thread_pool,
                                              std::vector<Game *> &games) {
  return thread_pool->encode_inputs_multi(games);
}

TensorDict
py_thread_pool_encode_inputs_all_powers_multi(ThreadPool *thread_pool,
                                              std::vector<Game *> &games) {
  return thread_pool->encode_inputs_all_powers_multi(games);
}

std::vector<std::vector<std::vector<std::string>>>
py_decode_order_idxs(ThreadPool *thread_pool, torch::Tensor *order_idxs) {
  return thread_pool->get_orders_encoder().decode_order_idxs(order_idxs);
}

} // namespace dipcc
