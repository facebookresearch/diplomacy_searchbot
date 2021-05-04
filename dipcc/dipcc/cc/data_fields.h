/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <torch/torch.h>

namespace dipcc {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;

TensorDict new_data_fields_state_only(long B);

TensorDict new_data_fields(long B, long max_seq_len = 17,
                           bool include_power = false);

} // namespace dipcc
