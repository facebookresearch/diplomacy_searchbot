/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "data_fields.h"

namespace dipcc {

TensorDict new_data_fields_state_only(long B) {
  return {
      {"x_board_state", torch::empty({B, 81, 35}, torch::kFloat32)},
      {"x_prev_state", torch::empty({B, 81, 35}, torch::kFloat32)},
      {"x_prev_orders", torch::empty({B, 2, 100}, torch::kLong)},
      {"x_season", torch::empty({B, 3}, torch::kFloat32)},
      {"x_in_adj_phase", torch::empty({B}, torch::kFloat32)},
      {"x_build_numbers", torch::empty({B, 7}, torch::kFloat32)},
  };
}

TensorDict new_data_fields(long B, long max_seq_len, bool include_power) {
  TensorDict fields(new_data_fields_state_only(B));
  fields["x_loc_idxs"] = torch::full({B, 7, 81}, -1, torch::kInt8);
  fields["x_possible_actions"] =
      torch::full({B, 7, max_seq_len, 469}, -1, torch::kInt32);

  if (include_power) {
    fields["x_power"] = torch::full({B, 7, max_seq_len}, -1, torch::kLong);
  }

  return fields;
}

} // namespace dipcc
