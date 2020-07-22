#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "checks.h"
#include "game.h"
#include "game_state.h"
#include "loc.h"
#include "power.h"
#include "util.h"

namespace dipcc {

class OrdersEncoder {
public:
  const static int MAX_SEQ_LEN = 17;
  const static int EOS_IDX = -1;

  OrdersEncoder(std::unordered_map<std::string, int> order_vocabulary_to_idx,
                int max_cands);

  // Encode x_valid_orders and x_loc_idxs into pre-allocated memory pointed to
  // by r_order_idxs and r_loc_idxs. Return the sequence length.
  int encode_valid_orders(Power power, GameState &state, int32_t *r_order_idxs,
                          int8_t *r_loc_idxs) const;

  // Encode x_prev_orders into pre-allocated memory pointed to by r.
  void encode_prev_orders_deepmind(Game *game, long *r) const;

  // Decode a batch_size * 7 * 17 array of EOS_IDX-padded order idxs.
  // Returns a 3d vector of string (batch, power, orders)
  std::vector<std::vector<std::vector<std::string>>>
  decode_order_idxs(long *order_idxs, size_t batch_size) const;

  int get_max_cands() const { return max_cands_; }

private:
  // Methods
  int smarter_order_index(const Order &) const;
  std::vector<int> filter_orders_in_vocab(const std::set<Order> &) const;
  std::vector<Loc> get_sorted_actual_orderable_locs(
      const std::unordered_set<Loc> &root_locs,
      const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
          &all_possible_orders) const;

  // Data
  std::unordered_map<std::string, int> order_vocabulary_to_idx_;
  std::vector<std::string> order_vocabulary_;
  int max_cands_;
};

} // namespace dipcc
