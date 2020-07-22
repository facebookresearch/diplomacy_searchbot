#include "orders_encoder.h"
#include <algorithm>
#include <glog/logging.h>
#include <string>
#include <utility>
#include <vector>

#define P_IDX(r, w, i, j) (*((r) + ((i) * (w)) + (j)))

#define PREV_ORDERS_WIDTH 100

using namespace std;

namespace dipcc {

// forward declares
vector<string> get_compound_build_orders(
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders,
    vector<Loc> orderable_locs, int n_builds);

// Constructor
OrdersEncoder::OrdersEncoder(
    std::unordered_map<std::string, int> order_vocabulary_to_idx, int max_cands)
    : max_cands_(max_cands), order_vocabulary_to_idx_(order_vocabulary_to_idx) {

  // init order_vocabulary_
  int max_idx = 0;
  for (auto &p : order_vocabulary_to_idx_) {
    if (p.second > max_idx) {
      max_idx = p.second;
    }
  }
  order_vocabulary_.resize(max_idx + 1);
  for (auto &p : order_vocabulary_to_idx_) {
    order_vocabulary_[p.second] = p.first;
  }
}

void OrdersEncoder::encode_prev_orders_deepmind(Game *game, long *r) const {
  memset(r, 0, 2 * PREV_ORDERS_WIDTH * sizeof(long));

  vector<pair<int32_t, int8_t>> prev_orders;
  prev_orders.reserve(100);

  for (auto it = game->get_order_history().rbegin();
       it != game->get_order_history().rend(); ++it) {

    for (auto jt : it->second) {
      Power power = jt.first;

      for (const Order &order : jt.second) {
        auto x = order_vocabulary_to_idx_.find(order.to_string());
        if (x != order_vocabulary_to_idx_.end()) {
          int32_t order_idx = x->second;
          int8_t loc_idx = static_cast<int>(order.get_unit().loc) - 1;
          prev_orders.push_back(make_pair(order_idx, loc_idx));
        }
      }
    }

    // Encode up to and including the most recent movement phase
    if (it->first.phase_type == 'M') {
      break;
    }
  }

  JCHECK(prev_orders.size() < PREV_ORDERS_WIDTH,
         "prev_orders exceeds max size");

  // Sort for deterministic encoding
  sort(prev_orders.begin(), prev_orders.end());

  // Add to preallocated tensor
  for (int i = 0; i < prev_orders.size(); ++i) {
    pair<int32_t, int8_t> p = prev_orders[i];
    P_IDX(r, PREV_ORDERS_WIDTH, 0, i) = p.first;
    P_IDX(r, PREV_ORDERS_WIDTH, 1, i) = p.second;
  }

} // encode_prev_orders_deepmind

int OrdersEncoder::encode_valid_orders(Power power, GameState &state,
                                       int32_t *r_order_idxs,
                                       int8_t *r_loc_idxs) const {
  // Init return value: all_order_idxs
  // py::array_t<int32_t> all_order_idxs({1, MAX_SEQ_LEN, max_cands_});
  memset(r_order_idxs, EOS_IDX, MAX_SEQ_LEN * max_cands_ * sizeof(int32_t));

  // Init return value: loc_idxs
  // py::array_t<int8_t> loc_idxs({1, 81});
  memset(r_loc_idxs, -1, 81 * sizeof(int8_t));

  // Early exit?
  auto orderable_locs_it = state.get_orderable_locations().find(power);
  if (orderable_locs_it == state.get_orderable_locations().end() ||
      orderable_locs_it->second.size() == 0) {
    return 0;
  }

  // Get orderable_locs sorted by coast-specific loc idx (orderable_locs returns
  // root_locs)
  auto &all_possible_orders(state.get_all_possible_orders());
  vector<Loc> orderable_locs(get_sorted_actual_orderable_locs(
      orderable_locs_it->second, all_possible_orders));

  int n_builds = state.get_n_builds(power);
  if (n_builds > 0) {
    // builds phase
    n_builds = min(n_builds, static_cast<int>(orderable_locs.size()));
    vector<string> orders(get_compound_build_orders(all_possible_orders,
                                                    orderable_locs, n_builds));
    vector<int> order_idxs(orders.size());
    for (int j = 0; j < orders.size(); ++j) {
      order_idxs[j] = order_vocabulary_to_idx_.at(orders[j]);
    }
    sort(order_idxs.begin(), order_idxs.end());
    for (int j = 0; j < orders.size(); ++j) {
      P_IDX(r_order_idxs, max_cands_, 0, j) = order_idxs[j];
    }
    for (Loc loc : orderable_locs) {
      r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = -2;
    }
    return n_builds;

  } else if (n_builds < 0) {
    // disband phase
    int n_disbands = -n_builds;
    vector<int> order_idxs;
    order_idxs.reserve(orderable_locs.size());
    for (Loc loc : orderable_locs) {
      for (int idx : filter_orders_in_vocab(all_possible_orders.at(loc))) {
        order_idxs.push_back(idx);
      }
    }
    sort(order_idxs.begin(), order_idxs.end());
    for (int i = 0; i < n_disbands; ++i) {
      for (int j = 0; j < order_idxs.size(); ++j) {
        P_IDX(r_order_idxs, max_cands_, i, j) = order_idxs[j];
      }
    }
    for (Loc loc : orderable_locs) {
      r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = -2;
    }
    return n_disbands;

  } else {
    // move or retreat phase
    for (int i = 0; i < orderable_locs.size(); ++i) {
      Loc loc = orderable_locs[i];
      vector<int> order_idxs(
          filter_orders_in_vocab(all_possible_orders.at(loc)));
      sort(order_idxs.begin(), order_idxs.end());
      for (int j = 0; j < order_idxs.size(); ++j) {
        P_IDX(r_order_idxs, max_cands_, i, j) = order_idxs[j];
        r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = i;
      }
    }
    return orderable_locs.size();
  }
} // encode_valid_orders

// Decode a batch_size * 7 * 17 array of EOS_IDX-padded order idxs.
// Returns a 3d vector of string (batch, power, orders)
vector<vector<vector<string>>>
OrdersEncoder::decode_order_idxs(long *order_idxs, size_t batch_size) const {
  vector<vector<vector<string>>> r(batch_size);
  for (size_t b = 0; b < batch_size; ++b) {
    auto &rb = r[b];
    rb.resize(7);

    for (int p = 0; p < 7; ++p) {
      auto &rbp = rb[p];
      rbp.reserve(17);

      for (int i = 0; i < 17; ++i, ++order_idxs) {
        long order_idx = *order_idxs;
        if (order_idx == EOS_IDX) {
          continue;
        }
        string order = order_vocabulary_[order_idx];
        for (size_t start = 0, end = 0; end != string::npos; start = end + 1) {
          end = order.find(';', start);
          rbp.push_back(order.substr(start, end - start));
        }
      }
    }
  }

  return r;
} // decode_order_idxs

vector<int>
OrdersEncoder::filter_orders_in_vocab(const set<Order> &orders) const {
  vector<int> idxs;
  idxs.reserve(orders.size());

  for (const Order &order : orders) {
    int idx = smarter_order_index(order);
    if (idx != -1) {
      idxs.push_back(idx);
    }
  }

  return idxs;
}

int OrdersEncoder::smarter_order_index(const Order &order) const {
  string order_s(order.to_string());
  auto it = order_vocabulary_to_idx_.find(order_s);
  if (it != order_vocabulary_to_idx_.end()) {
    return it->second;
  }

  // Try order with no coasts
  string order_s_no_coasts;
  order_s_no_coasts.reserve(order_s.size());
  for (int i = 0; i < order_s.size();) {
    char c = order_s[i];
    if (c == '/') {
      i += 3; // skip coast
    } else {
      order_s_no_coasts += c;
      i += 1;
    }
  }

  it = order_vocabulary_to_idx_.find(order_s_no_coasts);
  if (it != order_vocabulary_to_idx_.end()) {
    return it->second;
  }

  // Give up
  return -1;
}

vector<Loc> OrdersEncoder::get_sorted_actual_orderable_locs(
    const unordered_set<Loc> &root_locs,
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders)
    const {
  vector<Loc> locs;
  locs.reserve(root_locs.size());

  for (Loc rloc : root_locs) {
    auto &coasts = expand_coasts(rloc);
    if (coasts.size() == 1) {
      locs.push_back(rloc);
    } else {
      for (Loc cloc : coasts) {
        auto it = all_possible_orders.find(cloc);
        if (it != all_possible_orders.end() && it->second.size() > 0) {
          locs.push_back(cloc);
          break;
        }
      }
    }
  }

  sort(locs.begin(), locs.end());
  return locs;
}

// See combinations()
void combinations_impl(int min, int n, int c, vector<int> &v,
                       function<void(const vector<int> &)> foo) {
  for (int i = min; i <= (n - c); ++i) {
    v.push_back(i);
    if (c == 1) {
      foo(v);
    } else {
      combinations_impl(i + 1, n, c - 1, v, foo);
    }
    v.pop_back();
  }
}

// Call foo() once with each unique c-len combination of integers in [0, n-1]
void combinations(int n, int c, function<void(const vector<int> &)> foo) {
  JCHECK(n >= c, "Called combinations with n < c");
  vector<int> v;
  v.reserve(c);
  combinations_impl(0, n, c, v, foo);
}

vector<string> get_compound_build_orders(
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders,
    vector<Loc> orderable_locs, int n_builds) {

  vector<string> r;
  r.reserve(64);

  combinations(orderable_locs.size(), n_builds,
               [&](const vector<int> &orderable_locs_idxs) {
                 vector<vector<Order>> order_lists;
                 order_lists.resize(n_builds);
                 int product = 1;

                 for (int i = 0; i < n_builds; ++i) {
                   for (const Order &order : all_possible_orders.at(
                            orderable_locs[orderable_locs_idxs[i]])) {
                     order_lists[i].push_back(order);
                   }
                   product *= order_lists[i].size();
                 }

                 vector<int> counter(n_builds, 0);

                 vector<string> orders_to_cat;
                 orders_to_cat.resize(n_builds);

                 for (int i = 0; i < product; ++i) {

                   // Gather orders to cat
                   for (int j = 0; j < n_builds; ++j) {
                     orders_to_cat[j] =
                         (order_lists[j][counter[j]].to_string());
                   }
                   sort(orders_to_cat.begin(), orders_to_cat.end());

                   // Cat orders and add to final list
                   string cat(orders_to_cat[0]);
                   for (int k = 1; k < orders_to_cat.size(); ++k) {
                     cat += ";";
                     cat += orders_to_cat[k];
                   }
                   r.push_back(cat);

                   // Increase counter
                   for (int j = n_builds - 1; j >= 0; --j) {
                     if (counter[j] == order_lists[j].size() - 1) {
                       counter[j] = 0;
                     } else {
                       counter[j] += 1;
                       break;
                     }
                   }
                 }
               });

  return r;
}

} // namespace dipcc
