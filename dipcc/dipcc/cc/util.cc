#include <glog/logging.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hash.h"
#include "loc.h"
#include "order.h"
#include "util.h"

namespace dipcc {

std::unordered_map<Loc, Order> organize_orders_by_src(
    const std::unordered_map<Power, std::vector<Order>> &orders) {
  std::unordered_map<Loc, Order> r;
  for (auto &it : orders) {
    for (const Order order : it.second) {

      r[root_loc(order.get_unit().loc)] = order;
    }
  }
  return r;
}

bool is_implicit_via(const Order &order,
                     const std::set<Order> &loc_possible_orders) {
  return order.get_type() == OrderType::M && !order.get_via() &&
         // order not possible
         !set_contains(loc_possible_orders, order) &&
         // order with via is possible
         set_contains(loc_possible_orders, order.with_via(true));
}

bool is_implicit_via(
    const Order &order,
    const std::unordered_map<Loc, std::set<Order>> &all_possible_orders) {
  auto it = all_possible_orders.find(order.get_unit().loc);
  return it != all_possible_orders.end() && is_implicit_via(order, it->second);
}

} // namespace dipcc
