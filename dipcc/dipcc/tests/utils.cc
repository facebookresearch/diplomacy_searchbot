#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "utils.h"

using namespace std;

namespace dipcc {

void assert_possible_orders_eq(const std::set<Order> &got,
                               const std::set<Order> &exp,
                               const std::string &tag) {
  {
    set<Order> empty_set;
    set<Order> missing;
    set<Order> extra;
    set_difference(exp.begin(), exp.end(), got.begin(), got.end(),
                   inserter(missing, missing.end()));
    set_difference(got.begin(), got.end(), exp.begin(), exp.end(),
                   inserter(extra, extra.end()));

    ASSERT_THAT(missing, testing::ContainerEq(empty_set))
        << "Missing orders for " << tag;

    for (auto it = extra.begin(); it != extra.end();) {
      {
        Order order = *it;
        if (order.get_type() == OrderType::SM &&
            got.find(Order(order.get_unit(), OrderType::C, order.get_target(),
                           order.get_dest())) != got.end()) {
          {
            LOG(WARNING) << "[" << tag
                         << "] extra order: " << order.to_string();
            it = extra.erase(it);
          }
        } else {
          { ++it; }
        }
      }
    }

    ASSERT_THAT(extra, testing::ContainerEq(empty_set)) << "Extra orders for "
                                                        << tag;
  }
}

} // namespace dipcc
