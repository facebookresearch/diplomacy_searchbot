#pragma once

#include <set>
#include <string>

#include "../cc/order.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace dipcc {

void assert_possible_orders_eq(const std::set<Order> &got,
                               const std::set<Order> &exp,
                               const std::string &tag);

} // namespace dipcc
