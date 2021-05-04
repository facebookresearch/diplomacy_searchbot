/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

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
