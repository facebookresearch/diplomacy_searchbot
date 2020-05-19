#pragma once

#include "enums.h"
#include "loc.h"
#include "order.h"
#include "unit.h"

namespace std {

// template <> struct hash<dipcc::Order> {
//   size_t operator()(const dipcc::Order &order) const {
//     std::size_t r = 0;
//     // hash_combine(r, order.get_unit().type);
//     // hash_combine(r, order.get_unit().loc);
//     // hash_combine(r, order.get_type());
//     // hash_combine(r, order.get_target().type);
//     // hash_combine(r, order.get_target().loc);
//     // hash_combine(r, order.get_dest());
//     // hash_combine(r, order.get_via());
//     return r;
//   }
// };

} // namespace std

namespace dipcc {

template <class T> inline void hash_combine(std::size_t &s, const T &v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

struct HashOrder {
  std::size_t operator()(const Order &x) const {
    std::size_t r = 0;
    hash_combine(r, x.get_unit().type);
    hash_combine(r, x.get_unit().loc);
    hash_combine(r, x.get_type());
    hash_combine(r, x.get_target().type);
    hash_combine(r, x.get_target().loc);
    hash_combine(r, x.get_dest());
    hash_combine(r, x.get_via());
    return r;
  }
};

} // namespace dipcc
