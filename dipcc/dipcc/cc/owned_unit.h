#pragma once

#include <string>

#include "enums.h"
#include "loc.h"
#include "power.h"
#include "unit.h"

namespace dipcc {

struct Unit; // forward declare

struct OwnedUnit {
  Power power;
  UnitType type;
  Loc loc;

  std::string to_string() const;

  // Comparator (to enable use as set/map key)
  std::tuple<Power, UnitType, Loc> to_tuple() const;
  bool operator<(const OwnedUnit &other) const;

  // Equality operator, true if members equal
  bool operator==(const OwnedUnit &other) const;
  bool operator!=(const OwnedUnit &other) const { return !operator==(other); }

  // Conversion to unowned unit
  Unit unowned() const;

  // Print operator
  friend std::ostream &operator<<(std::ostream &, const OwnedUnit &);
};
void to_json(json &j, const OwnedUnit &x);

} // namespace dipcc
