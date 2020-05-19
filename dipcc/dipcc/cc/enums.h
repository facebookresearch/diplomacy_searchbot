#pragma once

#include <string>
#include <vector>

#include "thirdparty/nlohmann/json.hpp"

using nlohmann::json;

namespace dipcc {

enum class UnitType {
  NONE,
  ARMY,
  FLEET,
};
NLOHMANN_JSON_SERIALIZE_ENUM(UnitType,
                             {{UnitType::ARMY, "A"}, {UnitType::FLEET, "F"}})
std::ostream &operator<<(std::ostream &os, UnitType t);

enum class OrderType {
  NONE,
  H,
  M,
  SH,
  SM,
  C,
  R,
  B,
  D,
};

} // namespace dipcc
