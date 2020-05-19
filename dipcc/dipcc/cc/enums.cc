#include "enums.h"

namespace dipcc {

std::ostream &operator<<(std::ostream &os, UnitType t) {
  switch (t) {
  case UnitType::NONE:
    return os << "NONE";
  case UnitType::ARMY:
    return os << "ARMY";
  case UnitType::FLEET:
    return os << "FLEET";
  }
}

} // namespace dipcc
