#include <glog/logging.h>

#include "checks.h"
#include "power.h"

namespace dipcc {

std::string power_str(const Power &power) {
  return POWERS_STR.at(static_cast<size_t>(power) - 1); // -1 for NONE
}

Power power_from_str(const std::string &s) {
  for (int i = 0; i < 7; ++i) {
    if (power_str(POWERS[i]) == s) {
      return POWERS[i];
    }
  }
  JCHECK(false, "Bad arg to power_from_str: " + s);
}

} // nampespace dipcc
