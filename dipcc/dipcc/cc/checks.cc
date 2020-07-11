#include <exception>
#include <stdexcept>
#include <string>

#include "checks.h"

namespace dipcc {

void JCHECK(bool b, const std::string &msg) {
  if (!b) {
    throw std::runtime_error(msg);
  }
}

void JCHECK(bool b) {
  if (!b) {
    throw std::runtime_error("JCHECK failed");
  }
}

} // namespace dipcc
