#include <exception>
#include <string>

#include "checks.h"

namespace dipcc {

void JCHECK(bool b, const std::string &msg) {
  if (!b) {
    throw msg;
  }
}

void JCHECK(bool b) {
  if (!b) {
    throw "JCHECK failed";
  }
}

} // namespace dipcc
