#pragma once

#include "phase.h"
#include "power.h"
#include <string>

namespace dipcc {

struct Message {
  Power sender;
  Power recipient;
  Phase phase;
  std::string body;
};

} // namespace dipcc
