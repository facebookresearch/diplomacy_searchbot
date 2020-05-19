#pragma once

#include <exception>
#include <string>

namespace dipcc {

void JCHECK(bool b, const std::string &msg);
void JCHECK(bool b);

} // namespace dipcc
