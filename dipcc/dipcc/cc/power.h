/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <string>
#include <vector>

#include "thirdparty/nlohmann/json.hpp"

namespace dipcc {

enum class Power {
  NONE,
  AUSTRIA,
  ENGLAND,
  FRANCE,
  GERMANY,
  ITALY,
  RUSSIA,
  TURKEY,
};

const Power POWERS[] = {Power::AUSTRIA, Power::ENGLAND, Power::FRANCE,
                        Power::GERMANY, Power::ITALY,   Power::RUSSIA,
                        Power::TURKEY};

const std::vector<std::string> POWERS_STR{
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
};

std::string power_str(const Power &power);
Power power_from_str(const std::string &s);

NLOHMANN_JSON_SERIALIZE_ENUM(Power, {{Power::AUSTRIA, "AUSTRIA"},
                                     {Power::ENGLAND, "ENGLAND"},
                                     {Power::FRANCE, "FRANCE"},
                                     {Power::GERMANY, "GERMANY"},
                                     {Power::ITALY, "ITALY"},
                                     {Power::RUSSIA, "RUSSIA"},
                                     {Power::TURKEY, "TURKEY"}})

} // namespace dipcc
