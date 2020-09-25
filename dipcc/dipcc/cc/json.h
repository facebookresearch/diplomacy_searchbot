#pragma once

#include "thirdparty/nlohmann/json.hpp"

using nlohmann::json;

namespace dipcc {

void to_json(json &j, const Unit &x);
void to_json(json &j, const OwnedUnit &x);
void to_json(json &j, const Phase &x);
void to_json(json &j, const GameState &x);
void to_json(json &j, const Message &x);
void from_json(const json &j, Message &x);

} // namespace dipcc
