#include "enums.h"
#include "game.h"
#include "loc.h"
#include "owned_unit.h"
#include "unit.h"

#include "thirdparty/nlohmann/json.hpp"

using namespace std;
using nlohmann::json;

namespace dipcc {

void to_json(json &j, const Unit &x) { j = x.to_string(); }

void to_json(json &j, const OwnedUnit &x) { j = x.to_string(); }

void to_json(json &j, const Phase &x) { j = x.to_string(); }

void to_json(json &j, const GameState &x) {
  j["phase"] = x.phase_;

  for (auto &it : x.units_) {
    auto &unit = it.second;
    j["units"][power_str(unit.power)].push_back(unit);
  }

  for (auto &it : x.centers_) {
    j["centers"][power_str(it.second)].push_back(loc_str(it.first));
  }
}
} // namespace dipcc
