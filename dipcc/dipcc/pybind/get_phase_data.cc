#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/game.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

PhaseData Game::get_phase_data() {
  return PhaseData(*state_, std::unordered_map<Power, std::vector<Order>>());
}

} // namespace dipcc
