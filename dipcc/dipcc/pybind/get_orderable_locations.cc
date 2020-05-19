#include <string>
#include <vector>

#include "../cc/game.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

py::dict Game::py_get_orderable_locations() {
  py::dict d;

  for (Power power : POWERS) {
    d[py::cast<string>(power_str(power))] = py::list();
  }

  for (auto &it : this->get_orderable_locations()) {
    auto power_s = py::cast<string>(power_str(it.first));
    py::list list;
    for (Loc loc : it.second) {
      list.append(loc_str(loc));
    }
    d[power_s] = list;
  }

  return d;
}

} // namespace dipcc
