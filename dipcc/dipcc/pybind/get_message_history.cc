#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/game.h"
#include "py_dict.h"

using namespace std;

namespace dipcc {

py::dict Game::py_get_message_history() {
  return py_message_history_to_dict(message_history_);
}

} // namespace dipcc
