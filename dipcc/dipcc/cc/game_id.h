#pragma once

#include <random>
#include <string>

namespace dipcc {

// Return 12 random base64 chars
std::string gen_game_id() {
  const static std::string ALPHA =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<> distribution(0, ALPHA.size() - 1);

  std::string r;
  for (int i = 0; i < 12; ++i) {
    r += ALPHA[distribution(generator)];
  }
  return r;
}

} // namespace dipcc
