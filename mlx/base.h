#pragma once

#include "mlx/mlx.h"
#include <iostream>
#include <stdexcept>
#include <string>
namespace mx = mlx::core;

namespace mlx::core::nn {
template <typename T>
std::string getName(std::string Prelimiter, const T Value) {
  std::ostringstream Oss;
  if (Prelimiter.empty()) {
    Oss << Value;
  } else {
    Oss << Prelimiter << "." << Value;
  }
  return Oss.str();
}

template <typename T>
std::string getName(std::string &Prelimiter, const T Value1, const T Value2) {
  std::ostringstream Oss;
  if (Prelimiter.empty()) {
    Oss << Value1 << "." << Value2;
  } else {
    Oss << Prelimiter << "." << Value1 << "." << Value2;
  }
  return Oss.str();
}
class Module {
public:
  std::string Name;
  std::unordered_map<std::string, mx::array> Parameters{};
  std::unordered_map<std::string, Module *> Submodules{};
  StreamOrDevice Device = metal::is_available() ? Device::gpu : Device::cpu;
  void update(std::unordered_map<std::string, mx::array> Weight);
  void loadWeights(std::string Weight);
  mx::array &registerParameter(std::string Name, array &&W);
  template <typename T> void registerModule(std::string Name, T &&M) {
    if (!std::is_base_of<T, Module>::value) {
      throw std::invalid_argument("Invalid subModule.");
    }

    if (Submodules.find(Name) == Submodules.end()) {
      Submodules.insert({Name, &M});
      Submodules.at(Name)->Name = Name;
      // Submodules.at(Name)->named_parameters();
    }
  }
  template <typename T>
  void registerLayer(std::string Name, std::vector<T> &Layers) {
    if (!std::is_base_of<T, Module>::value) {
      throw std::invalid_argument("Invalid subModule.");
    }
    for (size_t Idx = 0; Idx < Layers.size(); Idx++) {
      registerModule(getName(Name, Idx), Layers[Idx]);
    }
  }
};
} // namespace mlx::core::nn