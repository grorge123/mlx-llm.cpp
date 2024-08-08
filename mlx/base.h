#pragma once

#include "mlx/mlx.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace mx = mlx::core;

namespace mlx::core::nn {
class Module {
public:
  std::string Name;
  std::unordered_map<std::string, mx::array> Parameters{};
  std::unordered_map<std::string, Module *> Submodules{};
  mx::array &registerParameter(std::string Name, array &&W);

  void update(std::unordered_map<std::string, mx::array> Parameters);
  void apply(std::string Key, mx::array Parameters);
  template <typename T> void registerModule(std::string Name, T &&M) {
    using DecayedT = std::decay_t<T>;
    if (!std::is_base_of<Module, DecayedT>::value) {
      throw std::invalid_argument("Invalid subModule.");
    }

    if (Submodules.find(Name) == Submodules.end()) {
      Submodules.insert({Name, &M});
      Submodules.at(Name)->Name = Name;
    }
  }
  template <typename T>
  void registerLayer(std::string Name, std::vector<T> &Layers) {
    if (!std::is_base_of<Module, T>::value) {
      throw std::invalid_argument("Invalid subModule.");
    }
    for (size_t Idx = 0; Idx < Layers.size(); Idx++) {
      registerModule(Name + "." +std::to_string(Idx), Layers[Idx]);
    }
  }
};
} // namespace mlx::core::nn