#pragma once

#include "mlx/mlx.h"
#include <iostream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace mx = mlx::core;
#define assumingUnreachable() __builtin_unreachable()

namespace mlx::core::nn {
class Module : public std::enable_shared_from_this<Module> {
public:
  std::string Name;
  std::unordered_map<std::string, mx::array> Parameters{};
  std::unordered_map<std::string, std::shared_ptr<Module>> Submodules{};
  mx::array &registerParameter(std::string Name, array &&W);
  std::unordered_map<std::string, mx::array>
  getWeigts(const std::string &Prefix = "model");
  virtual std::shared_ptr<nn::Module> toQuantized(int GroupSize = 64,
                                                  int Bits = 4);
  void update(std::unordered_map<std::string, mx::array> Parameters);
  void apply(std::string Key, mx::array Parameters);
  template <typename T>
  void registerModule(std::string ModuleName, std::shared_ptr<T> M) {
    using DecayedT = std::decay_t<T>;
    if (!std::is_base_of<Module, DecayedT>::value) {
      spdlog::error("Invalid subModule.");
      assumingUnreachable();
    }

    if (Submodules.find(ModuleName) == Submodules.end()) {
      Submodules.insert({ModuleName, M});
      Submodules.at(ModuleName)->Name = ModuleName;
    } else {
      spdlog::error("Module already exists.");
      assumingUnreachable();
    }
  }
  template <typename T>
  void registerLayer(std::string ModuleName,
                     std::vector<std::shared_ptr<T>> &Layers) {
    if (!std::is_base_of<Module, T>::value) {
      spdlog::error("Invalid subModule.");
      assumingUnreachable();
    }
    for (size_t Idx = 0; Idx < Layers.size(); Idx++) {
      registerModule(ModuleName + "." + std::to_string(Idx), Layers[Idx]);
    }
  }
};
} // namespace mlx::core::nn

template <typename T> void printVec(std::vector<T> Ve) {
  for (auto I : Ve) {
    spdlog::debug("{} ", I);
  }
}