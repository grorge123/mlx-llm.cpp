#include "base.h"
#include "../model/utils.h"
#include "spdlog/spdlog.h"
#include <mlx/array.h>
#include <unordered_map>

namespace mlx::core::nn {

mx::array &Module::registerParameter(std::string Name, mx::array &&W) {
  Parameters.insert({Name, W});
  return Parameters.at(Name);
}
void Module::update(std::unordered_map<std::string, mx::array> Parameters) {
  for (auto &[K, V] : Parameters) {
    apply(K, V);
  }
}
nn::Module *Module::toQuantized(int GroupSize, int Bits) {
  for (auto &[K, V] : Submodules) {
    auto *OldModule = V;
    V = V->toQuantized(GroupSize, Bits);
    if (OldModule != V) {
      delete OldModule;
    }
  }
  return this;
}
void Module::apply(std::string Key, mx::array Value) {
  std::vector<std::string> SplitKey = splitString(Key, '.');
  if (SplitKey.size() == 1) {
    if (Parameters.find(Key) == Parameters.end()) {
      spdlog::error("Unsupported weight: {}", Key);
      assumingUnreachable();
    }
    this->Parameters.at(Key) = Value;
  } else {
    std::string LayerName = SplitKey[0];
    SplitKey.erase(SplitKey.begin());
    if (LayerName == "layers") {
      LayerName += "." + SplitKey[0];
      SplitKey.erase(SplitKey.begin());
    }
    if (Submodules.find(LayerName) == Submodules.end()) {
      spdlog::error("[WASI-NN] MLX backend: Unsupported Layer: {}", LayerName);
      assumingUnreachable();
    }
    Submodules.at(LayerName)->apply(joinString(SplitKey, '.'), Value);
  }
}
std::unordered_map<std::string, mx::array>
Module::getWeigts(const std::string &Prefix) {
  std::unordered_map<std::string, mx::array> Weights;
  for (auto &[K, V] : Submodules) {
    auto Subweights = V->getWeigts(Prefix + Name + ".");
    Weights.insert(Subweights.begin(), Subweights.end());
  }
  for (auto &[K, V] : Parameters) {
    Weights.insert({Prefix + Name + "." + K, V});
  }
  return Weights;
}
} // namespace mlx::core::nn