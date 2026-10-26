#include "base.h"
#include "../model/utils.h"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <memory>
#include <mlx/array.h>
#include <mlx/ops.h>
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
std::shared_ptr<nn::Module> Module::toQuantized(
    int GroupSize, int Bits, const std::string &Prefix,
    const std::unordered_map<std::string, mx::array> &Parameters) {
  auto NewPrefix = Prefix + Name + (Prefix.empty() && Name.empty() ? "" : ".");
  for (auto &[K, V] : Submodules) {
    if (V->hasQuantize()) {
      auto Weights = V->Parameters.find("weight");
      if (Weights != V->Parameters.end() && !Parameters.empty()) {
        if (Parameters.count(NewPrefix + V->Name + ".scales") == 0) {
          continue;
        }
      }
      if (Weights != V->Parameters.end() &&
          Weights->second.shape().back() % GroupSize != 0) {
        continue;
      }
    }
    V = V->toQuantized(GroupSize, Bits,
                       Prefix + Name + (Name.empty() ? "" : "."), Parameters);
  }
  return shared_from_this();
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
    if (LayerName == "layers" || LayerName == "blocks") {
      LayerName += "." + SplitKey[0];
      SplitKey.erase(SplitKey.begin());
    }
    if (Submodules.find(LayerName) == Submodules.end()) {
      spdlog::error("Unsupported Layer: {}", LayerName);
      assumingUnreachable();
    }
    Submodules.at(LayerName)->apply(joinString(SplitKey, '.'), Value);
  }
}
std::unordered_map<std::string, mx::array>
Module::getWeigts(const std::string &Prefix) {
  std::unordered_map<std::string, mx::array> Weights;
  auto NewPrefix = Prefix + Name;
  for (auto &[K, V] : Submodules) {
    auto Subweights = V->getWeigts(NewPrefix + (NewPrefix.empty() ? "" : "."));
    Weights.insert(Subweights.begin(), Subweights.end());
  }
  for (auto &[K, V] : Parameters) {
    Weights.insert({NewPrefix + (NewPrefix.empty() ? "" : ".") + K, V});
  }
  return Weights;
}

} // namespace mlx::core::nn

// 64-bit FNV-1a over the array's row-major bytes.
uint64_t fnv1aHash(const mx::array &X) {
  mx::array Contiguous = mx::contiguous(X);
  Contiguous.eval();
  const auto *Bytes = Contiguous.data<uint8_t>();
  uint64_t Hash = 14695981039346656037ULL;
  for (size_t I = 0; I < Contiguous.nbytes(); ++I) {
    Hash ^= Bytes[I];
    Hash *= 1099511628211ULL;
  }
  return Hash;
}

void debugArray(const mx::array &X, const std::string &Name) {
  std::cout << Name << " shape: (";
  for (auto &Shape : X.shape()) {
    std::cout << Shape << " ";
  }
  std::cout << ") ";
  std::cout << fnv1aHash(X) << std::endl;
}
