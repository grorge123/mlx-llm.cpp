#include "base.h"
#include "../model/utils.h"
#include "spdlog/spdlog.h"
#include <memory>
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

// mx::array asContiguousArray(const mx::array &X) {
//   if (X.buffer_size() == X.nbytes()) {
//     return X;
//   }

//   mx::array contiguous = mx::empty(X.shape(), X.dtype());
//   // 透過 assign 或 copy 方法將 X 的數值複製進新 array 中，保證按照 row-major
//   的順序複製 contiguous.assign(X); return contiguous;
// }

uint64_t fnv1aHash(const mx::array &X) {
  std::string fileName = "./temp_array.npy";
  mx::save(fileName.c_str(), X);
  std::string Command = "python3.10 ../hash_script.py " + fileName;

  // 透過 popen 呼叫外部程序讀取 Python 腳本的輸出
  FILE *pipe = popen(Command.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to open pipe for Python script.");
  }

  // 讀取 pipe 中的輸出（hash 值）
  char buffer[128];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  // 將結果字串轉換成 uint64_t
  try {
    uint64_t hashVal = std::stoull(result);
    return hashVal;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to parse hash value from Python output.");
  }
}

void debugArray(const mx::array &X, const std::string &Name) {
  std::cout << Name << " shape: (";
  for (auto &Shape : X.shape()) {
    std::cout << Shape << " ";
  }
  std::cout << ") ";
  std::cout  << fnv1aHash(X) << std::endl;
}