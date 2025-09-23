#include "load.h"
#include <filesystem>
#include <stdexcept>

std::pair<std::unordered_map<std::string, mx::array>,
          std::unordered_map<std::string, std::string>>
mlxLoadSafetensorHelper(const std::string &File, mx::StreamOrDevice S) {
  return mx::load_safetensors(File, S);
}

mx::GGUFLoad mlxLoadGgufHelper(const std::string &File, mx::StreamOrDevice S) {
  return mx::load_gguf(File, S);
}

std::unordered_map<std::string, mx::array>
mlxLoadNpzHelper(const std::string &File, mx::StreamOrDevice S) {
  throw std::invalid_argument(
      "[load_npz] NPZ format is not supported in pure C++ implementation. "
      "NPZ files require zip archive handling which is not available in the "
      "current implementation. "
      "Please convert to .safetensors or individual .npy files.");
}

mx::array mlxLoadNpyHelper(const std::string &File, mx::StreamOrDevice S) {
  return mx::load(File, S);
}

LoadOutputTypes mlxLoadHelper(const std::string &File,
                              std::optional<std::string> Format,
                              bool ReturnMetadata, mx::StreamOrDevice S) {

  if (!Format.has_value()) {
    std::filesystem::path Filepath(File);
    std::string Extension = Filepath.extension().string();

    if (Extension.empty()) {
      throw std::invalid_argument(
          "[load] Could not infer file format from extension");
    }

    if (Extension[0] == '.') {
      Extension = Extension.substr(1);
    }

    Format = Extension;
  }

  if (ReturnMetadata && (Format.value() == "npy" || Format.value() == "npz")) {
    throw std::invalid_argument("[load] metadata not supported for format " +
                                Format.value());
  }

  if (Format.value() == "safetensors") {
    auto [Dict, Metadata] = mlxLoadSafetensorHelper(File, S);
    if (ReturnMetadata) {
      return std::make_pair(Dict, Metadata);
    }
    return Dict;
  } else if (Format.value() == "npz") {
    return mlxLoadNpzHelper(File, S);
  } else if (Format.value() == "npy") {
    return mlxLoadNpyHelper(File, S);
  } else if (Format.value() == "gguf") {
    auto [Weights, Metadata] = mlxLoadGgufHelper(File, S);
    if (ReturnMetadata) {
      return std::make_pair(Weights, Metadata);
    } else {
      return Weights;
    }
  } else {
    throw std::invalid_argument("[load] Unknown file format " + Format.value());
  }
}
