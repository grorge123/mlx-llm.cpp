#include "converter.h"
#include "utils.h"
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

// StreamOrDevice Device = metal::is_available() ? Device::gpu : Device::cpu;

std::unordered_map<std::string, mx::array>
weightsToMlx(std::string WeightPath, mx::StreamOrDevice Device) {
  const std::filesystem::path Path(WeightPath);
  std::cout << "TEST:" << std::filesystem::is_directory(Path) << std::endl;
  if (std::filesystem::is_directory(Path)) {
    std::unordered_map<std::string, mx::array> Loaded;
    for (const auto &Entry : std::filesystem::directory_iterator(Path)) {
      if (Entry.path().extension() == ".safetensors") {
        auto SubWeight = weightsToMlx(Entry.path(), Device);
        Loaded.insert(SubWeight.begin(), SubWeight.end());
      }
    }
    return Loaded;
  }
  if (WeightPath.ends_with(".safetensors")) {
    std::cout << "Loading model from .safetensors file...\n";
    const mx::SafetensorsLoad Loaded = load_safetensors(WeightPath, Device);
    return Loaded.first;
  }
  if (WeightPath.ends_with(".gguf")) {
    std::cout << "Loading model from .gguf file...\n";
    const mx::GGUFLoad Loaded = load_gguf(WeightPath, Device);
    return Loaded.first;
  }
  std::cout << "Can not regonize model file\n";
  throw std::invalid_argument("Invalid model path.");
}

std::unordered_map<std::string, mx::array>
llamaToMlxllm(std::string WeightPath, mx::StreamOrDevice Device) {
  std::unordered_map<std::string, mx::array> ModelWeights;
  auto Weight = weightsToMlx(WeightPath, Device);
  for (auto &[k, v] : Weight) {
    std::string NewKey = k;
    if (NewKey.starts_with("model.")) {
      replace(NewKey, "model.", "");
    }
    std::vector<std::string> SplitKey = splitString(NewKey, '.');
    if (find(SplitKey.begin(), SplitKey.end(), "layers") != SplitKey.end()) {
      if (find(SplitKey.begin(), SplitKey.end(), "rotary_emb") !=
          SplitKey.end()) {
        continue;
      }
      if (find(SplitKey.begin(), SplitKey.end(), "self_attn") !=
          SplitKey.end()) {
        ModelWeights.insert({SplitKey[0] + "." + SplitKey[1] + ".attention." +
                                 SplitKey[3] + SplitKey[4],
                             v});
      } else if (find(SplitKey.begin(), SplitKey.end(), "mlp") !=
                 SplitKey.end()) {

        ModelWeights.insert({NewKey, v});
      } else {
        const std::unordered_map<std::string, std::string> KeyMap = {
            {"input_layernorm", "attention_norm"},
            {"post_attention_layernorm", "mlp_norm"}};
        ModelWeights.insert({SplitKey[0] + "." + SplitKey[1] + "." +
                                 KeyMap.at(SplitKey[2]) + "." + SplitKey[3],
                             v});
      }
    } else {
      const std::unordered_map<std::string, std::string> KeyMap = {
          {"embed_tokens", "token_embed"},
          {"lm_head", "head"},
          {"norm", "norm"}};
        ModelWeights.insert({KeyMap.at(SplitKey[0]) + "." + SplitKey[1], v});
    }
  }
  return ModelWeights;
}