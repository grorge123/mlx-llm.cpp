#pragma once

#include "../vlm_base.h"
#include "simdjson.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn = mlx::core::nn;
namespace gemma3 {

struct TextConfig {
  std::string ModelType;
  int HiddenSize;
  int NumHiddenLayers;
  int IntermediateSize;
  int NumAttentionHeads = 8;
  int HeadDim = 256;
  float RmsNormEps = 1.0e-6;
  int VocabSize = 262208;
  int NumKeyValueHeads = 4;
  float RopeGlobalBaseFreq = 1000000.0f;
  float RopeLocalBaseFreq = 10000.0f;
  bool RopeTraditional = false;
  float QueryPreAttnScalar = 0.0625;
  int SlidingWindow = 1024;
  std::optional<
      std::unordered_map<std::string, std::variant<float, std::vector<float>>>>
      RopeScaling;
  int MmTokensPerImage = 256;
  int SlidingWindowPattern = 6;
  static TextConfig fromDict(const simdjson::dom::object &Obj);
};

class RMSNorm : public nn::Module {
public:
  RMSNorm(int Dims, float Eps = 1e-5);
  mx::array forward(const mx::array &X);

private:
  mx::array Weight;
  float Eps;
};

class Attention : public nn::Module {
public:
  Attention(const TextConfig &Config, int LayerIdx);
  mx::array forward(const mx::array &X,
                    const std::optional<mx::array> &Mask = std::nullopt,
                    const std::optional<vlm::KVCache *> &Cache = std::nullopt);

private:
  int NHeads;
  int NKVHeads;
  int Repeats;
  int HeadDim;
  int LayerIdx;
  float Scale;
  RMSNorm QNorm;
  RMSNorm KNorm;
  bool IsSliding;
};

class MLP : public nn::Module {
public:
  MLP(int Dim, int HiddenDim);
  mx::array forward(const mx::array &X);
};

class TransformerBlock : public nn::Module {
public:
  TransformerBlock(const TextConfig &Config, int LayerIdx);
  mx::array forward(const mx::array &X,
                    const std::optional<mx::array> &Mask = std::nullopt,
                    const std::optional<vlm::KVCache *> &Cache = std::nullopt);

private:
  int NumAttentionHeads;
  int HiddenSize;
};

class Gemma3Model : public nn::Module {
public:
  Gemma3Model(const TextConfig &Config);
  mx::array forward(
      const mx::array &Inputs,
      const std::optional<mx::array> &InputsEmbeds = std::nullopt,
      const std::optional<mx::array> &Mask = std::nullopt,
      const std::optional<std::vector<vlm::KVCache *>> &Cache = std::nullopt);
  std::vector<std::shared_ptr<TransformerBlock>> Layers;
  TextConfig Config;
};

struct LanguageModelOutput {
  mx::array Logits;
};

class LanguageModel : public nn::Module {
public:
  LanguageModel(const TextConfig &Config);
  LanguageModelOutput forward(
      const mx::array &Inputs,
      const std::optional<mx::array> &InputsEmbeds = std::nullopt,
      const std::optional<mx::array> &Mask = std::nullopt,
      const std::optional<std::vector<vlm::KVCache *>> &Cache = std::nullopt);
  std::unordered_map<std::string, mx::array>
  sanitize(const std::unordered_map<std::string, mx::array> &Weights);
  int headDim() const;
  int nKvHeads() const;
  //   std::vector<void *> makeCache();
  TextConfig Config;
};
} // namespace gemma3