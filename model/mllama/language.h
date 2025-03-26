#pragma once
#include <mlx/array.h>
#include <optional>
#include <string>
#include <vector>

#include "../vlm_base.h"
#include "base.h"
#include "linear.h"
#include "normalization.h"
#include "positional_encoding.h"
#include <simdjson.h>

namespace nn = mlx::core::nn;
namespace mllama {

struct TextConfig {
  std::string ModelType = "mllama";
  int VocabSize = 32000;
  int HiddenSize = 4096;
  int IntermediateSize = 14336;
  int NumHiddenLayers = 40;
  int NumAttentionHeads = 32;
  int NumKeyValueHeads = 8;
  std::string HiddenAct = "silu";
  int MaxPositionEmbeddings = 131072;
  float InitializerRange = 0.02f;
  float RmsNormEps = 1e-6f;
  bool TieWordEmbeddings = false;
  float RopeTheta = 10000.0f;
  bool RopeTraditional = false;
  std::vector<int> CrossAttentionLayers = {3, 8, 13, 18, 23, 28, 33, 38};

  TextConfig() = default;
  static TextConfig fromDict(const simdjson::dom::object &Obj);
};

class MllamaTextCrossAttention : public nn::Module {
public:
  MllamaTextCrossAttention(const TextConfig &Config,
                           std::optional<int> LayerIdx = std::nullopt);
  virtual ~MllamaTextCrossAttention() = default;

  mx::array
  forward(const mx::array &HiddenStates,
          const std::optional<mx::array> &CrossAttentionStates = std::nullopt,
          const std::optional<mx::array> &AttentionMask = std::nullopt,
          vlm::BaseCache *Cache = nullptr);

private:
  TextConfig Config;
  int HiddenSize;
  int NumHeads;
  int HeadDim;
  int NumKeyValueHeads;
  int NumKeyValueGroups;
  std::optional<int> LayerIdx;
  float Scale;
};

class MllamaTextSelfAttention : public nn::Module {
public:
  MllamaTextSelfAttention(const TextConfig &Config, int LayerIdx);
  virtual ~MllamaTextSelfAttention() = default;

  mx::array forward(const mx::array &X,
                    const std::optional<mx::array> &Mask = std::nullopt,
                    vlm::BaseCache *Cache = nullptr);

private:
  TextConfig Config;
  int HiddenSize;
  int NumHeads;
  int HeadDim;
  int NumKeyValueHeads;
  int NumKeyValueGroups;
  float Scale;
  int LayerIdx;
};

class MllamaTextMLP : public nn::Module {
public:
  MllamaTextMLP(const TextConfig &Config);
  virtual ~MllamaTextMLP() = default;

  mx::array forward(const mx::array &X);
};

class MllamaSelfAttentionDecoderLayer : public nn::Module {
public:
  MllamaSelfAttentionDecoderLayer(const TextConfig &Config, int LayerIdx);
  virtual ~MllamaSelfAttentionDecoderLayer() = default;

  mx::array forward(const mx::array &HiddenStates,
                    const std::optional<mx::array> &Mask = std::nullopt,
                    vlm::BaseCache *Cache = nullptr);

private:
  int HiddenSize;
};

class MllamaCrossAttentionDecoderLayer : public nn::Module {
public:
  MllamaCrossAttentionDecoderLayer(const TextConfig &Config, int LayerIdx);
  virtual ~MllamaCrossAttentionDecoderLayer() = default;

  mx::array forward(
      const mx::array &HiddenStates, const mx::array &CrossAttentionStates,
      const std::optional<mx::array> &AttentionMask = std::nullopt,
      const std::optional<mx::array> &FullTextRowMaskedOutMask = std::nullopt,
      vlm::BaseCache *Cache = nullptr);

private:
  int HiddenSize;
  mx::array CrossAttnAttnGate;
  mx::array CrossAttnMlpGate;
};

class MllamaTextModel : public nn::Module {
public:
  MllamaTextModel(const TextConfig &Config);
  virtual ~MllamaTextModel() = default;

  mx::array forward(
      const std::optional<mx::array> &InputIds = std::nullopt,
      const std::optional<mx::array> &Mask = std::nullopt,
      const std::optional<mx::array> &PositionIds = std::nullopt,
      const std::optional<mx::array> &CrossAttentionStates = std::nullopt,
      const std::optional<mx::array> &CrossAttentionMask = std::nullopt,
      const std::optional<mx::array> &FullTextRowMaskedOutMask = std::nullopt,
      const std::optional<mx::array> &InputsEmbeds = std::nullopt,
      std::vector<vlm::BaseCache *> *Cache = nullptr);

private:
  TextConfig Config;
  int VocabSize;
  int HiddenSize;
  std::vector<std::shared_ptr<nn::Module>> Layers;

public:
  const std::vector<std::shared_ptr<nn::Module>> &getLayers() const {
    return Layers;
  }
};

struct LanguageModelOutput {
  mx::array Logits;
  std::optional<mx::array> CrossAttentionStates;
};

class LanguageModel : public nn::Module {
public:
  LanguageModel(const TextConfig &Config);
  virtual ~LanguageModel() = default;

  LanguageModelOutput forward(
      const std::optional<mx::array> &InputIds = std::nullopt,
      const std::optional<mx::array> &Mask = std::nullopt,
      const std::optional<mx::array> &CrossAttentionStates = std::nullopt,
      const std::optional<mx::array> &CrossAttentionMask = std::nullopt,
      const std::optional<mx::array> &FullTextRowMaskedOutMask = std::nullopt,
      const std::optional<mx::array> &InputsEmbeds = std::nullopt,
      std::vector<vlm::BaseCache *> *Cache = nullptr);

  static std::unordered_map<std::string, mx::array>
  sanitize(const std::unordered_map<std::string, mx::array> &Weights);

  // const std::vector<std::unique_ptr<nn::Module>> &layers() const;
  int headDim() const;
  int nKvHeads() const;

private:
  TextConfig Config;
};

} // namespace mllama
