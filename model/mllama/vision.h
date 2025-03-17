#pragma once

#include "../vlm_base.h"
#include "base.h"
#include "convolution.h"
#include "embedding.h"
#include "linear.h"
#include "normalization.h"
#include "positional_encoding.h"
#include "simdjson.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
namespace nn = mlx::core::nn;
namespace mllama {

struct VisionConfig {
  int ImageSize = 560;
  int PatchSize = 14;
  int NumChannels = 3;
  int HiddenSize = 1280;
  int IntermediateSize = 5120;
  int NumHiddenLayers = 32;
  int NumAttentionHeads = 16;
  int MaxNumTiles = 4;
  int MaxAspectRatioId = 8;
  int NumGlobalLayers = 8;
  float NormEps = 1e-5f;
  float AttentionDropout = 0.0f;
  float HiddenDropout = 0.0f;
  int VisionOutputDim = 7680;
  std::vector<int> IntermediateLayersIndices = {3, 7, 15, 23, 30};
  std::vector<std::vector<int>> SupportedAspectRatios = {
      {1, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 1}, {2, 2}, {3, 1}, {4, 1}};

  VisionConfig() = default;
  static VisionConfig fromDict(const simdjson::dom::object &Obj);
};

bool checkArrayShape(const mx::array &Arr);

class MllamaVisionAttention : public nn::Module {
public:
  explicit MllamaVisionAttention(const VisionConfig &Config);
  mx::array
  forward(const mx::array &HiddenState,
          const std::optional<mx::array> &AttentionMask = std::nullopt);

private:
  int EmbedDim;
  int NumHeads;
  int HeadDim;
  float Scale;
};

class MllamaVisionMLP : public nn::Module {
public:
  explicit MllamaVisionMLP(const VisionConfig &Config);
  mx::array forward(const mx::array &HiddenStates);
};

class MllamaVisionEncoderLayer : public nn::Module {
public:
  MllamaVisionEncoderLayer(const VisionConfig &Config, bool IsGated = false);
  mx::array
  forward(const mx::array &HiddenState,
          const std::optional<mx::array> &AttentionMask = std::nullopt);

private:
  int HiddenSize;
  int NumAttentionHeads;
  bool IsGated;
  mx::array GateAttn = mx::array({});
  ;
  mx::array GateFfn = mx::array({});
  ;
};

class MllamaVisionEncoder : public nn::Module {
public:
  MllamaVisionEncoder(const VisionConfig &Config, int NumLayers = 32,
                      bool IsGated = false);
  std::pair<mx::array, std::vector<mx::array>>
  forward(const mx::array &HiddenStates,
          const std::optional<mx::array> &AttentionMask = std::nullopt);

private:
  std::vector<std::shared_ptr<MllamaVisionEncoderLayer>> Layers;
};

class MllamaPrecomputedAspectRatioEmbedding : public nn::Module {
public:
  MllamaPrecomputedAspectRatioEmbedding(const VisionConfig &Config,
                                        bool IsGated = true);
  mx::array forward(const mx::array &HiddenState,
                    const mx::array &AspectRatioIds);

private:
  int MaxNumTiles;
  int HiddenSize;
  int MaxAspectRatioId;
  bool IsGated;
  mx::array Gate = mx::array({});
};

class MllamaPrecomputedPositionEmbedding : public nn::Module {
public:
  explicit MllamaPrecomputedPositionEmbedding(const VisionConfig &Config);
  mx::array forward(const mx::array &HiddenState,
                    const mx::array &AspectRatioIds);

private:
  int MaxNumTiles;
  int MaxAspectRatioId;
  int NumPatches;
  int HiddenSize;
  float Scale;
  mx::array Gate = mx::array({});
  mx::array Embedding = mx::array({});
};

class VisionModel : public nn::Module {
public:
  explicit VisionModel(const VisionConfig &Config);
  mx::array forward(const mx::array &PixelValues,
                    const mx::array &AspectRatioIds,
                    const mx::array &AspectRatioMask);

  static std::unordered_map<std::string, mx::array>
  sanitize(const std::unordered_map<std::string, mx::array> &Weights);

private:
  int ImageSize;
  int PatchSize;
  int MaxNumTiles;
  int HiddenSize;
  int NumChannels;
  std::vector<int> IntermediateLayersIndices;
  int NumPatches;
  float Scale;
  mx::array ClassEmbedding = mx::array({});
};

mx::array _prepareAspectRatioAttentionMask(const mx::array &AspectRatioMask,
                                           int NumPatches, int TargetLength);
} // namespace mllama
