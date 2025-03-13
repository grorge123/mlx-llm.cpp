#pragma once

#include "base.h"
#include "language.h"
#include "mlllama_base.h"
#include "vision.h"
#include "simdjson.h"

namespace nn = mlx::core::nn;

struct ModelConfig {
  TextConfig TextConfig;
  VisionConfig VisionConfig;
  std::string ModelType;
  int IgnoreIndex = -100;
  int ImageTokenIndex = 128256;
  std::string VisionFeatureSelectStrategy = "default";
  int VisionFeatureLayer = -2;
  int VocabSize = 32000;
  static ModelConfig fromDict(const simdjson::dom::object &Obj);
};

class Model : public nn::Module {
public:
  explicit Model(const ModelConfig &Config);
  std::tuple<mx::array, std::optional<mx::array>>
  forward(const mx::array &InputIds, const mx::array &PixelValues,
          const mx::array &Mask, std::vector<vlm::KVCache *> *Cache = nullptr,
          const std::optional<mx::array> &AspectRatioIds = std::nullopt,
          const std::optional<mx::array> &AspectRatioMask = std::nullopt,
          const std::optional<mx::array> &CrossAttentionMask = std::nullopt);
  static Model fromPretrained(const std::string &PathOrHfRepo);

protected:
  std::pair<mx::array, mx::array>
  prepareCrossAttentionMask(const mx::array &CrossAttentionMask,
                            int NumVisionTokens);

public:
  ModelConfig Config;
};
