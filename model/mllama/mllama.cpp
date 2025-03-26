#include "mllama.h"
#include "base.h"
#include "language.h"
#include "linear.h"
#include "vision.h"
#include <filesystem>
#include <memory>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
namespace mllama {

ModelConfig ModelConfig::fromDict(const simdjson::dom::object &Obj) {
  ModelConfig Config;
  auto ModelTypeResult = Obj["model_type"].get_string();
  if (!ModelTypeResult.error()) {
    Config.ModelType = std::string(ModelTypeResult.value());
  }
  auto IgnoreResult = Obj["ignore_index"].get_int64();
  if (!IgnoreResult.error()) {
    Config.IgnoreIndex = static_cast<int>(IgnoreResult.value());
  }
  auto ImgTokenResult = Obj["image_token_index"].get_int64();
  if (!ImgTokenResult.error()) {
    Config.ImageTokenIndex = static_cast<int>(ImgTokenResult.value());
  }
  auto VfsResult = Obj["vision_feature_select_strategy"].get_string();
  if (!VfsResult.error()) {
    Config.VisionFeatureSelectStrategy = std::string(VfsResult.value());
  }
  auto VflResult = Obj["vision_feature_layer"].get_int64();
  if (!VflResult.error()) {
    Config.VisionFeatureLayer = static_cast<int>(VflResult.value());
  }
  auto VocabResult = Obj["vocab_size"].get_int64();
  if (!VocabResult.error()) {
    Config.VocabSize = static_cast<int>(VocabResult.value());
  }
  Config.TextConfig = TextConfig::fromDict(Obj);
  auto VisionObjResult = Obj["vision_config"].get_object();
  if (!VisionObjResult.error()) {
    Config.VisionConfig = VisionConfig::fromDict(VisionObjResult.value());
  }
  return Config;
}

Model::Model(const ModelConfig &Config) : Config(Config) {
  registerModule("vision_tower",
                 std::make_shared<VisionModel>(Config.VisionConfig));
  registerModule("language_model",
                 std::make_shared<LanguageModel>(Config.TextConfig));
  registerModule(
      "multi_modal_projector",
      std::make_shared<nn::Linear>(Config.VisionConfig.VisionOutputDim,
                                   Config.TextConfig.HiddenSize, true));
}

std::tuple<mx::array, std::optional<mx::array>>
Model::forward(const mx::array &InputIds, const mx::array &PixelValues,
               const mx::array &Mask, std::vector<vlm::BaseCache *> *Cache,
               const std::optional<mx::array> &AspectRatioIds,
               const std::optional<mx::array> &AspectRatioMask,
               const std::optional<mx::array> &CrossAttentionMask) {

  mx::array CrossAttentionStates = mx::array({});
  if (PixelValues.size() != 0) {
    if (!AspectRatioIds.has_value())
      throw std::invalid_argument(
          "`aspect_ratio_ids` must be provided if `pixel_values` is provided");
    auto VisionOutputs =
        std::dynamic_pointer_cast<VisionModel>(Submodules["vision_tower"])
            ->forward(PixelValues, AspectRatioIds.value(),
                      AspectRatioMask.value());
    CrossAttentionStates = take(VisionOutputs, 0);
    CrossAttentionStates = std::dynamic_pointer_cast<nn::Linear>(
                               Submodules["multi_modal_projector"])
                               ->forward(CrossAttentionStates);
    CrossAttentionStates = reshape(
        CrossAttentionStates,
        {-1,
         CrossAttentionStates.shape()[CrossAttentionStates.shape().size() - 2],
         Config.TextConfig.HiddenSize});
  } else {
    CrossAttentionStates = mx::array({});
  }
  mx::array FullTextRowMaskedOutMask = mx::array({});
  mx::array CrossAttnMask = mx::array({});
  if (CrossAttentionMask.has_value()) {
    int NumVisionTokens =
        (Config.VisionConfig.ImageSize / Config.VisionConfig.PatchSize) *
            (Config.VisionConfig.ImageSize / Config.VisionConfig.PatchSize) +
        1;
    auto Masks =
        prepareCrossAttentionMask(CrossAttentionMask.value(), NumVisionTokens);
    CrossAttnMask = Masks.first;
    FullTextRowMaskedOutMask = Masks.second;
  }
  if (CrossAttnMask.size() != 0) {
    mx::array CachePosition = mx::arange(InputIds.shape()[1], mx::int32);
    // cross_attention_mask = cross_attention_mask[:, :, cache_position]
    CrossAttnMask = take(CrossAttnMask, {CachePosition}, 2);
    // full_text_row_masked_out_mask = full_text_row_masked_out_mask[
    // 	:, :, cache_position
    // ]
    FullTextRowMaskedOutMask =
        take(FullTextRowMaskedOutMask, {CachePosition}, 2);
  }
  auto Outputs =
      std::dynamic_pointer_cast<LanguageModel>(Submodules["language_model"])
          ->forward(InputIds, Mask, CrossAttentionStates, CrossAttnMask,
                    FullTextRowMaskedOutMask, mx::array({}), Cache);
  return {Outputs.Logits, Outputs.CrossAttentionStates};
}

std::pair<mx::array, mx::array>
Model::prepareCrossAttentionMask(const mx::array &CrossAttentionMask,
                                 int NumVisionTokens) {
  auto Shape = CrossAttentionMask.shape();
  int BatchSize = Shape[0];
  int TextTotalLength = Shape[1];
  mx::array CrossAttnMask = mx::repeat(CrossAttentionMask, NumVisionTokens, 3);
  CrossAttnMask = reshape(CrossAttnMask, {BatchSize, TextTotalLength, -1});
  CrossAttnMask = mlx::core::expand_dims(CrossAttnMask, 1);
  mx::array InvertedMask = 1.0 - CrossAttnMask;
  mx::array FillArray = mx::array(-1e9);
  FillArray = mlx::core::broadcast_to(FillArray, InvertedMask.shape());
  CrossAttnMask = mx::where(InvertedMask, FillArray, CrossAttnMask);
  mx::array FullTextRowMaskedOutMask = mx::any(CrossAttnMask != -1e9, -1, true);
  CrossAttnMask = CrossAttnMask * FullTextRowMaskedOutMask;
  return {CrossAttnMask, FullTextRowMaskedOutMask};
}

std::shared_ptr<Model> Model::fromPretrained(const std::string &ModelPath) {
  fs::path Path(ModelPath);
  simdjson::dom::parser Parser;
  simdjson::dom::element Doc;
  auto Error = Parser.load((Path / "config.json").string()).get(Doc);
  if (Error) {
    spdlog::error("Could not open config.json");
    assumingUnreachable();
  }
  auto Obj = Doc.get_object();
  ModelConfig ModelConfig = ModelConfig::fromDict(Obj.value());
  ModelConfig.VisionConfig =
      VisionConfig::fromDict(Obj["vision_config"].get_object().value());
  ModelConfig.TextConfig = TextConfig::fromDict(Obj.value());
  auto Model = std::make_shared<mllama::Model>(mllama::Model(ModelConfig));
  auto QuantResult = Obj["quantization"].get_object();
  if (!QuantResult.error()) {
    auto GroupSize = static_cast<int>(QuantResult.value()["group_size"]);
    auto Bits = static_cast<int>(QuantResult.value()["bits"]);
    Model = std::dynamic_pointer_cast<mllama::Model>(
        Model->toQuantized(GroupSize, Bits));
  }
  std::vector<fs::path> WeightFiles;
  for (auto &P : fs::directory_iterator(Path)) {
    if (P.path().extension() == ".safetensors")
      WeightFiles.push_back(P.path());
  }
  if (WeightFiles.empty())
    throw std::runtime_error("No safetensors found in " + Path.string());
  std::unordered_map<std::string, mx::array> Weights;
  for (auto &Wf : WeightFiles) {
    auto W = mx::load_safetensors(Wf.string());
    Weights.insert(W.first.begin(), W.first.end());
  }
  Weights = VisionModel::sanitize(Weights);
  Weights = LanguageModel::sanitize(Weights);
  Model->update(Weights);
  return Model;
}
} // namespace mllama
