#include "vision.h"
#include "../../mlx/mlx_compat.h"
#include "activations.h"
#include <cmath>
#include <mlx/array.h>
#include <mlx/ops.h>

namespace mllama {

VisionConfig VisionConfig::fromDict(const simdjson::dom::object &Obj) {
  VisionConfig Config;
  auto Result = Obj["image_size"].get_int64();
  if (!Result.error()) {
    Config.ImageSize = static_cast<int>(Result.value());
  }
  Result = Obj["patch_size"].get_int64();
  if (!Result.error()) {
    Config.PatchSize = static_cast<int>(Result.value());
  }
  Result = Obj["num_channels"].get_int64();
  if (!Result.error()) {
    Config.NumChannels = static_cast<int>(Result.value());
  }
  Result = Obj["hidden_size"].get_int64();
  if (!Result.error()) {
    Config.HiddenSize = static_cast<int>(Result.value());
  }
  Result = Obj["intermediate_size"].get_int64();
  if (!Result.error()) {
    Config.IntermediateSize = static_cast<int>(Result.value());
  }
  Result = Obj["num_hidden_layers"].get_int64();
  if (!Result.error()) {
    Config.NumHiddenLayers = static_cast<int>(Result.value());
  }
  Result = Obj["num_attention_heads"].get_int64();
  if (!Result.error()) {
    Config.NumAttentionHeads = static_cast<int>(Result.value());
  }
  Result = Obj["max_num_tiles"].get_int64();
  if (!Result.error()) {
    Config.MaxNumTiles = static_cast<int>(Result.value());
  }
  Result = Obj["max_aspect_ratio_id"].get_int64();
  if (!Result.error()) {
    Config.MaxAspectRatioId = static_cast<int>(Result.value());
  }
  Result = Obj["num_global_layers"].get_int64();
  if (!Result.error()) {
    Config.NumGlobalLayers = static_cast<int>(Result.value());
  }
  auto DResult = Obj["norm_eps"].get_double();
  if (!DResult.error()) {
    Config.NormEps = static_cast<float>(DResult.value());
  }
  DResult = Obj["attention_dropout"].get_double();
  if (!DResult.error()) {
    Config.AttentionDropout = static_cast<float>(DResult.value());
  }
  DResult = Obj["hidden_dropout"].get_double();
  if (!DResult.error()) {
    Config.HiddenDropout = static_cast<float>(DResult.value());
  }
  Result = Obj["vision_output_dim"].get_int64();
  if (!Result.error()) {
    Config.VisionOutputDim = static_cast<int>(Result.value());
  }
  simdjson::dom::array Indices;
  if (Obj["intermediate_layers_indices"].get_array().get(Indices) ==
      simdjson::SUCCESS) {
    Config.IntermediateLayersIndices.clear();
    for (auto Element : Indices) {
      auto IResult = Element.get_int64();
      if (!IResult.error()) {
        Config.IntermediateLayersIndices.push_back(
            static_cast<int>(IResult.value()));
      }
    }
  }
  simdjson::dom::array Ratios;
  if (Obj["supported_aspect_ratios"].get_array().get(Ratios) ==
      simdjson::SUCCESS) {
    Config.SupportedAspectRatios.clear();
    for (auto Sub : Ratios) {
      simdjson::dom::array Pair;
      if (Sub.get_array().get(Pair) == simdjson::SUCCESS) {
        std::vector<int> Pr;
        for (auto P : Pair) {
          auto IResult = P.get_int64();
          if (!IResult.error()) {
            Pr.push_back(static_cast<int>(IResult.value()));
          }
        }
        if (Pr.size() == 2) {
          Config.SupportedAspectRatios.push_back(Pr);
        }
      }
    }
  }
  return Config;
}

bool checkArrayShape(const mx::array &Arr) {
  auto Shape = Arr.shape();
  if (Shape.size() != 4)
    return false;
  int OutChannels = Shape[0];
  int KH = Shape[1];
  int KW = Shape[2];
  return (OutChannels >= KH) && (OutChannels >= KW) && (KH == KW);
}

MllamaVisionAttention::MllamaVisionAttention(const VisionConfig &Config)
    : EmbedDim(Config.HiddenSize), NumHeads(Config.NumAttentionHeads),
      HeadDim(Config.HiddenSize / Config.NumAttentionHeads),
      Scale(1.0f / std::sqrt(static_cast<float>(HeadDim))) {
  registerModule("q_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NumHeads * HeadDim, false));
  registerModule("k_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NumHeads * HeadDim, false));
  registerModule("v_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NumHeads * HeadDim, false));
  registerModule("o_proj", std::make_shared<nn::Linear>(
                               NumHeads * HeadDim, Config.HiddenSize, false));
}

mx::array
MllamaVisionAttention::forward(const mx::array &HiddenState,
                               const std::optional<mx::array> &AttentionMask) {
  mx::array Query = std::dynamic_pointer_cast<nn::Linear>(Submodules["q_proj"])
                        ->forward(HiddenState);
  mx::array Key = std::dynamic_pointer_cast<nn::Linear>(Submodules["k_proj"])
                      ->forward(HiddenState);
  mx::array Value = std::dynamic_pointer_cast<nn::Linear>(Submodules["v_proj"])
                        ->forward(HiddenState);
  auto QShape = Query.shape();
  int BatchSize = QShape[0];
  int QSeqLen = QShape[1];
  auto KShape = Key.shape();
  int KVSeqLen = KShape[1];
  Query = transpose(reshape(Query, {BatchSize, QSeqLen, NumHeads, HeadDim}),
                    {0, 2, 1, 3});
  Key = transpose(reshape(Key, {BatchSize, KVSeqLen, NumHeads, HeadDim}),
                  {0, 2, 1, 3});
  Value = transpose(reshape(Value, {BatchSize, KVSeqLen, NumHeads, HeadDim}),
                    {0, 2, 1, 3});
  std::optional<mx::array> MaskOpt = AttentionMask;
  if (MaskOpt.has_value()) {
    // attention_mask = attention_mask[:, :, : key.shape[-2], :]
    mx::array Indices = mx::arange(Key.shape()[Key.size() - 3]);
    MaskOpt = take(MaskOpt.value(), Indices, -2);
  }
  mx::array AttnOutput = mlx_compat::scaled_dot_product_attention(
      Query, Key, Value, Scale, MaskOpt);
  AttnOutput = reshape(transpose(AttnOutput, {0, 2, 1, 3}),
                       {BatchSize, QSeqLen, EmbedDim});
  return std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
      ->forward(AttnOutput);
}

MllamaVisionMLP::MllamaVisionMLP(const VisionConfig &Config) {
  registerModule("fc1", std::make_shared<nn::Linear>(
                            Config.HiddenSize, Config.IntermediateSize, true));
  registerModule("fc2", std::make_shared<nn::Linear>(Config.IntermediateSize,
                                                     Config.HiddenSize, true));
}

mx::array MllamaVisionMLP::forward(const mx::array &HiddenStates) {
  mx::array X = std::dynamic_pointer_cast<nn::Linear>(Submodules["fc1"])
                    ->forward(HiddenStates);
  X = mlx::core::gelu(X);
  X = std::dynamic_pointer_cast<nn::Linear>(Submodules["fc2"])->forward(X);
  return X;
}

MllamaVisionEncoderLayer::MllamaVisionEncoderLayer(const VisionConfig &Config,
                                                   bool IsGated)
    : HiddenSize(Config.HiddenSize),
      NumAttentionHeads(Config.NumAttentionHeads), IsGated(IsGated) {
  registerModule("self_attn", std::make_shared<MllamaVisionAttention>(Config));
  registerModule("mlp", std::make_shared<MllamaVisionMLP>(Config));
  registerModule("input_layernorm", std::make_shared<nn::LayerNorm>(
                                        Config.HiddenSize, Config.NormEps));
  registerModule(
      "post_attention_layernorm",
      std::make_shared<nn::LayerNorm>(Config.HiddenSize, Config.NormEps));
  if (IsGated) {
    GateAttn = mx::zeros({1});
    GateFfn = mx::zeros({1});
  }
}

mx::array MllamaVisionEncoderLayer::forward(
    const mx::array &HiddenState,
    const std::optional<mx::array> &AttentionMask) {
  mx::array Residual = HiddenState;
  mx::array X =
      std::dynamic_pointer_cast<nn::LayerNorm>(Submodules["input_layernorm"])
          ->forward(HiddenState);
  X = std::dynamic_pointer_cast<MllamaVisionAttention>(Submodules["self_attn"])
          ->forward(X, AttentionMask);
  if (IsGated) {
    X = mx::tanh(GateAttn) * X;
  }
  X = Residual + X;
  Residual = X;
  X = std::dynamic_pointer_cast<nn::LayerNorm>(
          Submodules["post_attention_layernorm"])
          ->forward(X);
  X = std::dynamic_pointer_cast<MllamaVisionMLP>(Submodules["mlp"])->forward(X);
  if (IsGated) {
    X = mx::tanh(GateFfn) * X;
  }
  return Residual + X;
}

MllamaVisionEncoder::MllamaVisionEncoder(const VisionConfig &Config,
                                         int NumLayers, bool IsGated) {
  for (int Index = 0; Index < NumLayers; ++Index) {
    auto Layer = std::make_shared<MllamaVisionEncoderLayer>(Config, IsGated);
    registerModule("encoder_layer_" + std::to_string(Index), Layer);
    Layers.push_back(Layer);
  }
}

std::pair<mx::array, std::vector<mx::array>>
MllamaVisionEncoder::forward(const mx::array &HiddenStates,
                             const std::optional<mx::array> &AttentionMask) {
  mx::array X = HiddenStates;
  std::vector<mx::array> EncoderStates;
  for (auto &Layer : Layers) {
    X = Layer->forward(X, AttentionMask);
    EncoderStates.push_back(X);
  }
  return {X, EncoderStates};
}

MllamaPrecomputedAspectRatioEmbedding::MllamaPrecomputedAspectRatioEmbedding(
    const VisionConfig &Config, bool IsGated)
    : MaxNumTiles(Config.MaxNumTiles), HiddenSize(Config.HiddenSize),
      MaxAspectRatioId(Config.MaxAspectRatioId), IsGated(IsGated) {
  registerModule("embedding",
                 std::make_shared<nn::Embedding>(MaxAspectRatioId + 1,
                                                 MaxNumTiles * HiddenSize));
  if (IsGated) {
    Gate = mx::zeros({1});
  }
}

mx::array MllamaPrecomputedAspectRatioEmbedding::forward(
    const mx::array &HiddenState, const mx::array &AspectRatioIds) {
  mx::array Embeddings =
      std::dynamic_pointer_cast<nn::Embedding>(Submodules["embedding"])
          ->forward(AspectRatioIds);
  Embeddings = reshape(Embeddings, {-1, MaxNumTiles, 1, HiddenSize});
  if (IsGated) {
    Embeddings = Embeddings * mx::tanh(Gate);
  }
  return HiddenState + Embeddings;
}

MllamaPrecomputedPositionEmbedding::MllamaPrecomputedPositionEmbedding(
    const VisionConfig &Config)
    : MaxNumTiles(Config.MaxNumTiles),
      MaxAspectRatioId(Config.MaxAspectRatioId), HiddenSize(Config.HiddenSize),
      Scale(1.0f / std::sqrt(static_cast<float>(Config.HiddenSize))) {
  int Patches = (Config.ImageSize / Config.PatchSize) *
                    (Config.ImageSize / Config.PatchSize) +
                1;
  NumPatches = Patches;
  Gate = mx::zeros({1});
  Embedding = mx::random::normal({NumPatches, HiddenSize}) * Scale;
  registerModule("tile_embedding", std::make_shared<nn::Embedding>(
                                       MaxAspectRatioId + 1,
                                       MaxNumTiles * NumPatches * HiddenSize));
}

mx::array
MllamaPrecomputedPositionEmbedding::forward(const mx::array &HiddenState,
                                            const mx::array &AspectRatioIds) {
  mx::array GatedPositionEmbedding = (1 - mx::tanh(Gate)) * Embedding;
  mx::array PosEmbedReshaped =
      reshape(GatedPositionEmbedding, {1, 1, NumPatches, HiddenSize});
  mx::array X = HiddenState + PosEmbedReshaped;
  mx::array TilePositionEmbedding =
      std::dynamic_pointer_cast<nn::Embedding>(Submodules["tile_embedding"])
          ->forward(AspectRatioIds);
  int BatchSize = HiddenState.shape()[0];
  TilePositionEmbedding = reshape(
      TilePositionEmbedding, {BatchSize, MaxNumTiles, NumPatches, HiddenSize});
  mx::array GatedTilePositionEmbedding = mx::tanh(Gate) * TilePositionEmbedding;
  return X + GatedTilePositionEmbedding;
}

mx::array _prepareAspectRatioAttentionMask(const mx::array &AspectRatioMask,
                                           int NumPatches, int TargetLength) {

  mx::array Mask = mlx::core::astype(AspectRatioMask, mx::float32);
  auto MaskShape = Mask.shape();
  int BatchSize = MaskShape[0];
  int MaxNumTiles = MaskShape[1];
  Mask = reshape(Mask, {BatchSize, MaxNumTiles, 1, 1});
  Mask = mx::tile(Mask, {1, 1, TargetLength, 1});
  int PadPatches = TargetLength - NumPatches;
  // attention_mask[:, :, -pad_patches:] = 0
  auto End = Mask.shape();
  MlxShape Start(End.size(), 0);
  MlxShape Stride(End.size(), 1);
  Start[2] = Mask.size() - PadPatches;
  Mask = mlx::core::slice_update(
      Mask, mx::zeros({Mask.shape()[0], Mask.shape()[1], PadPatches}), Start,
      End, Stride);

  Mask = 1 - Mask;
  Mask = reshape(Mask, {BatchSize, MaxNumTiles * TargetLength, 1});
  float MinValue = -1e9f;
  mx::array MaskT = transpose(Mask, {0, 2, 1});
  mx::array AttnMask = mlx::core::matmul(Mask, MaskT) * MinValue;
  auto ReshapeDim = AttnMask.shape();
  ReshapeDim.insert(ReshapeDim.begin() + 1, 1);
  // attention_mask = attention_mask[:, None, :, :]
  AttnMask = reshape(AttnMask, ReshapeDim);
  return AttnMask;
}

VisionModel::VisionModel(const VisionConfig &Config)
    : ImageSize(Config.ImageSize), PatchSize(Config.PatchSize),
      MaxNumTiles(Config.MaxNumTiles), HiddenSize(Config.HiddenSize),
      NumChannels(Config.NumChannels),
      IntermediateLayersIndices(Config.IntermediateLayersIndices),
      Scale(1.0f / std::sqrt(static_cast<float>(Config.HiddenSize))) {
  NumPatches = (ImageSize / PatchSize) * (ImageSize / PatchSize) + 1;
  registerModule(
      "patch_embedding",
      std::make_shared<nn::Conv2d>(nn::Conv2d(
          Config.NumChannels, HiddenSize, Config.PatchSize,
          {Config.PatchSize, Config.PatchSize}, {0, 0}, {1, 1}, 1, false)));
  ClassEmbedding = mx::random::normal({HiddenSize}) * Scale;
  registerModule("gated_positional_embedding",
                 std::make_shared<MllamaPrecomputedPositionEmbedding>(Config));
  registerModule(
      "pre_tile_positional_embedding",
      std::make_shared<MllamaPrecomputedAspectRatioEmbedding>(Config, true));
  registerModule(
      "post_tile_positional_embedding",
      std::make_shared<MllamaPrecomputedAspectRatioEmbedding>(Config, true));
  registerModule("layernorm_pre",
                 std::make_shared<nn::LayerNorm>(HiddenSize, Config.NormEps));
  registerModule("layernorm_post",
                 std::make_shared<nn::LayerNorm>(HiddenSize, Config.NormEps));
  registerModule("transformer", std::make_shared<MllamaVisionEncoder>(
                                    Config, Config.NumHiddenLayers, false));
  registerModule("global_transformer",
                 std::make_shared<MllamaVisionEncoder>(
                     Config, Config.NumGlobalLayers, true));
}

mx::array VisionModel::forward(const mx::array &PixelValues,
                               const mx::array &AspectRatioIds,
                               const mx::array &AspectRatioMask) {
  auto Shape = PixelValues.shape();
  int BatchSize = Shape[0];
  int NumConcurrentMedia = Shape[1];
  int NumTiles = Shape[2];
  int Channels = Shape[3];
  int Height = Shape[4];
  int Width = Shape[5];
  mx::array AspectRatioIdsReshaped =
      reshape(AspectRatioIds, {BatchSize * NumConcurrentMedia, -1});
  mx::array PixelValuesReshaped =
      reshape(PixelValues, {BatchSize * NumConcurrentMedia * NumTiles, Channels,
                            Height, Width});
  mx::array Temp = moveaxis(PixelValuesReshaped, 1, 3);
  mx::array PatchEmbeds = moveaxis(
      std::dynamic_pointer_cast<nn::Conv2d>(Submodules["patch_embedding"])
          ->forward(Temp),
      3, 1);
  auto PatchShape = PatchEmbeds.shape();
  int Dim = PatchShape[1];
  mx::array HiddenState = transpose(
      reshape(PatchEmbeds, {PatchShape[0], PatchShape[1], -1}), {0, 2, 1});
  HiddenState =
      reshape(HiddenState, {BatchSize * NumConcurrentMedia, NumTiles, -1, Dim});
  HiddenState =
      std::dynamic_pointer_cast<MllamaPrecomputedAspectRatioEmbedding>(
          Submodules["pre_tile_positional_embedding"])
          ->forward(HiddenState, AspectRatioIdsReshaped);
  HiddenState = reshape(HiddenState,
                        {BatchSize * NumConcurrentMedia * NumTiles, -1, Dim});
  mx::array ClassEmbeddingBroadcast = mx::broadcast_to(
      ClassEmbedding, {BatchSize * NumConcurrentMedia * NumTiles, 1, Dim});
  HiddenState = mx::concatenate({ClassEmbeddingBroadcast, HiddenState}, 1);
  int NumPatchesLocal = HiddenState.shape()[1];
  HiddenState = reshape(HiddenState, {BatchSize * NumConcurrentMedia, NumTiles,
                                      NumPatchesLocal, Dim});
  HiddenState = std::dynamic_pointer_cast<MllamaPrecomputedPositionEmbedding>(
                    Submodules["gated_positional_embedding"])
                    ->forward(HiddenState, AspectRatioIdsReshaped);
  HiddenState =
      std::dynamic_pointer_cast<nn::LayerNorm>(Submodules["layernorm_pre"])
          ->forward(HiddenState);
  int CurrentNumPatches = HiddenState.shape()[2];
  int NumPaddingPatches = (8 - (CurrentNumPatches % 8)) % 8;
  std::vector<std::pair<int, int>> Padding = {
      {0, 0}, {0, 0}, {0, NumPaddingPatches}, {0, 0}};
  HiddenState = mx::pad(HiddenState, Padding);
  int SliceIndex =
      (NumPaddingPatches > 0) ? -NumPaddingPatches : HiddenState.shape()[2];
  mx::array AttentionMask =
      reshape(AspectRatioMask, {BatchSize * NumConcurrentMedia, -1});
  AttentionMask = _prepareAspectRatioAttentionMask(AttentionMask, NumPatches,
                                                   HiddenState.shape()[2]);
  HiddenState =
      reshape(HiddenState, {BatchSize * NumConcurrentMedia, -1, HiddenSize});
  auto EncoderOutput =
      std::dynamic_pointer_cast<MllamaVisionEncoder>(Submodules["transformer"])
          ->forward(HiddenState, AttentionMask);
  mx::array EncodedHidden = EncoderOutput.first;
  EncodedHidden =
      std::dynamic_pointer_cast<nn::LayerNorm>(Submodules["layernorm_post"])
          ->forward(EncodedHidden);
  EncodedHidden =
      reshape(EncodedHidden, {BatchSize * NumConcurrentMedia, NumTiles,
                              NumPatchesLocal + NumPaddingPatches, HiddenSize});
  EncodedHidden =
      std::dynamic_pointer_cast<MllamaPrecomputedAspectRatioEmbedding>(
          Submodules["post_tile_positional_embedding"])
          ->forward(EncodedHidden, AspectRatioIdsReshaped);
  EncodedHidden =
      reshape(EncodedHidden, {BatchSize * NumConcurrentMedia, -1, HiddenSize});
  auto GlobalOutput = std::dynamic_pointer_cast<MllamaVisionEncoder>(
                          Submodules["global_transformer"])
                          ->forward(EncodedHidden, AttentionMask);
  mx::array GlobalHidden = GlobalOutput.first;
  GlobalHidden =
      reshape(GlobalHidden, {BatchSize * NumConcurrentMedia, NumTiles,
                             NumPatchesLocal + NumPaddingPatches, Dim});
  GlobalHidden = take(GlobalHidden, mx::array({SliceIndex}), 2);
  GlobalHidden = reshape(GlobalHidden, {BatchSize, NumConcurrentMedia, NumTiles,
                                        NumPatchesLocal, Dim});
  std::vector<mx::array> AllIntermediateHiddenStates = EncoderOutput.second;
  mx::array IntermediateHiddenStates =
      mx::stack(AllIntermediateHiddenStates, -1);
  //   intermediate_hidden_states = intermediate_hidden_states[
  // 	..., self.intermediate_layers_indices
  // ]
  IntermediateHiddenStates =
      take(IntermediateHiddenStates,
           mx::array(IntermediateLayersIndices.data(),
                     {static_cast<int>(IntermediateLayersIndices.size())}),
           -1);
  IntermediateHiddenStates = reshape(IntermediateHiddenStates,
                                     {BatchSize * NumConcurrentMedia, NumTiles,
                                      NumPatchesLocal + NumPaddingPatches, -1});
  // intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
  IntermediateHiddenStates =
      take(IntermediateHiddenStates, mx::arange(SliceIndex), 2);
  IntermediateHiddenStates =
      reshape(IntermediateHiddenStates,
              {BatchSize, NumConcurrentMedia, NumTiles, NumPatchesLocal, -1});
  mx::array FinalHidden =
      mx::concatenate({GlobalHidden, IntermediateHiddenStates}, -1);
  return FinalHidden;
}

std::unordered_map<std::string, mx::array> VisionModel::sanitize(
    const std::unordered_map<std::string, mx::array> &Weights) {
  std::unordered_map<std::string, mx::array> SanitizedWeights;
  for (const auto &Pair : Weights) {
    if (Pair.first.find("position_ids") != std::string::npos) {
      continue;
    }
    if (Pair.first.find("patch_embedding.weight") != std::string::npos) {
      if (checkArrayShape(Pair.second)) {
        SanitizedWeights.insert({Pair.first, Pair.second});
      } else {
        SanitizedWeights.insert(
            {Pair.first, transpose(Pair.second, {0, 2, 3, 1})});
      }
    } else {
      SanitizedWeights.insert({Pair.first, Pair.second});
    }
  }
  return SanitizedWeights;
}
} // namespace mllama
