#include "language.h"
#include "activations.h"
#include "base.h"
#include "embedding.h"
#include "linear.h"
#include "normalization.h"
#include "positional_encoding.h"
#include <cmath>
#include <memory>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <stdexcept>
#include <vector>

namespace gemma3 {

TextConfig TextConfig::fromDict(const simdjson::dom::object &Obj) {
  TextConfig Config;
  auto SResult = Obj["model_type"].get_string();
  if (!SResult.error())
    Config.ModelType = std::string(SResult.value());
  auto IResult = Obj["hidden_size"].get_int64();
  if (!IResult.error())
    Config.HiddenSize = static_cast<int>(IResult.value());
  IResult = Obj["num_hidden_layers"].get_int64();
  if (!IResult.error())
    Config.NumHiddenLayers = static_cast<int>(IResult.value());
  IResult = Obj["intermediate_size"].get_int64();
  if (!IResult.error())
    Config.IntermediateSize = static_cast<int>(IResult.value());
  IResult = Obj["num_attention_heads"].get_int64();
  if (!IResult.error())
    Config.NumAttentionHeads = static_cast<int>(IResult.value());
  IResult = Obj["head_dim"].get_int64();
  if (!IResult.error())
    Config.HeadDim = static_cast<int>(IResult.value());
  auto DResult = Obj["rms_norm_eps"].get_double();
  if (!DResult.error())
    Config.RmsNormEps = static_cast<float>(DResult.value());
  IResult = Obj["vocab_size"].get_int64();
  if (!IResult.error())
    Config.VocabSize = static_cast<int>(IResult.value());
  IResult = Obj["num_key_value_heads"].get_int64();
  if (!IResult.error())
    Config.NumKeyValueHeads = static_cast<int>(IResult.value());
  DResult = Obj["rope_global_base_freq"].get_double();
  if (!DResult.error())
    Config.RopeGlobalBaseFreq = static_cast<float>(DResult.value());
  DResult = Obj["rope_local_base_freq"].get_double();
  if (!DResult.error())
    Config.RopeLocalBaseFreq = static_cast<float>(DResult.value());
  auto BResult = Obj["rope_traditional"].get_bool();
  if (!BResult.error())
    Config.RopeTraditional = BResult.value();
  DResult = Obj["query_pre_attn_scalar"].get_double();
  if (!DResult.error())
    Config.QueryPreAttnScalar = static_cast<float>(DResult.value());
  IResult = Obj["sliding_window"].get_int64();
  if (!IResult.error())
    Config.SlidingWindow = static_cast<int>(IResult.value());
  IResult = Obj["mm_tokens_per_image"].get_int64();
  if (!IResult.error())
    Config.MmTokensPerImage = static_cast<int>(IResult.value());
  IResult = Obj["sliding_window_pattern"].get_int64();
  if (!IResult.error())
    Config.SlidingWindowPattern = static_cast<int>(IResult.value());
  return Config;
}

RMSNorm::RMSNorm(int Dims, float Eps) : Eps(Eps) {
  registerParameter("weight", mx::ones({Dims}));
}

mx::array RMSNorm::forward(const mx::array &X) {
  return mx::fast::rms_norm(X, 1.0 + Parameters.at("weight"), Eps);
}

Attention::Attention(const TextConfig &Config, int LayerIdx)
    : NHeads(Config.NumAttentionHeads), NKVHeads(Config.NumKeyValueHeads),
      Repeats(Config.NumAttentionHeads / Config.NumKeyValueHeads),
      HeadDim(Config.HeadDim), LayerIdx(LayerIdx),
      Scale(std::pow(Config.QueryPreAttnScalar, -0.5)),
      QNorm(HeadDim, Config.RmsNormEps), KNorm(HeadDim, Config.RmsNormEps) {
  registerModule("q_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NHeads * HeadDim, false));
  registerModule("k_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NKVHeads * HeadDim, false));
  registerModule("v_proj", std::make_shared<nn::Linear>(
                               Config.HiddenSize, NKVHeads * HeadDim, false));
  registerModule("o_proj", std::make_shared<nn::Linear>(
                               NHeads * HeadDim, Config.HiddenSize, false));
  registerModule("q_norm",
                 std::make_shared<gemma3::RMSNorm>(HeadDim, Config.RmsNormEps));
  registerModule("k_norm",
                 std::make_shared<gemma3::RMSNorm>(HeadDim, Config.RmsNormEps));
  IsSliding = ((LayerIdx + 1) % Config.SlidingWindowPattern) != 0;
  registerModule("rope", std::make_shared<nn::RoPE>(
                             HeadDim, Config.RopeTraditional,
                             (IsSliding ? Config.RopeLocalBaseFreq
                                        : Config.RopeGlobalBaseFreq)));
}

mx::array Attention::forward(const mx::array &X,
                             const std::optional<mx::array> &Mask,
                             const std::optional<vlm::KVCache *> &Cache) {
  auto Shape = X.shape();
  int B = Shape[0], L = Shape[1];
  mx::array Queries =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["q_proj"])->forward(X);
  mx::array Keys =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["k_proj"])->forward(X);
  mx::array Values =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["v_proj"])->forward(X);
  Queries = transpose(reshape(Queries, {B, L, NHeads, HeadDim}), {0, 2, 1, 3});
  Keys = transpose(reshape(Keys, {B, L, NKVHeads, HeadDim}), {0, 2, 1, 3});
  Values = transpose(reshape(Values, {B, L, NKVHeads, HeadDim}), {0, 2, 1, 3});
  Queries = std::dynamic_pointer_cast<gemma3::RMSNorm>(Submodules["q_norm"])
                ->forward(Queries);
  Keys = std::dynamic_pointer_cast<gemma3::RMSNorm>(Submodules["k_norm"])
             ->forward(Keys);
  if (Cache.has_value()) {
    Queries = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                  ->forward(Queries, Cache.value() ? Cache.value()->Offset : 0);
    Keys = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
               ->forward(Keys, Cache.value() ? Cache.value()->Offset : 0);
    std::tie(Keys, Values) = Cache.value()->updateAndFetch(Keys, Values);
  } else {
    Queries = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                  ->forward(Queries);
    Keys =
        std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])->forward(Keys);
  }
  if (Mask.has_value() &&
      Mask.value().shape().back() != Keys.shape().at(Keys.shape().size() - 2)) {
    mx::array M =
        take(Mask.value(), {-Keys.shape().at(Keys.shape().size() - 2)}, -2);
    return std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
        ->forward(transpose(reshape(mx::fast::scaled_dot_product_attention(
                                        Queries, Keys, Values, Scale, M),
                                    {B, L, -1}),
                            {0, 2, 1}));
  }
  return Mask.has_value()
             ? std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
                   ->forward(transpose(
                       reshape(mx::fast::scaled_dot_product_attention(
                                   Queries, Keys, Values, Scale, Mask.value()),
                               {B, L, -1}),
                       {0, 2, 1}))
             : std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
                   ->forward(
                       transpose(reshape(mx::fast::scaled_dot_product_attention(
                                             Queries, Keys, Values, Scale),
                                         {B, L, -1}),
                                 {0, 2, 1}));
}

MLP::MLP(int Dim, int HiddenDim) {
  registerModule("gate_proj",
                 std::make_shared<nn::Linear>(Dim, HiddenDim, false));
  registerModule("down_proj",
                 std::make_shared<nn::Linear>(HiddenDim, Dim, false));
  registerModule("up_proj",
                 std::make_shared<nn::Linear>(Dim, HiddenDim, false));
}

mx::array MLP::forward(const mx::array &X) {
  mx::array A = std::dynamic_pointer_cast<nn::Linear>(Submodules["gate_proj"])
                    ->forward(X);
  A = geluApprox(A);
  mx::array B =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["up_proj"])->forward(X);
  A = A * B;
  return std::dynamic_pointer_cast<nn::Linear>(Submodules["down_proj"])
      ->forward(A);
}

TransformerBlock::TransformerBlock(const TextConfig &Config, int LayerIdx)
    : NumAttentionHeads(Config.NumAttentionHeads),
      HiddenSize(Config.HiddenSize) {
  registerModule("self_attn", std::make_shared<Attention>(Config, LayerIdx));
  registerModule(
      "mlp", std::make_shared<MLP>(Config.HiddenSize, Config.IntermediateSize));
  registerModule("input_layernorm", std::make_shared<gemma3::RMSNorm>(
                                        Config.HiddenSize, Config.RmsNormEps));
  registerModule(
      "post_attention_layernorm",
      std::make_shared<gemma3::RMSNorm>(Config.HiddenSize, Config.RmsNormEps));
  registerModule(
      "pre_feedforward_layernorm",
      std::make_shared<gemma3::RMSNorm>(Config.HiddenSize, Config.RmsNormEps));
  registerModule(
      "post_feedforward_layernorm",
      std::make_shared<gemma3::RMSNorm>(Config.HiddenSize, Config.RmsNormEps));
}

mx::array
TransformerBlock::forward(const mx::array &X,
                          const std::optional<mx::array> &Mask,
                          const std::optional<vlm::KVCache *> &Cache) {
  mx::array R = std::dynamic_pointer_cast<Attention>(Submodules["self_attn"])
                    ->forward(std::dynamic_pointer_cast<gemma3::RMSNorm>(
                                  Submodules["input_layernorm"])
                                  ->forward(X),
                              Mask, Cache);
  mx::array H = X + std::dynamic_pointer_cast<gemma3::RMSNorm>(
                        Submodules["post_attention_layernorm"])
                        ->forward(R);
  R = std::dynamic_pointer_cast<MLP>(Submodules["mlp"])
          ->forward(std::dynamic_pointer_cast<gemma3::RMSNorm>(
                        Submodules["pre_feedforward_layer_norm"])
                        ->forward(H));
  return H + std::dynamic_pointer_cast<gemma3::RMSNorm>(
                 Submodules["post_feedforward_layernorm"])
                 ->forward(R);
}

Gemma3Model::Gemma3Model(const TextConfig &Config) : Config(Config) {
  if (Config.VocabSize <= 0) {
    assumingUnreachable();
  }
  registerModule("embed_tokens", std::make_shared<nn::Embedding>(
                                     Config.VocabSize, Config.HiddenSize));
  for (int I = 0; I < Config.NumHiddenLayers; I++) {
    Layers.push_back(std::make_shared<TransformerBlock>(Config, I));
  }
  registerLayer("layers", Layers);
  registerModule("norm", std::make_shared<gemma3::RMSNorm>(Config.HiddenSize,
                                                           Config.RmsNormEps));
}

mx::array
Gemma3Model::forward(const mx::array &Inputs,
                     const std::optional<mx::array> &InputsEmbeds,
                     const std::optional<mx::array> &Mask,
                     const std::optional<std::vector<vlm::KVCache *>> &Cache) {
  mx::array H =
      InputsEmbeds.has_value()
          ? InputsEmbeds.value()
          : std::dynamic_pointer_cast<nn::Embedding>(Submodules["embed_tokens"])
                ->forward(Inputs);
  H = H *
      mlx::core::astype(
          mx::array(std::pow(Config.HiddenSize, 0.5), mx::bfloat16), H.dtype());
  const std::vector<vlm::KVCache *> &CacheValue =
      Cache.has_value()
          ? Cache.value()
          : std::vector<vlm::KVCache *>(Config.NumHiddenLayers, nullptr);
  mx::array FullMask = mx::array({});
  mx::array SlidingWindowMask = mx::array({});
  if (!Mask.has_value()) {
    int J = Config.SlidingWindowPattern;
    FullMask = vlm::createAttentionMask(
        H, std::vector<vlm::KVCache *>(CacheValue.begin() + J - 1,
                                       CacheValue.begin() + J));
    SlidingWindowMask = vlm::createAttentionMask(H, CacheValue);
  }
  for (size_t I = 0; I < Layers.size(); I++) {
    bool IsSliding =
        (I % Config.SlidingWindowPattern == Config.SlidingWindowPattern - 1);
    mx::array MaskLocal = mx::array({});
    if (!Mask.has_value() && IsSliding)
      MaskLocal = SlidingWindowMask;
    else if (!Mask.has_value())
      MaskLocal = FullMask;
    else
      MaskLocal = Mask.value();
    H = dynamic_cast<TransformerBlock *>(Layers[I].get())
            ->forward(H, MaskLocal, CacheValue[I]);
  }
  return std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["norm"])->forward(H);
}

LanguageModel::LanguageModel(const TextConfig &Config) : Config(Config) {
  registerModule("model", std::make_shared<Gemma3Model>(Config));
  registerModule("lm_head", std::make_shared<nn::Linear>(
                                Config.HiddenSize, Config.VocabSize, false));
}

LanguageModelOutput LanguageModel::forward(
    const mx::array &Inputs, const std::optional<mx::array> &InputsEmbeds,
    const std::optional<mx::array> &Mask,
    const std::optional<std::vector<vlm::KVCache *>> &Cache) {
  mx::array Out = std::dynamic_pointer_cast<Gemma3Model>(Submodules["model"])
                      ->forward(Inputs, InputsEmbeds, Mask, Cache);
  Out = std::dynamic_pointer_cast<nn::Linear>(Submodules["lm_head"])
            ->forward(Out);
  return LanguageModelOutput{Out};
}

std::unordered_map<std::string, mx::array> LanguageModel::sanitize(
    const std::unordered_map<std::string, mx::array> &Weights) {
  std::unordered_map<std::string, mx::array> Sanitized;
  if (Weights.find("lm_head.weight") == Weights.end())
    Sanitized.insert({"language_model.lm_head.weight",
                      Weights.at("language_model.model.embed_tokens.weight")});
  for (auto &Pair : Weights) {
    if (Pair.first.find("self_attn.rotary_emb.inv_freq") == std::string::npos)
      Sanitized.insert({Pair.first, Pair.second});
  }
  return Sanitized;
}

int LanguageModel::headDim() const { return Config.HeadDim; }

int LanguageModel::nKvHeads() const { return Config.NumKeyValueHeads; }

// std::vector<void *> LanguageModel::makeCache() {
//   std::vector<void *> caches;
//   for (int i = 0; i < Config.NumHiddenLayers; i++) {
//     if (i % Config.SlidingWindowPattern == Config.SlidingWindowPattern - 1)
//       caches.push_back(new KVCache(Config.HeadDim, Config.NumKeyValueHeads));
//     else
//       caches.push_back(new RotatingKVCache(Config.SlidingWindow, 0));
//   }
//   return caches;
// }
} // namespace gemma3