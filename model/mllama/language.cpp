#include "language.h"
#include "embedding.h"
#include "linear.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <mlx/array.h>
#include <stdexcept>
namespace mllama {

TextConfig TextConfig::fromDict(const simdjson::dom::object &Obj) {
  TextConfig Config;
  auto SResult = Obj["model_type"].get_string();
  if (!SResult.error()) {
    Config.ModelType = std::string(SResult.value());
  }
  auto IResult = Obj["vocab_size"].get_int64();
  if (!IResult.error()) {
    Config.VocabSize = static_cast<int>(IResult.value());
  }
  IResult = Obj["hidden_size"].get_int64();
  if (!IResult.error()) {
    Config.HiddenSize = static_cast<int>(IResult.value());
  }
  IResult = Obj["intermediate_size"].get_int64();
  if (!IResult.error()) {
    Config.IntermediateSize = static_cast<int>(IResult.value());
  }
  IResult = Obj["num_hidden_layers"].get_int64();
  if (!IResult.error()) {
    Config.NumHiddenLayers = static_cast<int>(IResult.value());
  }
  IResult = Obj["num_attention_heads"].get_int64();
  if (!IResult.error()) {
    Config.NumAttentionHeads = static_cast<int>(IResult.value());
  }
  IResult = Obj["num_key_value_heads"].get_int64();
  if (!IResult.error()) {
    Config.NumKeyValueHeads = static_cast<int>(IResult.value());
  }
  SResult = Obj["hidden_act"].get_string();
  if (!SResult.error()) {
    Config.HiddenAct = std::string(SResult.value());
  }
  IResult = Obj["max_position_embeddings"].get_int64();
  if (!IResult.error()) {
    Config.MaxPositionEmbeddings = static_cast<int>(IResult.value());
  }
  auto DResult = Obj["initializer_range"].get_double();
  if (!DResult.error()) {
    Config.InitializerRange = static_cast<float>(DResult.value());
  }
  DResult = Obj["rms_norm_eps"].get_double();
  if (!DResult.error()) {
    Config.RmsNormEps = static_cast<float>(DResult.value());
  }
  auto BResult = Obj["tie_word_embeddings"].get_bool();
  if (!BResult.error()) {
    Config.TieWordEmbeddings = BResult.value();
  }
  DResult = Obj["rope_theta"].get_double();
  if (!DResult.error()) {
    Config.RopeTheta = static_cast<float>(DResult.value());
  }
  BResult = Obj["rope_traditional"].get_bool();
  if (!BResult.error()) {
    Config.RopeTraditional = BResult.value();
  }
  simdjson::dom::array Layers;
  if (Obj["cross_attention_layers"].get_array().get(Layers) ==
      simdjson::SUCCESS) {
    Config.CrossAttentionLayers.clear();
    for (auto Layer : Layers) {
      auto ILayer = Layer.get_int64();
      if (!ILayer.error()) {
        Config.CrossAttentionLayers.push_back(static_cast<int>(ILayer.value()));
      }
    }
  }
  return Config;
}

MllamaTextCrossAttention::MllamaTextCrossAttention(const TextConfig &Config,
                                                   std::optional<int> LayerIdx)
    : Config(Config), HiddenSize(Config.HiddenSize),
      NumHeads(Config.NumAttentionHeads),
      HeadDim(Config.HiddenSize / Config.NumAttentionHeads),
      NumKeyValueHeads(Config.NumKeyValueHeads),
      NumKeyValueGroups(Config.NumAttentionHeads / Config.NumKeyValueHeads),
      LayerIdx(LayerIdx), Scale(1.0f / std::sqrt(static_cast<float>(HeadDim))) {
  registerModule("q_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumHeads * HeadDim, false)));
  registerModule("k_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumKeyValueHeads * HeadDim, false)));
  registerModule("v_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumKeyValueHeads * HeadDim, false)));
  registerModule("o_proj", std::make_shared<nn::Linear>(nn::Linear(
                               NumHeads * HeadDim, HiddenSize, false)));
  registerModule("q_norm", std::make_shared<nn::RMSNorm>(
                               nn::RMSNorm(HeadDim, Config.RmsNormEps)));
  registerModule("k_norm", std::make_shared<nn::RMSNorm>(
                               nn::RMSNorm(HeadDim, Config.RmsNormEps)));
}

mx::array MllamaTextCrossAttention::forward(
    const mx::array &HiddenStates,
    const std::optional<mx::array> &CrossAttentionStates,
    const std::optional<mx::array> &AttentionMask,
    std::shared_ptr<vlm::BaseCache> Cache) {
  auto Shape = HiddenStates.shape();
  int BatchSize = Shape[0];
  int QLen = Shape[1];

  mx::array Query = std::dynamic_pointer_cast<nn::Linear>(Submodules["q_proj"])
                        ->forward(HiddenStates);

  Query = transpose(reshape(Query, {BatchSize, QLen, NumHeads, HeadDim}),
                    {0, 2, 1, 3});
  mx::array QueryStates =
      std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["q_norm"])
          ->forward(Query);

  mx::array KeyStates = mx::array({}), ValueStates = mx::array({});
  if (CrossAttentionStates.has_value()) {
    KeyStates = std::dynamic_pointer_cast<nn::Linear>(Submodules["k_proj"])
                    ->forward(CrossAttentionStates.value());
    KeyStates = transpose(
        reshape(KeyStates, {BatchSize, -1, NumKeyValueHeads, HeadDim}),
        {0, 2, 1, 3});
    ValueStates = std::dynamic_pointer_cast<nn::Linear>(Submodules["v_proj"])
                      ->forward(CrossAttentionStates.value());
    ValueStates = transpose(
        reshape(ValueStates, {BatchSize, -1, NumKeyValueHeads, HeadDim}),
        {0, 2, 1, 3});
    KeyStates = std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["k_norm"])
                    ->forward(KeyStates);
  } else if (Cache != nullptr && Cache->Offset > 0) {
    std::tie(KeyStates, ValueStates) =
        std::dynamic_pointer_cast<vlm::KVCache>(Cache)->fetch();
  } else {
    auto Splits = mx::split(Query, 2, 1);
    KeyStates = Splits[0];
    ValueStates = Splits[1];
    KeyStates = std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["k_norm"])
                    ->forward(KeyStates);
  }
  mx::array AttnOutput =
      AttentionMask.has_value()
          ? mx::fast::scaled_dot_product_attention(QueryStates, KeyStates,
                                                   ValueStates, Scale,
                                                   AttentionMask.value())
          : mx::fast::scaled_dot_product_attention(QueryStates, KeyStates,
                                                   ValueStates, Scale);

  AttnOutput = reshape(transpose(AttnOutput, {0, 2, 1, 3}),
                       {BatchSize, QLen, HiddenSize});
  return std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
      ->forward(AttnOutput);
}

MllamaTextSelfAttention::MllamaTextSelfAttention(const TextConfig &Config,
                                                 int LayerIdx)
    : Config(Config), HiddenSize(Config.HiddenSize),
      NumHeads(Config.NumAttentionHeads),
      HeadDim(Config.HiddenSize / Config.NumAttentionHeads),
      NumKeyValueHeads(Config.NumKeyValueHeads),
      NumKeyValueGroups(Config.NumAttentionHeads / Config.NumKeyValueHeads),
      Scale(1.0f / std::sqrt(static_cast<float>(HeadDim))), LayerIdx(LayerIdx) {
  registerModule("q_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumHeads * HeadDim, false)));

  registerModule("k_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumKeyValueHeads * HeadDim, false)));
  registerModule("v_proj", std::make_shared<nn::Linear>(nn::Linear(
                               HiddenSize, NumKeyValueHeads * HeadDim, false)));
  registerModule("o_proj", std::make_shared<nn::Linear>(nn::Linear(
                               NumHeads * HeadDim, HiddenSize, false)));
  registerModule("rope",
                 std::make_shared<nn::RoPE>(nn::RoPE(
                     HeadDim, Config.RopeTraditional, Config.RopeTheta, 1)));
}

mx::array
MllamaTextSelfAttention::forward(const mx::array &X,
                                 const std::optional<mx::array> &Mask,
                                 std::shared_ptr<vlm::BaseCache> Cache) {
  auto Shape = X.shape();
  int BatchSize = Shape[0];
  int QLen = Shape[1];

  mx::array QueryStates =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["q_proj"])->forward(X);

  QueryStates = transpose(
      reshape(QueryStates, {BatchSize, QLen, NumHeads, HeadDim}), {0, 2, 1, 3});

  mx::array KeyStates =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["k_proj"])->forward(X);
  KeyStates = transpose(
      reshape(KeyStates, {BatchSize, QLen, NumHeads, HeadDim}), {0, 2, 1, 3});

  mx::array ValueStates =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["k_proj"])->forward(X);
  ValueStates = transpose(
      reshape(ValueStates, {BatchSize, QLen, NumHeads, HeadDim}), {0, 2, 1, 3});

  if (Cache != nullptr) {
    QueryStates = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                      ->forward(QueryStates, Cache->Offset);
    KeyStates = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                    ->forward(KeyStates, Cache->Offset);
    std::tie(KeyStates, ValueStates) =
        Cache->updateAndFetch(KeyStates, ValueStates);
  } else {
    QueryStates = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                      ->forward(QueryStates);
    KeyStates = std::dynamic_pointer_cast<nn::RoPE>(Submodules["rope"])
                    ->forward(KeyStates);
  }

  mx::array AttnOutput =
      Mask.has_value()
          ? mx::fast::scaled_dot_product_attention(
                QueryStates, KeyStates, ValueStates, Scale, Mask.value())
          : mx::fast::scaled_dot_product_attention(QueryStates, KeyStates,
                                                   ValueStates, Scale);
  AttnOutput = reshape(transpose(AttnOutput, {0, 2, 1, 3}),
                       {BatchSize, QLen, HiddenSize});
  return std::dynamic_pointer_cast<nn::Linear>(Submodules["o_proj"])
      ->forward(AttnOutput);
}

MllamaTextMLP::MllamaTextMLP(const TextConfig &Config) {
  registerModule("gate_proj",
                 std::make_shared<nn::Linear>(nn::Linear(
                     Config.HiddenSize, Config.IntermediateSize, false)));
  registerModule("up_proj",
                 std::make_shared<nn::Linear>(nn::Linear(
                     Config.HiddenSize, Config.IntermediateSize, false)));
  registerModule("down_proj",
                 std::make_shared<nn::Linear>(nn::Linear(
                     Config.IntermediateSize, Config.HiddenSize, false)));
}

mx::array MllamaTextMLP::forward(const mx::array &X) {
  mx::array Gate =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["gate_proj"])
          ->forward(X);
  mx::array Activated = Gate * mx::sigmoid(Gate);
  mx::array Up =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["up_proj"])->forward(X);
  return std::dynamic_pointer_cast<nn::Linear>(Submodules["down_proj"])
      ->forward(Activated * Up);
}

MllamaSelfAttentionDecoderLayer::MllamaSelfAttentionDecoderLayer(
    const TextConfig &Config, int LayerIdx)
    : HiddenSize(Config.HiddenSize) {
  registerModule("self_attn",
                 std::make_shared<MllamaTextSelfAttention>(Config, LayerIdx));
  registerModule("mlp", std::make_shared<MllamaTextMLP>(Config));
  registerModule("input_layernorm", std::make_shared<nn::RMSNorm>(nn::RMSNorm(
                                        Config.HiddenSize, Config.RmsNormEps)));
  registerModule("post_attention_layernorm",
                 std::make_shared<nn::RMSNorm>(
                     nn::RMSNorm(Config.HiddenSize, Config.RmsNormEps)));
}

mx::array MllamaSelfAttentionDecoderLayer::forward(
    const mx::array &HiddenStates, const std::optional<mx::array> &Mask,
    std::shared_ptr<vlm::BaseCache> Cache) {
  mx::array Residual = HiddenStates;
  mx::array Normed =
      std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["input_layernorm"])
          ->forward(HiddenStates);
  mx::array Attn = std::dynamic_pointer_cast<MllamaTextSelfAttention>(
                       Submodules["post_attention_layernorm"])
                       ->forward(Normed, Mask, Cache);
  mx::array Out = Residual + Attn;

  Residual = Out;
  Normed = std::dynamic_pointer_cast<nn::RMSNorm>(
               Submodules["post_attention_layernorm"])
               ->forward(Out);
  mx::array MlpOut = std::dynamic_pointer_cast<MllamaTextMLP>(Submodules["mlp"])
                         ->forward(Normed);
  Out = Residual + MlpOut;
  return Out;
}

MllamaCrossAttentionDecoderLayer::MllamaCrossAttentionDecoderLayer(
    const TextConfig &Config, int LayerIdx)
    : HiddenSize(Config.HiddenSize), CrossAttnAttnGate(mx::zeros({1})),
      CrossAttnMlpGate(mx::zeros({1})) {
  registerModule("cross_attn",
                 std::make_shared<MllamaTextCrossAttention>(Config, LayerIdx));
  registerModule("mlp", std::make_shared<MllamaTextMLP>(Config));
  registerModule("input_layernorm", std::make_shared<nn::RMSNorm>(nn::RMSNorm(
                                        Config.HiddenSize, Config.RmsNormEps)));
  registerModule("post_attention_layernorm",
                 std::make_shared<nn::RMSNorm>(
                     nn::RMSNorm(Config.HiddenSize, Config.RmsNormEps)));
}

mx::array MllamaCrossAttentionDecoderLayer::forward(
    const mx::array &HiddenStates, const mx::array &CrossAttentionStates,
    const std::optional<mx::array> &AttentionMask,
    const std::optional<mx::array> &FullTextRowMaskedOutMask,
    std::shared_ptr<vlm::BaseCache> Cache) {
  mx::array Residual = HiddenStates;
  mx::array Normed =
      std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["input_layernorm"])
          ->forward(HiddenStates);
  mx::array CrossOut =
      std::dynamic_pointer_cast<MllamaTextCrossAttention>(
          Submodules["cross_attn"])
          ->forward(Normed, CrossAttentionStates, AttentionMask, Cache);
  mx::array Out = Residual + mx::tanh(CrossAttnAttnGate) * CrossOut;

  Residual = Out;
  Normed = std::dynamic_pointer_cast<nn::RMSNorm>(
               Submodules["post_attention_layernorm"])
               ->forward(Out);
  mx::array MlpOut = std::dynamic_pointer_cast<MllamaTextMLP>(Submodules["mlp"])
                         ->forward(Normed);
  if (FullTextRowMaskedOutMask.has_value()) {
    // full_text_row_masked_out_mask[:, 0]
    MlpOut = take(FullTextRowMaskedOutMask.value(), mx::array({0}), 1) * MlpOut;
  }
  Out = Residual + mx::tanh(CrossAttnMlpGate) * MlpOut;
  return Out;
}

MllamaTextModel::MllamaTextModel(const TextConfig &Config)
    : Config(Config), VocabSize(Config.VocabSize),
      HiddenSize(Config.HiddenSize) {
  registerModule("embed_tokens", std::make_shared<nn::Embedding>(nn::Embedding(
                                     Config.VocabSize + 8, Config.HiddenSize)));
  for (int LayerIdx = 0; LayerIdx < Config.NumHiddenLayers; ++LayerIdx) {
    bool IsCross = std::find(Config.CrossAttentionLayers.begin(),
                             Config.CrossAttentionLayers.end(),
                             LayerIdx) != Config.CrossAttentionLayers.end();
    if (IsCross) {
      Layers.push_back(
          std::make_unique<MllamaCrossAttentionDecoderLayer>(Config, LayerIdx));
    } else {
      Layers.push_back(
          std::make_unique<MllamaSelfAttentionDecoderLayer>(Config, LayerIdx));
    }
  }
  registerLayer("layers", Layers);
  registerModule("norm", std::make_shared<nn::RMSNorm>(nn::RMSNorm(
                             Config.HiddenSize, Config.RmsNormEps)));
}

mx::array MllamaTextModel::forward(
    const std::optional<mx::array> &InputIds,
    const std::optional<mx::array> &Mask,
    const std::optional<mx::array> &PositionIds,
    const std::optional<mx::array> &CrossAttentionStates,
    const std::optional<mx::array> &CrossAttentionMask,
    const std::optional<mx::array> &FullTextRowMaskedOutMask,
    const std::optional<mx::array> &InputsEmbeds,
    const std::optional<std::vector<std::shared_ptr<vlm::BaseCache>>> &Cache) {
  mx::array InputsEmbedsLocal = mx::array({});
  int BatchSize, SeqLength;
  if (InputIds.has_value() && InputsEmbeds.has_value()) {
    throw std::invalid_argument(
        "You cannot specify both InputIds and InputsEmbeds at the same time");
  }
  if (InputIds.has_value()) {
    auto Shape = InputIds.value().shape();
    BatchSize = Shape[0];
    SeqLength = Shape[1];
    InputsEmbedsLocal =
        std::dynamic_pointer_cast<nn::Embedding>(Submodules["embed_tokens"])
            ->forward(InputIds.value());
  } else if (InputsEmbeds.has_value()) {
    InputsEmbedsLocal = InputsEmbeds.value();
    auto Shape = InputsEmbedsLocal.shape();
    BatchSize = Shape[0];
    SeqLength = Shape[1];
  } else {
    throw std::invalid_argument(
        "You have to specify either InputIds or InputsEmbeds");
  }

  // mx::array PositionIdsLocal = mx::array({});
  // if (!PositionIds.has_value()) {
  //   PositionIdsLocal = mx::arange(SeqLength).unsqueeze(0).repeat(BatchSize,
  //   0);
  // } else {
  //   PositionIdsLocal = PositionIds.value();
  // }

  mx::array HiddenStates = InputsEmbedsLocal;
  mx::array MaskLocal = vlm::createAttentionMask(HiddenStates);

  for (size_t Idx = 0; Idx < Layers.size(); ++Idx) {
    std::shared_ptr<vlm::BaseCache> LayerCache =
        (Cache.has_value() && Idx < Cache.value().size()) ? (Cache.value())[Idx]
                                                          : nullptr;
    bool IsCross = std::find(Config.CrossAttentionLayers.begin(),
                             Config.CrossAttentionLayers.end(),
                             Idx) != Config.CrossAttentionLayers.end();
    if (IsCross) {
      HiddenStates =
          dynamic_cast<MllamaCrossAttentionDecoderLayer *>(Layers[Idx].get())
              ->forward(HiddenStates, CrossAttentionStates.value(),
                        CrossAttentionMask, FullTextRowMaskedOutMask,
                        LayerCache);
    } else {
      HiddenStates =
          dynamic_cast<MllamaSelfAttentionDecoderLayer *>(Layers[Idx].get())
              ->forward(HiddenStates, MaskLocal, LayerCache);
    }
  }

  HiddenStates = std::dynamic_pointer_cast<nn::RMSNorm>(Submodules["norm"])
                     ->forward(HiddenStates);
  return HiddenStates;
}

LanguageModel::LanguageModel(const TextConfig &Config) : Config(Config) {
  registerModule("model", std::make_shared<MllamaTextModel>(Config));
  registerModule("lm_head", std::make_shared<nn::Linear>(nn::Linear(
                                Config.HiddenSize, Config.VocabSize, false)));
}

std::tuple<mx::array, std::optional<mx::array>> LanguageModel::forward(
    const std::optional<mx::array> &InputIds,
    const std::optional<mx::array> &Mask,
    const std::optional<mx::array> &CrossAttentionStates,
    const std::optional<mx::array> &CrossAttentionMask,
    const std::optional<mx::array> &FullTextRowMaskedOutMask,
    const std::optional<mx::array> &InputsEmbeds,
    const std::optional<std::vector<std::shared_ptr<vlm::BaseCache>>> &Cache) {
  mx::array HiddenStates =
      std::dynamic_pointer_cast<MllamaTextModel>(Submodules["model"])
          ->forward(InputIds, Mask, std::nullopt, CrossAttentionStates,
                    CrossAttentionMask, FullTextRowMaskedOutMask, InputsEmbeds,
                    Cache);
  mx::array Logits =
      std::dynamic_pointer_cast<nn::Linear>(Submodules["lm_head"])
          ->forward(HiddenStates);
  return {Logits, CrossAttentionStates};
}

std::unordered_map<std::string, mx::array> LanguageModel::sanitize(
    const std::unordered_map<std::string, mx::array> &Weights) {
  std::unordered_map<std::string, mx::array> Sanitized;
  for (const auto &Pair : Weights) {
    if (Pair.first.find("self_attn.rotary_emb.inv_freq") == std::string::npos) {
      Sanitized.insert({Pair.first, Pair.second});
    }
  }
  return Sanitized;
}

int LanguageModel::headDim() const {
  return Config.HiddenSize / Config.NumAttentionHeads;
}

int LanguageModel::nKvHeads() const { return Config.NumKeyValueHeads; }
int LanguageModel::layers() const { return Config.NumHiddenLayers; }

} // namespace mllama
