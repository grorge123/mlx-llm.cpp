#include "../mlx/transformer.h"
#include "embedding.h"
#include "linear.h"
#include "transformer.h"
#include <cstddef>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <tuple>
#include <vector>

mx::array RMSNorm::forward(mx::array Input) {
  return mx::fast::rms_norm(Input, 1.0 + Parameters.at("weight"), Eps);
}
std::tuple<mx::array, std::tuple<mx::array, mx::array>>
Attention::forward(mx::array Input, std::optional<mx::array> Mask,
                   std::optional<std::tuple<mx::array, mx::array>> KVCache) {
  const auto &[B, L, D] =
      std::tie(Input.shape()[0], Input.shape()[1], Input.shape()[2]);
  mx::array Queries =
      static_cast<nn::Linear *>(Submodules["q_proj"])->forward(Input);
  mx::array Keys =
      static_cast<nn::Linear *>(Submodules["k_proj"])->forward(Input);
  mx::array Values =
      static_cast<nn::Linear *>(Submodules["v_proj"])->forward(Input);
  Queries = transpose(reshape(Queries, {B, L, NHeads, -1}), {0, 2, 1, 3});
  Keys = transpose(reshape(Keys, {B, L, NKVHeads, -1}), {0, 2, 1, 3});
  Values = transpose(reshape(Values, {B, L, NKVHeads, -1}), {0, 2, 1, 3});

  if (NormQKProj) {
    Queries = static_cast<RMSNorm *>(Submodules["q_norm"])->forward(Queries);
    Keys = static_cast<RMSNorm *>(Submodules["k_norm"])->forward(Keys);
  }

  if (!KVCache) {
    const auto &[KeyCache, ValueCache] = *KVCache;
    Queries = static_cast<nn::RoPE *>(Submodules["nn::RoPE"])
                  ->forward(Queries, KeyCache.shape(2));
    Keys = static_cast<nn::RoPE *>(Submodules["nn::RoPE"])
               ->forward(Keys, KeyCache.shape(2));
    Keys = mx::concatenate({KeyCache, Keys}, 2);
    Values = mx::concatenate({KeyCache, Keys}, 2);
  } else {
    Queries = static_cast<nn::RoPE *>(Submodules["nn::RoPE"])->forward(Queries);
    Keys = static_cast<nn::RoPE *>(Submodules["nn::RoPE"])->forward(Keys);
  }
  mx::array Output = mx::fast::scaled_dot_product_attention(
      Queries, Keys, Values, Scale, Mask);
  Output = reshape(transpose(Output, {0, 2, 1, 3}), {B, L, -1});
  return {static_cast<nn::Linear *>(Submodules["o_proj"])->forward(Output),
          {Keys, Values}};
}
mx::array MLP::forward(mx::array Input) {
  if (Gemma) {
    return static_cast<nn::Linear *>(Submodules["down_proj"])
        ->forward(gelu(
            static_cast<nn::Linear *>(Submodules["gate_proj"])->forward(Input) *
            static_cast<nn::Linear *>(Submodules["up_proj"])->forward(Input)));
  }
  return static_cast<nn::Linear *>(Submodules["down_proj"])
      ->forward(silu(
          static_cast<nn::Linear *>(Submodules["gate_proj"])->forward(Input) *
          static_cast<nn::Linear *>(Submodules["up_proj"])->forward(Input)));
}
std::tuple<mx::array, std::tuple<mx::array, mx::array>>
TransformerBlock::forward(
    mx::array Input, std::optional<mx::array> Mask,
    std::optional<std::tuple<mx::array, mx::array>> KVCachePar) {
  auto [R, KVCache] =
      static_cast<Attention *>(Submodules["attention"])
          ->forward(static_cast<RMSNorm *>(Submodules["attention_norm"])
                        ->forward(Input),
                    Mask, KVCachePar);
  auto H = Input + R;
  R = static_cast<MLP *>(Submodules["mlp"])
          ->forward(static_cast<RMSNorm *>(Submodules["mlp_norm"])->forward(H));
  return {H + R, KVCache};
}
std::tuple<mx::array,
           std::optional<std::vector<std::tuple<mx::array, mx::array>>>>
Transformer::embed(
    mx::array Input,
    std::optional<std::vector<std::tuple<mx::array, mx::array>>> KVCachePar,
    bool Norm) {
  auto H =
      static_cast<mx::nn::Embedding *>(Submodules["embed"])->forward(Input);
  if (Gemma) {
    H = H * (pow(Dim, 0.5));
  }
  std::optional<mx::array> Mask;
  if (H.shape()[1] > 1) {
    Mask = mx::nn::MultiHeadAttention::createAdditiveCausalMask(H.shape()[1]);
    Mask = astype(*Mask, H.dtype());
  }
  std::vector<std::tuple<mx::array, mx::array>> KVCache;
  KVCache.reserve(Layers.size());
  if (KVCachePar) {
    for (size_t Idx = 0; Idx < Layers.size(); Idx++) {
      auto Result = Layers[Idx]->forward(H, Mask, (*KVCachePar)[Idx]);
      H = get<0>(Result);
      KVCache[Idx] = get<1>(Result);
    }
  } else {
    for (size_t Idx = 0; Idx < Layers.size(); Idx++) {
      auto Result = Layers[Idx]->forward(H, Mask, {});
      H = get<0>(Result);
      KVCache[Idx] = get<1>(Result);
    }
  }
  if (Norm) {
    if (!Gemma) {
      return {static_cast<nn::RMSNorm *>(Submodules["norm"])->forward(H), {}};
    }
    return {static_cast<RMSNorm *>(Submodules["norm"])->forward(H), {}};
  }
  return {H, KVCache};
}
std::tuple<mx::array,
           std::optional<std::vector<std::tuple<mx::array, mx::array>>>>
Transformer::forward(
    mx::array Input,
    std::optional<std::vector<std::tuple<mx::array, mx::array>>> KVCachePar) {
  auto [X, KVCache] = embed(Input, KVCachePar);
  mx::array Out = {};
  if (EmbedAsHead) {
    Out = static_cast<mx::nn::Embedding *>(Submodules["token_embed"])
              ->asLinear(X);
  } else {
    Out = static_cast<nn::Linear *>(Submodules["head"])->forward(X);
  }
  return {Out, KVCache};
}
std::tuple<mx::array,
           std::optional<std::vector<std::tuple<mx::array, mx::array>>>>
Transformer::generate(mx::array Input, std::optional<float> Temp) {
  // Reshape Input to input[:, None]
  std::vector<int> ReshapeDim = Input.shape();
  ReshapeDim.insert(ReshapeDim.begin(), 1);
  auto [Logits, KVCache] = forward(reshape(Input, ReshapeDim));
  const int H = Logits.shape()[1] - 1;
  // take logits[:, -1, :]
  Logits = take(Logits, mx::array({H}), 1);
  ReshapeDim = Logits.shape();
  ReshapeDim.erase(ReshapeDim.begin()+1);
  Logits = reshape(Logits, ReshapeDim);
  mx::array Y = {};
  if (Temp == 0) {
    Y = mx::argmax(Logits, -1);
  } else {
    Y = mx::random::categorical(Logits * (1.0 / *Temp));
  }
  return {Y, KVCache};
}
std::tuple<mx::array,
           std::optional<std::vector<std::tuple<mx::array, mx::array>>>>
Transformer::nextGenerate(
    mx::array Y, std::optional<float> Temp,
    std::optional<std::vector<std::tuple<mx::array, mx::array>>> KVCachePar) {
  // Reshape Y to y[:, None]
  std::vector<int> ReshapeDim = Y.shape();
  ReshapeDim.insert(ReshapeDim.begin() + 1, 1);
  auto [Logits, KVCache] = forward(reshape(reshape(Y, ReshapeDim), {}), KVCachePar);
  Logits = squeeze(Logits, 1);
  mx::array NextY = {};
  if (Temp == 0) {
    NextY = mx::argmax(Logits, -1);
  } else {
    NextY = mx::random::categorical(Logits * (1.0 / *Temp));
  }
  return {NextY, KVCache};
}