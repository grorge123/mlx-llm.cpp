#pragma once
/// @file mlx_compat.h
/// @brief MLX version compatibility layer for mlx-llm.cpp
///
/// This header provides compatibility macros and type aliases to support
/// both MLX <0.30 and MLX >=0.30 APIs without source code modifications.

#include <mlx/mlx.h>

//==============================================================================
// Version Detection
//==============================================================================

/// Check if MLX version is at least major.minor.patch
/// MLX_VERSION_NUMERIC = 100000*MAJOR + 1000*MINOR + PATCH
#define MLX_VERSION_AT_LEAST(major, minor, patch) \
  (MLX_VERSION_NUMERIC >= (100000 * (major) + 1000 * (minor) + (patch)))

/// Convenience macro for MLX 0.30+ detection
#define MLX_0_30_OR_LATER MLX_VERSION_AT_LEAST(0, 30, 0)

//==============================================================================
// Shape Type Compatibility
//==============================================================================
// MLX 0.30+ changed array.shape() return type from std::vector<int> to
// mx::Shape (SmallVector<int32_t>). Both types support the same operations
// (size(), operator[], begin(), end(), insert(), push_back()), so using
// 'auto' is the most portable approach. For explicit type declarations,
// use these aliases:

#if MLX_0_30_OR_LATER
using MlxShape = mx::Shape;
using MlxStrides = mx::Strides;
#else
using MlxShape = std::vector<int>;
using MlxStrides = std::vector<int64_t>;
#endif

//==============================================================================
// Quantize Result Access
//==============================================================================
// MLX 0.30+ changed mx::quantize() return type from
// std::tuple<mx::array, mx::array, mx::array> to std::vector<mx::array>.
// These macros provide unified access to the quantize results.

#if MLX_0_30_OR_LATER
#define MLX_QUANTIZE_WEIGHTS(q) ((q)[0])
#define MLX_QUANTIZE_SCALES(q) ((q)[1])
#define MLX_QUANTIZE_BIASES(q) ((q)[2])
#else
#define MLX_QUANTIZE_WEIGHTS(q) (std::get<0>(q))
#define MLX_QUANTIZE_SCALES(q) (std::get<1>(q))
#define MLX_QUANTIZE_BIASES(q) (std::get<2>(q))
#endif

//==============================================================================
// Scaled Dot Product Attention Wrapper
//==============================================================================
// MLX 0.30+ added a 'mask_mode' string parameter to
// mx::fast::scaled_dot_product_attention(). This wrapper provides a
// unified interface that works with both old and new APIs.

namespace mlx_compat {

/// @brief Version-compatible wrapper for scaled_dot_product_attention
/// @param queries Query tensor [batch, heads, seq_len, head_dim]
/// @param keys Key tensor [batch, heads, kv_len, head_dim]
/// @param values Value tensor [batch, heads, kv_len, head_dim]
/// @param scale Attention scale factor (typically 1/sqrt(head_dim))
/// @param mask Optional attention mask
/// @return Attention output tensor
inline mx::array scaled_dot_product_attention(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    float scale, std::optional<mx::array> mask = std::nullopt) {
#if MLX_0_30_OR_LATER
  if (mask.has_value()) {
    return mx::fast::scaled_dot_product_attention(queries, keys, values, scale,
                                                  "", mask.value());
  }
  return mx::fast::scaled_dot_product_attention(queries, keys, values, scale);
#else
  if (mask.has_value()) {
    return mx::fast::scaled_dot_product_attention(queries, keys, values, scale,
                                                  mask.value());
  }
  return mx::fast::scaled_dot_product_attention(queries, keys, values, scale);
#endif
}

} // namespace mlx_compat
