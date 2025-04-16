#pragma once

#include "mlx/mlx.h"
namespace mx = mlx::core;

namespace vlm {

mx::array topPSampling(
    const mx::array& Logits,
    float TopP,
    float Temperature);

} // namespace vlm