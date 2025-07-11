#pragma once
#include "base.h"
#include <mlx/mlx.h>
namespace vlm {

mx::array topPSampling(const mx::array &Logits, float TopP, float Temperature);

} // namespace vlm