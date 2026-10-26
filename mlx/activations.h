#pragma once
#include "base.h"
#include <cmath>
#include <mlx/array.h>
namespace mlx::core {
mx::array gelu(const mx::array &X);
mx::array silu(const mx::array &X);
mx::array geluApprox(const mx::array &X);
} // namespace mlx::core