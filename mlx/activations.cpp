#include "activations.h"
#include "math.h"
#include <mlx/ops.h>
namespace mlx::core {
mx::array gelu(mx::array X) {
  return X * (1 + mx::erf(X / std::sqrt(2.0))) / 2.0;
}
mx::array silu(mx::array X) { return X * mx::sigmoid(X); }
mx::array geluApprox(mx::array X) {
  return 0.5 * X *
         (1 + mx::tanh((X + 0.044715 * X * X * X) * std::sqrt(2.0 / M_PI)));
}
} // namespace mlx::core