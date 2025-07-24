#include "activations.h"
#include "math.h"
#include <mlx/ops.h>
namespace mlx::core {
mx::array gelu(mx::array X) {
  auto Result = X *
                (mx::array({1}, X.dtype()) +
                 mx::erf(X / mx::array({std::sqrt(2)}, X.dtype()))) /
                mx::array({2}, X.dtype());
  // auto Result = X * (1 + mx::erf(X / std::sqrt(2))) / 2;
  return Result;
}
mx::array silu(mx::array X) { return X * mx::sigmoid(X); }
mx::array geluApprox(mx::array X) {
  return mx::array({0.5}, X.dtype()) * X *
         (mx::array({1}, X.dtype()) +
          mx::tanh(mx::array({std::sqrt(2.0 / M_PI)}, X.dtype()) *
                   (X + mx::array({0.044715}, X.dtype()) *
                            mx::power(X, mx::array({3}, X.dtype())))));
}
} // namespace mlx::core