#include "positional_encoding.h"

namespace mlx::core::nn {
mx::array RoPE::forward(const mx::array &Input, int Offset) {
  return mx::fast::rope(Input, Dims, Tranditional, Base, Scale, Offset);
}

} // namespace mlx::core::nn