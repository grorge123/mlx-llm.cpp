#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/transformer.h"
#include <vector>
int main() {
  mx::StreamOrDevice Device =
      mx::metal::is_available() ? mx::Device::gpu : mx::Device::cpu;
  int VocabSize = 32000;
  float NormEps = 1e-5;
  float RopeTheta = 10000.0;
  bool RopeTraditional = false;
  std::cout << "Create Model...\n";
  auto Model = Transformer(4096, std::vector<int>{11008}, VocabSize, 32,
                           std::vector<int>{32}, std::vector<int>{32}, NormEps,
                           RopeTheta, RopeTraditional);
  std::cout << "Load Model...\n";
  Model.update(llamaToMlxllm("../llama2-7b", Device));
  return 0;
}