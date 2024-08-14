#include "mlx/mlx.h"
#include "../model/converter.h"
#include "../model/transformer.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <mlx/array.h>
#include <mlx/dtype.h>
int main(){
  const mx::StreamOrDevice Device =
      mx::metal::is_available() ? mx::Device::gpu : mx::Device::cpu;
  auto Model = Transformer(32, std::vector<int>{64}, 1024, 3,
                           std::vector<int>{5}, std::vector<int>{4}, 1e-5,
                           10000, false);
  Model.update(weightsToMlx("../test/test_model.safetensors", Device));
  mx::array Input = mx::array({{1,23,35,48,87,62},{6}});
  std::cout << "Start Generate..." << std::endl;
  Model.generate(Input);
}