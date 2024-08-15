#include "../mlx/linear.h"
#include "../model/converter.h"
#include "../model/transformer.h"
#include "base.h"
#include "mlx/mlx.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <mlx/array.h>
#include <mlx/dtype.h>
#include <vector>

int main() {
  // mx::array Input = mx::array({{1, 2, 3, 4, 5, 6}, {6}});
  // auto Linear = mx::nn::Linear(6, 12, false);
  // Linear.update(weightsToMlx("../test/linear.safetensors", Device));
  // // array([0.233722, -0.808297, -3.21808, ..., -3.49855, -2.01105,
  // -1.09417],
  // // dtype=float32)
  // std::cout << Linear.forward(Input) << std::endl;
  // printVec(Linear.forward(Input).shape());
  // auto Rmsnorm = RMSNorm(6, 1e-5);
  // Rmsnorm.update(weightsToMlx("../test/rmsnorm.safetensors", Device));
  // std::cout << Rmsnorm.forward(Input) << std::endl;
  // printVec(Rmsnorm.forward(Input).shape());
  // auto Embedding = mx::nn::Embedding(10, 6);
  // Embedding.update(weightsToMlx("../test/embed.safetensors", Device));
  // std::cout << Embedding.forward(Input) << std::endl;
  // Input = mx::array({{1, 2, 3, 4, 5, 6}, {1, 1, 6}});
  // auto Atten = Attention(6, 10, 5, 6);
  // Atten.update(weightsToMlx("../test/attention.safetensors", Device));
  // auto AttenResult = Atten.forward(Input);
  // std::cout << std::get<0>(AttenResult) << std::endl;
  // std::cout << std::get<0>(std::get<1>(AttenResult)) << std::endl;
  // std::cout << std::get<1>(std::get<1>(AttenResult)) << std::endl;

  auto Model =
      Transformer(32, std::vector<int>{64}, 1024, 3, std::vector<int>{6},
                  std::vector<int>{2}, 1e-5, {}, false, 10000);
  Model.update(weightsToMlx("../test/test_model.safetensors"));
  mx::array Input = mx::array({{1, 23, 35, 48, 87, 62}, {6}});
  std::cout << "Start Generate..." << std::endl;
  int MaxLen = 12;
  auto [Y, KVCache] = Model.generate(Input, 0);
  std::vector<int> Answer;
  while (MaxLen--) {
    std::cout << Y << std::endl;
    Answer.push_back(Y.data<int>()[0]);
    auto [NY, NKVCache] = Model.nextGenerate(Y, 0, KVCache);

    Y = NY, KVCache = NKVCache;
  }
  std::cout << "Answer:" << std::endl;
  printVec(Answer);
}