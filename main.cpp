#include "base.h"
#include <iostream>
int main() {
  std::cout << "Hello" << std::endl;
  mlx::core::nn::Module Model;
  Model.loadWeights("ggml-model-f16.gguf"); 
  return 0;
}