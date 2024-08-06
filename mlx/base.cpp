#include "base.h"
#include <mlx/array.h>

namespace mlx::core::nn {

mx::array& Module::registerParameter(std::string Name, array &&W) {
  Parameters.insert({Name, W});
  return Parameters.at(Name);
}
void Module::update(std::unordered_map<std::string, mx::array> Weight) {
  
}
void Module::loadWeights(std::string Weight) {
  if (Weight.ends_with(".safetensors")) {
    std::cout << "Loading model from .safetensors file...\n";
    const SafetensorsLoad Loaded = load_safetensors(Weight, Device);
    update(Loaded.first);
  } else if (Weight.ends_with(".gguf")) {
    std::cout << "Loading model from .gguf file...\n";
    const GGUFLoad Loaded = load_gguf(Weight, Device);
    update(Loaded.first);
  } else {
    std::cout << "Loading model from weight...\n";
  }
}
} // namespace mlx::core::nn