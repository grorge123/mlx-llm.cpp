#include "base.h"
#include "model/converter.h"
#include "model/gemma3/gemma3.h"
#include "model/mllama/mllama.h"
#include "model/vlm_base.h"
#include <memory>
#include <mlx/io.h>
#include <optional>
#include <string>
#include <variant>
int main() {
  // std::string ModelPath = "../../Llama-3.2-11B-Vision-Instruct-4bit/";
  // auto Model = mllama::Model::fromPretrained(ModelPath, {64, 4});
  std::string ModelPath = "../../gemma-3-4b-it-bf16";
  auto Model = gemma3::Model::fromPretrained(ModelPath);
  auto InputIds = mx::load("../example/input_ids.npy");
  auto PixelValues = mx::load("../example/pixel_values.npy");
  auto Mask = mx::load("../example/mask.npy");
  std::map<std::string, std::variant<mx::array, int, float, std::string>>
      Kwargs;
  Kwargs.insert({"image_token_index", 262144});
  Kwargs.insert({"input_ids", InputIds});
  Kwargs.insert({"pixel_values", PixelValues});
  Kwargs.insert({"mask", Mask});
  auto ToeknList = vlm::generate(std::dynamic_pointer_cast<vlm::Module>(Model),
                                 {}, std::nullopt, true, Kwargs);
  for (auto &Token : ToeknList) {
    std::cout << Token << " ";
  }

  return 0;
}