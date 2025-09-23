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
  // std::string ModelPath = "../../gemma-3-4b-pt-4bit";
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
  const auto Start{std::chrono::steady_clock::now()};
  auto TokenList = Model->generate({}, std::nullopt, true, Kwargs);
  for (auto &Token : TokenList) {
    std::cout << Token << " ";
  }
  std::cout << std::endl;
  mx::array TokenArr =
      mx::array(TokenList.data(), {static_cast<int>(TokenList.size())});
  mx::save("output.npy", TokenArr);
  std::system((std::string("python3.10 ../example/decode.py ") + ModelPath +
               " output.npy")
                  .c_str());
  const auto End{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> ElapsedSeconds{End - Start};
  spdlog::info("Elapsed time: {} s. TPS: {}.", ElapsedSeconds.count(),
               TokenList.size() / ElapsedSeconds.count());

  return 0;
}