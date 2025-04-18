#include "base.h"
#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/llm/registry.h"
#include "model/llm/transformer.h"
#include "model/utils.h"
#include "prompt/prompt.h"
#include "spdlog/spdlog.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/dtype.h>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>
using tokenizers::Tokenizer;

std::string loadBytesFromFile(const std::string &Path) {
  std::ifstream Fs(Path, std::ios::in | std::ios::binary);
  if (Fs.fail()) {
    std::cerr << "Cannot open " << Path << std::endl;
    exit(1);
  }
  std::string Data;
  Fs.seekg(0, std::ios::end);
  const size_t Size = static_cast<size_t>(Fs.tellg());
  Fs.seekg(0, std::ios::beg);
  Data.resize(Size);
  Fs.read(Data.data(), Size);
  return Data;
}

int main() {
  spdlog::debug("Device: {}, Metal avaiuable: {}.",
                (mx::default_device() == mx::Device::cpu ? "CPU" : "GPU"),
                mx::metal::is_available());
  auto Tok =
      Tokenizer::FromBlobJSON(loadBytesFromFile("../tokenizer-llama3.json"));
  const int MaxToken = 512;
  spdlog::info("Create Model...");
  auto Model = llm::llama38b();
  // auto Model = llama27bChat();
  // auto Model = tinyLlama11BChatV10();
  spdlog::info("Load Model...");
  // Model->update(llamaToMlxllm("../llama2-7b"));
  Model->update(llamaToMlxllm("../llama3-8b"));
  // Model->update(llamaToMlxllm("../tiny"));
  Model =
      std::dynamic_pointer_cast<llm::Transformer>(Model->toQuantized(64, 4));
  auto W = Model->getWeigts();
  saveWeights(W, "Llama-3-8B-4bit-64g.safetensors");
  spdlog::info("Start generate...");
  const LLaMA3Prompt ModelPrmopt;
  std::string Prompt = "Where are you from?";
  auto Result = Model->generate(Prompt, ModelPrmopt, MaxToken, true, Tok);

  const auto Start{std::chrono::steady_clock::now()};

  std::cout << std::endl;
  const auto End{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> ElapsedSeconds{End - Start};
  spdlog::info("Elapsed time: {} s. TPS: {}.", ElapsedSeconds.count(),
               Result.TokenList.size() / ElapsedSeconds.count());
  return 0;
}