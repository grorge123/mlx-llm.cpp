#include "base.h"
#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/registry.h"
#include "model/transformer.h"
#include "prompt/llama.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/dtype.h>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>
using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}
enum AnserSataus {
  STOP,
  WAIT,
  GO,
};
AnserSataus answerSataus(std::string Text, std::string End) {
  if (Text.ends_with(End)) {
    return STOP;
  }
  for (int Idx = 1; Idx < End.size(); Idx++) {
    if (Text.ends_with(End.substr(0, Idx))) {
      return WAIT;
    }
  }
  return GO;
}
int main() {
  std::cout << mx::default_device() << " " << mx::metal::is_available()
            << std::endl;
  mx::set_default_device(mx::Device::gpu);
  auto Tok = Tokenizer::FromBlobJSON(LoadBytesFromFile("../tokenizer.json"));
  const int MaxToken = 512;
  mx::array Token = mx::array({{1, 23, 35, 48, 87, 62}, {6}});
  std::cout << "Create Model...\n";
  const int VocabSize = 32000;
  const float NormEps = 1e-5;
  const float RopeTheta = 10000.0;
  const bool RopeTraditional = false;
  auto Model = llama38b();
  std::cout << "Load Model...\n";
  // Model.update(llamaToMlxllm("../llama2-7b"));
  Model.update(llamaToMlxllm("../llama3-8b"));
  std::cout << "Start generate...\n";
  const TinyLLaMAPrompt Prmopt;
  const std::vector<int> Ids = Tok->Encode("Where are you from?");
  Token = mx::array(Ids.data(), {static_cast<int>(Ids.size())}, mx::int32);
  std::vector<int32_t> TokenList;
  std::string Answer;
  int Skip = 0;
  int TokenCount = 0;
  auto [Y, KVCache] = Model.generate(Token, 0.1);
  while (true) {
    TokenCount++;
    if (TokenCount > MaxToken) {
      break;
    }
    eval(Y);
    std::vector<int32_t> Tokens;
    auto *Data = Y.data<int32_t>();
    for (int Idx = 0; Idx < Y.size(); Idx++) {
      Tokens.emplace_back(Data[Idx]);
    }
    // TODO: break when the token is the eos_token_id
    TokenList.insert(TokenList.end(), Tokens.begin(), Tokens.end());
    Answer = Tok->Decode(TokenList);
    const AnserSataus Status = answerSataus(Answer, Prmopt.TextEnd);
    if (Status == STOP) {
      break;
    }
    if (Status == GO) {
      std::cout << Answer.substr(Skip) << std::flush;
      Skip = Answer.size();
    }
    auto [NY, NKVCache] = Model.nextGenerate(Y, 0.1, KVCache);
    Y = NY, KVCache = NKVCache;
  }
  return 0;
}