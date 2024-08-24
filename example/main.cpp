#include "base.h"
#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/registry.h"
#include "model/transformer.h"
#include "model/utils.h"
#include "prompt/prompt.h"
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
enum AnserSataus {
  STOP,
  WAIT,
  GO,
};
AnserSataus answerSataus(std::string Text, std::string End) {
  if (endsWith(Text, End)) {
    return STOP;
  }
  for (int Idx = 1; static_cast<int>(End.size()); Idx++) {
    if (endsWith(Text, End.substr(0, Idx))) {
      return WAIT;
    }
  }
  return GO;
}
int main() {
  std::cout << mx::default_device() << " " << mx::metal::is_available()
            << std::endl;
  mx::set_default_device(mx::Device::gpu);
  auto Tok =
      Tokenizer::FromBlobJSON(loadBytesFromFile("../tokenizer_llama2.json"));
  const int MaxToken = 512;
  mx::array Token = mx::array({{1, 23, 35, 48, 87, 62}, {6}});
  std::cout << "Create Model...\n";
  auto *Model = tinyLlama11BChatV10();
  // auto Model = llama27bChat();
  std::cout << "Load Model...\n";
  // Model.update(llamaToMlxllm("../llama2-7b"));
  Model->update(llamaToMlxllm("../tiny"));
  std::cout << "Start generate...\n";
  const TinyLLaMAPrompt Prmopt;
  const std::vector<int> Ids = Tok->Encode("Where are you from?");
  Token = mx::array(Ids.data(), {static_cast<int>(Ids.size())}, mx::int32);
  std::vector<int32_t> TokenList;
  std::string Answer;
  int Skip = 0;
  int TokenCount = 0;
  const auto Start{std::chrono::steady_clock::now()};
  auto [Y, KVCache] = Model->generate(Token, 0.1);
  while (true) {
    TokenCount++;
    if (TokenCount > MaxToken) {
      break;
    }
    eval(Y);
    std::vector<int32_t> Tokens;
    auto *Data = Y.data<int32_t>();
    for (int Idx = 0; Idx < static_cast<int>(Y.size()); Idx++) {
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
    auto [NY, NKVCache] = Model->nextGenerate(Y, 0.1, KVCache);
    Y = NY, KVCache = NKVCache;
  }
  const auto End{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> ElapsedSeconds{End - Start};
  std::cout << "Elapsed time: " << ElapsedSeconds.count() << "s. "
            << "TPS: " << TokenList.size() / ElapsedSeconds.count()
            << std::endl;
  return 0;
}