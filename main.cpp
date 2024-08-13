#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/transformer.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mlx/array.h>
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

int main() {
  auto Tok = Tokenizer::FromBlobJSON(LoadBytesFromFile("../tokenizer.json"));
  const std::string Prompt = "What is the capital of Canada?";
  const int MaxLen = 512;
  const std::vector<int> Ids = Tok->Encode(Prompt);
  const mx::array Token =
      mx::array(Ids.data(), {static_cast<int>(Ids.size())}, mx::int32);
  // const mx::array Token = mx::array({{1,2,3},{3}});
  std::cout << Token << std::endl;
  const mx::StreamOrDevice Device =
      mx::metal::is_available() ? mx::Device::gpu : mx::Device::cpu;
  const int VocabSize = 32000;
  const float NormEps = 1e-5;
  const float RopeTheta = 10000.0;
  const bool RopeTraditional = false;
  std::cout << "Create Model...\n";
  auto Model = Transformer(4096, std::vector<int>{11008}, VocabSize, 32,
                           std::vector<int>{32}, std::vector<int>{32}, NormEps,
                           RopeTheta, RopeTraditional);
  std::cout << "Load Model...\n";
  Model.update(llamaToMlxllm("../llama2-7b", Device));
  std::cout << "Start generate...\n";
  auto [Y, KVCache] = Model.generate(Token, 0.1);
  std::vector<int> TokenList;
  std::string Answer;
  int Skip = 0;
  while (Answer.size() > MaxLen) {
    auto [NY, NKVCache] = Model.nextGenerate(Y, 0.1, KVCache);
    if (Y.shape()[1] > 1) {
      break;
    }
    Y = NY, NKVCache = KVCache;
    auto *Data = Y.data<int>();
    for (int Idx = 0; Idx < Y.size(); Idx++) {
      TokenList.emplace_back(Data[Idx]);
    }
    if (TokenList.back() == Tok->Encode("<|eot_id|>")[0]) {
      break;
    }
    Answer += Tok->Decode(TokenList);
    std::cout << Answer.substr(Skip);
    Skip = Answer.size();
  }
  return 0;
}