#include "base.h"
#include "mlx/mlx.h"
#include "model/converter.h"
#include "model/transformer.h"
#include "prompt/llama.h"
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
enum AnserSataus{
  STOP,
  WAIT,
  GO,
};
AnserSataus answerSataus(std::string Text, std::string End){
  if(Text.ends_with(End)){
    return STOP;
  }
  for(int Idx = 1 ; Idx < End.size() ; Idx++){
    if(Text.ends_with(End.substr(0, Idx))){
      return WAIT;
    }
  }
  return GO;
}
int main() {
  auto Tok = Tokenizer::FromBlobJSON(LoadBytesFromFile("../tokenizer.json"));
  // const std::string Prompt = "What is the capital of Canada?";
  const int MaxToken = 512;
  // const std::vector<int> Ids = Tok->Encode(Prompt);
  // const mx::array Token =
  //     mx::array(Ids.data(), {static_cast<int>(Ids.size())}, mx::int32);
  mx::array Token = mx::array({{1, 23, 35, 48, 87, 62}, {6}});
  std::cout << "Create Model...\n";
  // llama2-7b
  //  const int VocabSize = 32000;
  //  const float NormEps = 1e-5;
  //  const float RopeTheta = 10000.0;
  //  const bool RopeTraditional = false;
  //  auto Model = Transformer(4096, std::vector<int>{11008}, VocabSize, 32,
  //                           std::vector<int>{32}, std::vector<int>{32},
  //                           NormEps, RopeTheta, RopeTraditional);
  const int VocabSize = 32000;
  const float NormEps = 1e-5;
  const float RopeTheta = 10000.0;
  const bool RopeTraditional = false;
  auto Model = Transformer(2048, std::vector<int>{5632}, VocabSize, 22,
                           std::vector<int>{32}, std::vector<int>{4}, NormEps,
                           {}, RopeTraditional, RopeTheta);
  std::cout << "Load Model...\n";
  // Model.update(llamaToMlxllm("../llama2-7b"));
  Model.update(llamaToMlxllm("../tiny"));
  std::cout << "Start generate...\n";
  const TinyLLaMAPrompt Prmopt;
  const std::vector<int> Ids = Tok->Encode("How are you?");
  Token =
      mx::array(Ids.data(), {static_cast<int>(Ids.size())}, mx::int32);
  std::vector<int32_t> TokenList;
  std::string Answer;
  int Skip = 0;
  int TokenCount = 0;
  auto [Y, KVCache] = Model.generate(Token, 0.1);
  while (true) {
    TokenCount++;
    if(TokenCount > MaxToken){
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
    if(Status == STOP){
      break;
    }
    if(Status == GO){
      std::cout << Answer.substr(Skip) << std::flush;
      Skip = Answer.size();
    }
    auto [NY, NKVCache] = Model.nextGenerate(Y, 0.1, KVCache);
    Y = NY, KVCache = NKVCache;
  }
  return 0;
}