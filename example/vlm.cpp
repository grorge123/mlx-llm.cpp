#include "model/converter.h"
#include "model/gemma3/gemma3.h"
#include "model/mllama/mllama.h"
#include <string>
int main() {
  // std::string ModelPath = "../../Llama-3.2-11B-Vision-Instruct-4bit/";
  // auto Model = mllama::Model::fromPretrained(ModelPath, {64, 4});
  std::string ModelPath = "../../gemma-3-4b-it-bf16";
  auto Model = gemma3::Model::fromPretrained(ModelPath);
  return 0;
}