#include "model/converter.h"
#include "model/mllama/mllama.h"
int main() {
  std::string ModelPath = "../../Llama-3.2-11B-Vision-Instruct-4bit/";
  auto Model = mllama::Model::fromPretrained(ModelPath, {64, 4});
  return 0;
}