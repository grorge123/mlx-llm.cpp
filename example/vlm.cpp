#include "model/converter.h"
#include "model/mllama/mllama.h"
int main() {
  std::string ModelPath = "../../Llama-3.2-11B-Vision-Instruct-4bit/";
  mllama::Model Model = mllama::Model::fromPretrained(ModelPath);
  return 0;
}