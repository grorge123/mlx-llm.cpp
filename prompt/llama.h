#pragma once

#include <cstring>
#include <iostream>

class TinyLLaMAPrompt {
public:
  std::string SystemStart = "<|system|>";
  std::string User = "<|user|>";
  std::string Assistant = "<|assistant|>";
  std::string TextEnd = "</s>";
  std::string prepare(std::string Prompt);
};