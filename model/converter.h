#pragma once
#include "base.h"
#include "mlx/mlx.h"
#include<cstring>

#define replace(Str, From, To) Str.replace(Str.find(From), strlen(From), To)

std::unordered_map<std::string, mx::array>
weightsToMlx(std::string WeightPath, mx::StreamOrDevice Device);

std::unordered_map<std::string, mx::array>
llamaToMlxllm(std::string WeightPath, mx::StreamOrDevice Device);