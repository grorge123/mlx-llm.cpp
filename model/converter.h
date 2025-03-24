#pragma once
#include "base.h"
#include "mlx/mlx.h"
#include <cstring>

std::unordered_map<std::string, mx::array> weightsToMlx(std::string WeightPath);

std::unordered_map<std::string, mx::array>
llamaToMlxllm(std::string WeightPath);