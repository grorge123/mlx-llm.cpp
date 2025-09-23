#pragma once

#include <mlx/array.h>
#include <mlx/io.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace mx = mlx::core;

using LoadOutputTypes =
    std::variant<mx::array, std::unordered_map<std::string, mx::array>,
                 std::pair<std::unordered_map<std::string, mx::array>,
                           std::unordered_map<std::string, std::string>>,
                 std::pair<std::unordered_map<std::string, mx::array>,
                           std::unordered_map<std::string, mx::GGUFMetaData>>>;

std::pair<std::unordered_map<std::string, mx::array>,
          std::unordered_map<std::string, std::string>>
mlxLoadSafetensorHelper(const std::string &File, mx::StreamOrDevice S = {});

mx::GGUFLoad mlxLoadGgufHelper(const std::string &File,
                               mx::StreamOrDevice S = {});

std::unordered_map<std::string, mx::array>
mlxLoadNpzHelper(const std::string &File, mx::StreamOrDevice S = {});

mx::array mlxLoadNpyHelper(const std::string &File, mx::StreamOrDevice S = {});

LoadOutputTypes mlxLoadHelper(const std::string &File,
                              std::optional<std::string> Format = std::nullopt,
                              bool ReturnMetadata = false,
                              mx::StreamOrDevice S = {});
