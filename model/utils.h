#pragma once
#include "base.h"
#include <iostream>
#include <string>
#include <vector>

#define strReplace(Str, From, To) Str.replace(Str.find(From), strlen(From), To)

std::vector<std::string> splitString(const std::string &S, char Delim);
std::string joinString(std::vector<std::string> &S, char Delim);
bool endsWith(std::string const &Value, std::string const &Ending);
bool startsWith(std::string const &Value, std::string const &Starting);
void saveWeights(const std::unordered_map<std::string, mx::array> &Weights,
                 const std::string Path);
void saveWeights(const mx::array &Weights, const std::string &Path);

std::string loadBytesFromFile(const std::string &Path);

void fillPlaceholders(std::ostringstream &oss, const std::string &fmt,
                      size_t &pos);

template <typename T> std::string toString(const T &value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

template <typename T> std::string toString(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    oss << toString(vec[i]);
    if (i + 1 < vec.size()) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

template <typename T, typename... Args>
void fillPlaceholders(std::ostringstream &oss, const std::string &fmt,
                      size_t &pos, T &&value, Args &&...args) {
  auto placeholderPos = fmt.find("{}", pos);
  if (placeholderPos == std::string::npos) {
    oss << fmt.substr(pos);
    return;
  }
  oss << fmt.substr(pos, placeholderPos - pos);
  oss << toString(value);
  pos = placeholderPos + 2;
  fillPlaceholders(oss, fmt, pos, std::forward<Args>(args)...);
}

template <typename... Args>
std::string formatStr(const std::string &fmt, Args &&...args) {
  std::ostringstream oss;
  size_t pos = 0;
  fillPlaceholders(oss, fmt, pos, std::forward<Args>(args)...);
  return oss.str();
}

template <typename... Args> void debug(const std::string &fmt, Args &&...args) {
  std::cout << "[DEBUG] " << formatStr(fmt, std::forward<Args>(args)...)
            << std::endl;
}