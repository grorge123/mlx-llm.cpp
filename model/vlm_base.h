#pragma once
#include "base.h"
#include <mlx/array.h>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace vlm {

class BaseCache {
public:
  int Offset = 0;

  virtual ~BaseCache() = default;

  virtual std::tuple<mx::array, mx::array>
  updateAndFetch(const mx::array &NewKeys, const mx::array &NewValues) = 0;

  virtual std::vector<mx::array> getState() const;
  virtual void setState(const std::vector<mx::array> &State);

  virtual std::string getMetaState() const;
  virtual void setMetaState(const std::string &Value);

  virtual bool isTrimmable() const;
  virtual int trim(int N);
};

class KVCache : public BaseCache {
public:
  int NKVHeads;
  int KHeadDim;
  int VHeadDim;
  mx::array Keys = mx::array({});
  mx::array Values = mx::array({});
  int Offset;
  int Step;

  KVCache(int HeadDim, int NKVHeads, int Step = 256);
  KVCache(std::pair<int, int> HeadDims, int NKVHeads, int Step = 256);
  virtual std::tuple<mx::array, mx::array>
  updateAndFetch(const mx::array &NewKeys, const mx::array &NewValues);

  std::tuple<mx::array, mx::array> fetch() const;

  void update(const mx::array &NewKeys, const mx::array &NewValues);

  std::vector<mx::array> getState() const override;
  void setState(const std::vector<mx::array> &State) override;

  bool isTrimmable() const override;
  int trim(int N) override;
};

class RotatingKVCache : public KVCache {
public:
  int Keep;
  int MaxSize;
  int Idx;

  RotatingKVCache(int MaxSize = -1, int Keep = 0, int Step = 256);

  std::tuple<mx::array, mx::array>
  updateAndFetch(const mx::array &NewKeys, const mx::array &NewValues) override;

  std::tuple<mx::array, mx::array> updateInPlace(const mx::array &NewKeys,
                                                 const mx::array &NewValues);

  std::tuple<mx::array, mx::array> updateConcat(const mx::array &NewKeys,
                                                const mx::array &NewValues);

  mx::array trim(int TrimSize, const mx::array &V,
                 std::optional<mx::array> Append = std::nullopt);

  mx::array temporalOrder(const mx::array &V);

  std::string getMetaState() const override;
  void setMetaState(const std::string &Value) override;

  bool isTrimmable() const override;
  int trim(int N) override;
};

mx::array createAttentionMask(
    mx::array H, std::optional<std::vector<vlm::BaseCache *>> = std::nullopt);

mx::array createAdditiveCausalMask(int N, int Offset = 0);
} // namespace vlm
