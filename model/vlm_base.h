#pragma once
#include "base.h"
#include <mlx/array.h>
#include <optional>
#include <tuple>
#include <utility>

namespace vlm {

class KVCache {
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
  std::tuple<mx::array, mx::array> updateAndFetch(const mx::array &NewKeys,
                                                  const mx::array &NewValues);

  std::tuple<mx::array, mx::array> fetch() const;

  void update(const mx::array &NewKeys, const mx::array &NewValues);
};

mx::array
createAttentionMask(mx::array H,
                    std::optional<std::vector<vlm::KVCache *>> = std::nullopt);

mx::array createAdditiveCausalMask(int N, int Offset = 0);
} // namespace vlm
