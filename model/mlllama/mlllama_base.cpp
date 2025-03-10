#include "mlllama_base.h"
#include "base.h"
#include <mlx/array.h>
#include <mlx/ops.h>
#include <vector>

namespace vlm {

KVCache::KVCache(int HeadDim, int NKVHeads, int Step)
    : NKVHeads(NKVHeads), KHeadDim(HeadDim), VHeadDim(HeadDim), Offset(0),
      Step(Step) {}

KVCache::KVCache(std::pair<int, int> HeadDims, int NKVHeads, int Step)
    : NKVHeads(NKVHeads), KHeadDim(HeadDims.first), VHeadDim(HeadDims.second),
      Offset(0), Step(Step) {}

std::tuple<mx::array, mx::array>
KVCache::updateAndFetch(const mx::array &NewKeys, const mx::array &NewValues) {
  update(NewKeys, NewValues);
  return fetch();
}

std::tuple<mx::array, mx::array> KVCache::fetch() const {
  // self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
  mx::array Indices = mx::arange(Offset);
  mx::array KeysSlice = take(Keys, Indices, -2);
  mx::array ValuesSlice = take(Values, Indices, -2);
  return {KeysSlice, ValuesSlice};
}

void KVCache::update(const mx::array &NewKeys, const mx::array &NewValues) {
  int Prev = Offset;
  std::vector<int> NewShape = NewKeys.shape();
  int NewLen = NewShape[2];

  if (Keys.size() == 0 || (Prev + NewLen) > Keys.shape()[2]) {
    int NSteps = (Step + NewLen - 1) / Step;
    int NewCapacity = NSteps * Step;
    std::vector<int> KShape = {1, NKVHeads, NewCapacity, KHeadDim};
    std::vector<int> VShape = {1, NKVHeads, NewCapacity, VHeadDim};
    mx::array NewK = mx::zeros(KShape, NewKeys.dtype());
    mx::array NewV = mx::zeros(VShape, NewValues.dtype());
    if (Keys.size() != 0) {
      if (Prev % Step != 0) {
        mx::array Indices = mx::arange(Prev);
        Keys = take(Keys, Indices, -2);
        Values = take(Values, Indices, -2);
      }
      Keys = mx::concatenate({Keys, NewK}, 2);
      Values = mx::concatenate({Values, NewV}, 2);
    } else {
      Keys = NewK;
      Values = NewV;
    }
  }

  Offset += NewLen;
  // self.keys[..., prev : self.offset, :] = keys
  // self.values[..., prev : self.offset, :] = values
  auto End = NewKeys.shape();
  std::vector<int> Start(End.size(), 0);
  std::vector<int> Stride(End.size(), 1);
  Start[End.size() - 2] = Prev;
  End[End.size() - 2] = Offset;
  mx::slice_update(Keys, NewKeys, Start, End, Stride);
  mx::slice_update(Values, NewValues, Start, End, Stride);
}

mx::array createAdditiveCausalMask(int N, int Offset) {
  auto Rinds = mx::arange(Offset + N);
  mx::array Linds = mx::array({});
  if (Offset) {
    Linds = mx::arange(Offset, Offset + N);
  } else {
    Linds = Rinds;
  }
  // mask = linds[:, None] < rinds[None]
  return mx::less(mx::expand_dims(Linds, 1), mx::expand_dims(Rinds, 0));
}

mx::array createAttentionMask(mx::array H, std::optional<mx::array> Cache) {
  int T = H.shape()[1];
  mx::array Mask = mx::array({});
  if (T > 1) {
    int Offset = 0;
    if (Cache.has_value()) {
      assumingUnreachable();
    }
    Mask = createAdditiveCausalMask(T, Offset);
  }
  return Mask;
}

} // namespace vlm