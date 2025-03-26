#include "vlm_base.h"
#include "base.h"
#include <mlx/array.h>
#include <mlx/ops.h>
#include <stdexcept>
#include <vector>

namespace vlm {

// BaseCache implementation
std::vector<mx::array> BaseCache::getState() const { return {}; }

void BaseCache::setState(const std::vector<mx::array> &State) {
  if (!State.empty()) {
    throw std::runtime_error("This cache has no state but a state was set.");
  }
}

std::string BaseCache::getMetaState() const { return ""; }

void BaseCache::setMetaState(const std::string &Value) {
  if (!Value.empty()) {
    throw std::runtime_error(
        "This cache has no meta_state but a meta_state was set.");
  }
}

bool BaseCache::isTrimmable() const { return false; }

int BaseCache::trim(int N) { return 0; }

// KVCache implementation
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

std::vector<mx::array> KVCache::getState() const {
  if (Offset == Keys.shape()[2]) {
    return {Keys, Values};
  }
  mx::array Indices = mx::arange(Offset);
  mx::array KeysSlice = take(Keys, Indices, -2);
  mx::array ValuesSlice = take(Values, Indices, -2);
  return {KeysSlice, ValuesSlice};
}

void KVCache::setState(const std::vector<mx::array> &State) {
  if (State.size() != 2) {
    throw std::runtime_error("KVCache state must contain exactly two arrays");
  }
  Keys = State[0];
  Values = State[1];
  Offset = Keys.shape()[2];
}

bool KVCache::isTrimmable() const { return true; }

int KVCache::trim(int N) {
  N = std::min(Offset, N);
  Offset -= N;
  return N;
}

// RotatingKVCache implementation
RotatingKVCache::RotatingKVCache(int MaxSize, int Keep, int Step)
    : KVCache(0, 0, Step), Keep(Keep), MaxSize(MaxSize), Idx(0) {}

mx::array RotatingKVCache::trim(int TrimSize, const mx::array &V,
                                std::optional<mx::array> Append) {
  std::vector<mx::array> ToCat;

  if (TrimSize > 0) {
    mx::array KeepIndices = mx::arange(Keep);
    mx::array TrimIndices = mx::arange(TrimSize + Keep, V.shape()[2]);
    mx::array KeepPart = take(V, KeepIndices, -2);
    mx::array TrimPart = take(V, TrimIndices, -2);
    ToCat = {KeepPart, TrimPart};
  } else {
    ToCat = {V};
  }

  if (Append.has_value()) {
    ToCat.push_back(Append.value());
  }

  return mx::concatenate(ToCat, 2);
}

mx::array RotatingKVCache::temporalOrder(const mx::array &V) {
  if (Idx == V.shape()[2]) {
    return V;
  }
  if (Idx < Offset) {
    mx::array KeepIndices = mx::arange(Keep);
    mx::array IdxToEndIndices = mx::arange(Idx, V.shape()[2]);
    mx::array KeepToIdxIndices = mx::arange(Keep, Idx);

    mx::array KeepPart = take(V, KeepIndices, -2);
    mx::array IdxToEndPart = take(V, IdxToEndIndices, -2);
    mx::array KeepToIdxPart = take(V, KeepToIdxIndices, -2);

    return mx::concatenate({KeepPart, IdxToEndPart, KeepToIdxPart}, 2);
  }
  mx::array IdxIndices = mx::arange(Idx);
  return take(V, IdxIndices, -2);
}

std::tuple<mx::array, mx::array>
RotatingKVCache::updateConcat(const mx::array &NewKeys,
                              const mx::array &NewValues) {
  if (Keys.size() == 0) {
    Keys = NewKeys;
    Values = NewValues;
  } else {
    // Put the keys/values in temporal order to preserve context
    Keys = temporalOrder(Keys);
    Values = temporalOrder(Values);

    // The largest size is MaxSize + S to ensure every token gets at least
    // MaxSize context
    int TrimSize = Idx - MaxSize;
    Keys = trim(TrimSize, Keys, NewKeys);
    Values = trim(TrimSize, Values, NewValues);
  }

  Offset += NewKeys.shape()[2];
  Idx = Keys.shape()[2];
  return {Keys, Values};
}

std::tuple<mx::array, mx::array>
RotatingKVCache::updateInPlace(const mx::array &NewKeys,
                               const mx::array &NewValues) {
  // May not have hit the max size yet, so potentially keep growing the cache
  std::vector<int> KeysShape = NewKeys.shape();
  int B = KeysShape[0];
  int NKVHeads = KeysShape[1];
  int S = KeysShape[2];
  int KHeadDim = KeysShape[3];
  int VHeadDim = NewValues.shape()[3];

  int Prev = Offset;
  if (Keys.size() == 0 ||
      (Prev >= Keys.shape()[2] && Keys.shape()[2] < MaxSize)) {
    int NewSize = std::min(Step, MaxSize - Prev);
    std::vector<int> KShape = {B, NKVHeads, NewSize, KHeadDim};
    std::vector<int> VShape = {B, NKVHeads, NewSize, VHeadDim};

    mx::array NewK = mx::zeros(KShape, NewKeys.dtype());
    mx::array NewV = mx::zeros(VShape, NewValues.dtype());

    if (Keys.size() != 0) {
      Keys = mx::concatenate({Keys, NewK}, 2);
      Values = mx::concatenate({Values, NewV}, 2);
    } else {
      Keys = NewK;
      Values = NewV;
    }
    Idx = Prev;
  }

  // Trim if needed
  int TrimSize = Keys.shape()[2] - MaxSize;
  if (TrimSize > 0) {
    Keys = trim(TrimSize, Keys);
    Values = trim(TrimSize, Values);
    Idx = MaxSize;
  }

  // Rotate
  if (Idx == MaxSize) {
    Idx = Keep;
  }

  // Assign
  std::vector<int> Start(4, 0);
  std::vector<int> End = {B, NKVHeads, Idx + S, KHeadDim};
  std::vector<int> Stride(4, 1);
  Start[2] = Idx;

  mx::slice_update(Keys, NewKeys, Start, End, Stride);

  End[3] = VHeadDim;
  mx::slice_update(Values, NewValues, Start, End, Stride);

  Offset += S;
  Idx += S;

  // If the buffer is not full, slice off the end
  if (Offset < MaxSize) {
    mx::array OffsetIndices = mx::arange(Offset);
    mx::array KeysSlice = take(Keys, OffsetIndices, -2);
    mx::array ValuesSlice = take(Values, OffsetIndices, -2);
    return {KeysSlice, ValuesSlice};
  }

  return {Keys, Values};
}

std::tuple<mx::array, mx::array>
RotatingKVCache::updateAndFetch(const mx::array &NewKeys,
                                const mx::array &NewValues) {
  if (NewKeys.shape()[2] == 1) {
    return updateInPlace(NewKeys, NewValues);
  }
  return updateConcat(NewKeys, NewValues);
}

std::string RotatingKVCache::getMetaState() const {
  return std::to_string(Keep) + "," + std::to_string(MaxSize) + "," +
         std::to_string(Step) + "," + std::to_string(Offset) + "," +
         std::to_string(Idx);
}

void RotatingKVCache::setMetaState(const std::string &Value) {
  std::stringstream SS(Value);
  std::string Item;
  std::vector<int> Values;

  while (std::getline(SS, Item, ',')) {
    Values.push_back(std::stoi(Item));
  }

  if (Values.size() == 5) {
    Keep = Values[0];
    MaxSize = Values[1];
    Step = Values[2];
    Offset = Values[3];
    Idx = Values[4];
  } else {
    throw std::runtime_error("Invalid meta state format");
  }
}

bool RotatingKVCache::isTrimmable() const { return Offset < MaxSize; }

int RotatingKVCache::trim(int N) {
  N = std::min(Offset, N);
  Offset -= N;
  Idx -= N;
  return N;
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

mx::array
createAttentionMask(mx::array H,
                    std::optional<std::vector<vlm::BaseCache *>> Cache) {
  int T = H.shape()[1];
  mx::array Mask = mx::array({});
  if (T > 1) {
    int Offset = 0;
    if (Cache.has_value() && Cache.value().size() > 0 &&
        Cache.value()[0] != nullptr) {
      auto *C = Cache.value()[0];
      auto *RotCache = dynamic_cast<RotatingKVCache *>(C);
      if (RotCache) {
        Offset = std::min(RotCache->MaxSize - 1, RotCache->Offset);
      } else {
        Offset = C->Offset;
      }
    }
    Mask = createAdditiveCausalMask(T, Offset);
    Mask = mx::astype(Mask, H.dtype());
  }
  return Mask;
}

} // namespace vlm