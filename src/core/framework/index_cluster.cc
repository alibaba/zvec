// namespace aitheta2
// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <limits>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/core/framework/index_bundle.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include "utility/paged_build_buffer.h"

namespace zvec {
namespace core {

static const std::string CLUSTER_CENTROIDS_FEATURES_NAME =
    "IndexCluster.Centroids.Features";
static const std::string CLUSTER_CENTROIDS_INDEXES_NAME =
    "IndexCluster.Centroids.Indexes";

/*! Item Centroid Format
 */
struct ItemCentroidFormat {
  uint32_t parent;
  uint32_t reserved0_;
  uint64_t follows;
  double score;
  uint64_t reserved1_;
};

namespace {
using CentroidReader = std::function<int(size_t, void *, size_t)>;
class LoadedCentroidFeatures : public IndexCluster::FeatureStorage {
 public:
  PagedBuildBuffer data;
  int read(size_t offset, void *out, size_t bytes) const override {
    return data.read(offset, out, bytes);
  }
};

int ReadCentroids(const IndexMeta &meta, size_t feature_bytes,
                  size_t index_bytes, const CentroidReader &read_features,
                  const CentroidReader &read_indexes,
                  IndexCluster::CentroidList *out) {
  if (!out || meta.element_size() == 0) return IndexError_InvalidArgument;
  const size_t stride = meta.element_size();
  if (feature_bytes % stride || index_bytes % sizeof(ItemCentroidFormat))
    return IndexError_InvalidLength;
  const size_t count = feature_bytes / stride;
  if (count != index_bytes / sizeof(ItemCentroidFormat))
    return IndexError_InvalidLength;
  if (count == 0 || count >= UINT32_MAX) return IndexError_InvalidFormat;

  std::vector<uint32_t, ailego::BudgetedAllocator<uint32_t>> subitems(count +
                                                                      1);
  ailego::BudgetedVector<ItemCentroidFormat> records(
      std::min(size_t{128}, count));
  auto visit_records = [&](auto &&visit) {
    for (size_t first = 0; first < count; first += records.size()) {
      const size_t n = std::min(records.size(), count - first);
      int ret = read_indexes(first * sizeof(records[0]), records.data(),
                             n * sizeof(records[0]));
      if (ret != 0) return ret;
      for (size_t j = 0; j < n; ++j) {
        ret = visit(first + j, records[j]);
        if (ret != 0) return ret;
      }
    }
    return 0;
  };
  int ret =
      visit_records([&](size_t i, const ItemCentroidFormat &record) -> int {
        if (record.parent != UINT32_MAX && record.parent >= i)
          return IndexError_InvalidFormat;
        ++subitems[record.parent == UINT32_MAX ? 0 : record.parent + 1];
        return 0;
      });
  if (ret != 0) return ret;

  auto features = std::make_shared<LoadedCentroidFeatures>();
  ret = features->data.init(feature_bytes, 64 * 1024);
  if (ret != 0) return ret;
  ailego::BudgetedMappedBuffer buffer(
      std::min(size_t{64 * 1024}, feature_bytes));
  for (size_t first = 0; first < feature_bytes; first += buffer.size()) {
    const size_t n = std::min(buffer.size(), feature_bytes - first);
    ret = read_features(first, buffer.data(), n);
    if (ret == 0) ret = features->data.write(first, buffer.data(), n);
    if (ret != 0) return ret;
  }

  std::vector<IndexCluster::Centroid *,
              ailego::BudgetedAllocator<IndexCluster::Centroid *>>
      items;
  items.reserve(count);
  IndexCluster::CentroidList candidate;
  // Charge list capacity before allocation; individual nodes carry that
  // charge after construction, including across subtree moves/copies.
  ailego::MemoryReservation directory;
  directory.add_array(count, IndexCluster::Centroid::loaded_metadata_bytes());
  candidate.reserve(subitems.front());
  ret = visit_records([&](size_t i, const ItemCentroidFormat &record) {
    auto *current = &candidate;
    if (record.parent != UINT32_MAX && record.parent >= items.size())
      return int(IndexError_InvalidFormat);
    if (record.parent != UINT32_MAX)
      current = items[record.parent]->mutable_subitems();
    if (current->size() == current->capacity())
      return int(IndexError_InvalidFormat);
    current->emplace_back();
    auto &centroid = current->back();
    centroid.adopt_metadata(
        directory.take(IndexCluster::Centroid::loaded_metadata_bytes()));
    centroid.set_feature_storage(features, i * stride, stride);
    centroid.set_follows(static_cast<size_t>(record.follows));
    centroid.set_score(record.score);
    centroid.mutable_subitems()->reserve(subitems[i + 1]);
    items.push_back(&centroid);
    return 0;
  });
  if (ret != 0) return ret;
  out->swap(candidate);
  return 0;
}
}  // namespace

int IndexCluster::Deserialize(const IndexMeta &meta,
                              IndexBundle::Pointer bundle,
                              CentroidList *cents) try {
  if (!bundle || !cents) return IndexError_InvalidArgument;
  const auto features = bundle->get(CLUSTER_CENTROIDS_FEATURES_NAME);
  const auto indexes = bundle->get(CLUSTER_CENTROIDS_INDEXES_NAME);
  if (!features.is_valid() || !indexes.is_valid())
    return IndexError_InvalidArgument;
  auto reader = [](ailego::BlobWrap blob) -> CentroidReader {
    return [blob](size_t offset, void *out, size_t bytes) -> int {
      if (offset > blob.size() || bytes > blob.size() - offset)
        return IndexError_ReadData;
      std::memcpy(out, static_cast<const char *>(blob.buffer()) + offset,
                  bytes);
      return 0;
    };
  };
  return ReadCentroids(meta, features.size(), indexes.size(), reader(features),
                       reader(indexes), cents);
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int IndexCluster::Load(const IndexMeta &meta, IndexStorage::Pointer storage,
                       CentroidList *cents) try {
  if (!storage || !cents) return IndexError_InvalidArgument;
  auto features = storage->get(CLUSTER_CENTROIDS_FEATURES_NAME);
  auto indexes = storage->get(CLUSTER_CENTROIDS_INDEXES_NAME);
  if (!features || !indexes) return IndexError_InvalidFormat;
  auto reader = [](IndexStorage::Segment::Pointer segment) -> CentroidReader {
    return [segment](size_t offset, void *out, size_t bytes) {
      return segment->fetch(offset, out, bytes) == bytes ? 0
                                                         : IndexError_ReadData;
    };
  };
  return ReadCentroids(meta, features->data_size(), indexes->data_size(),
                       reader(features), reader(indexes), cents);
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

namespace {
template <typename Visitor>
int VisitCentroids(const IndexCluster::CentroidList &cents, size_t *ordinal,
                   Visitor &visit, uint32_t parent = UINT32_MAX) {
  for (const auto &cent : cents) {
    if (*ordinal >= UINT32_MAX) return IndexError_InvalidLength;
    const auto id = static_cast<uint32_t>((*ordinal)++);
    int ret = visit(cent, id, parent);
    if (ret != 0) return ret;
    ret = VisitCentroids(cent.subitems(), ordinal, visit, id);
    if (ret != 0) return ret;
  }
  return 0;
}

int CountCentroids(const IndexMeta &meta,
                   const IndexCluster::CentroidList &cents, size_t *count) {
  *count = 0;
  auto check = [&](const auto &cent, uint32_t, uint32_t) {
    return cent.size() == meta.element_size() ? 0 : IndexError_Mismatch;
  };
  int ret = VisitCentroids(cents, count, check);
  if (ret != 0) return ret;
  if (meta.element_size() == 0 ||
      *count > std::numeric_limits<size_t>::max() / meta.element_size() ||
      *count > std::numeric_limits<size_t>::max() / sizeof(ItemCentroidFormat))
    return IndexError_InvalidLength;
  return 0;
}

// The legacy bundle API requires contiguous views. Keep their allocation
// charged for the entire bundle lifetime, without a second string copy.
class SerializedCentroidBundle : public TrivialIndexBundle {
 public:
  std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>> features;
  std::vector<ItemCentroidFormat, ailego::BudgetedAllocator<ItemCentroidFormat>>
      indexes;
};
}  // namespace

int IndexCluster::Serialize(const IndexMeta &meta, const CentroidList &cents,
                            IndexBundle::Pointer *out) try {
  if (!out) return IndexError_InvalidArgument;
  size_t count = 0;
  int ret = CountCentroids(meta, cents, &count);
  if (ret != 0) return ret;
  auto bundle = std::make_shared<SerializedCentroidBundle>();
  bundle->features.resize(count * meta.element_size());
  bundle->indexes.resize(count);
  auto serialize = [&](const auto &cent, uint32_t id, uint32_t parent) {
    bundle->indexes[id] = {parent, 0, cent.follows(), cent.score(), 0};
    return cent.read_feature(bundle->features.data() + id * meta.element_size(),
                             meta.element_size());
  };
  size_t ordinal = 0;
  ret = VisitCentroids(cents, &ordinal, serialize);
  if (ret != 0) return ret;
  bundle->set(CLUSTER_CENTROIDS_FEATURES_NAME, bundle->features.data(),
              bundle->features.size());
  bundle->set(CLUSTER_CENTROIDS_INDEXES_NAME, bundle->indexes.data(),
              bundle->indexes.size() * sizeof(ItemCentroidFormat));
  *out = std::move(bundle);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int IndexCluster::Dump(const IndexMeta &meta, const CentroidList &cents,
                       const IndexDumper::Pointer &dumper) try {
  if (!dumper) return IndexError_InvalidArgument;
  size_t count = 0;
  int ret = CountCentroids(meta, cents, &count);
  if (ret != 0) return ret;
  ret = dumper->append(CLUSTER_CENTROIDS_FEATURES_NAME,
                       count * meta.element_size(), 0, 0);
  if (ret != 0) return ret;
  std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>> row;
  auto write_feature = [&](const auto &cent, uint32_t, uint32_t) {
    const void *data = nullptr;
    if (cent.feature_stored()) {
      row.resize(meta.element_size());
      int read_ret = cent.read_feature(row.data(), row.size());
      if (read_ret != 0) return read_ret;
      data = row.data();
    } else {
      data = cent.feature();
    }
    return dumper->write(data, meta.element_size()) == meta.element_size()
               ? 0
               : IndexError_WriteData;
  };
  size_t ordinal = 0;
  ret = VisitCentroids(cents, &ordinal, write_feature);
  if (ret != 0) return ret;
  ret = dumper->append(CLUSTER_CENTROIDS_INDEXES_NAME,
                       count * sizeof(ItemCentroidFormat), 0, 0);
  if (ret != 0) return ret;
  std::array<ItemCentroidFormat, 128> items{};
  size_t filled = 0;
  auto flush = [&]() -> int {
    const size_t bytes = filled * sizeof(ItemCentroidFormat);
    if (bytes && dumper->write(items.data(), bytes) != bytes)
      return IndexError_WriteData;
    filled = 0;
    return 0;
  };
  auto write_index = [&](const auto &cent, uint32_t, uint32_t parent) {
    items[filled++] = {parent, 0, cent.follows(), cent.score(), 0};
    return filled == items.size() ? flush() : 0;
  };
  ordinal = 0;
  ret = VisitCentroids(cents, &ordinal, write_index);
  return ret != 0 ? ret : flush();
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

}  // namespace core
}  // namespace zvec
