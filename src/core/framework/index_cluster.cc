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

#include <zvec/core/framework/index_bundle.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>

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

static inline bool GatherSubitemsCount(const ItemCentroidFormat *format,
                                       size_t count,
                                       std::vector<uint32_t> *out) {
  out->resize(count + 1);

  for (const ItemCentroidFormat *it = format, *end = format + count; it != end;
       ++it) {
    uint32_t parent = it->parent + 1;
    if (parent > count) {
      return false;
    }
    (*out)[parent] += 1;
  }
  return (out->front() != 0);
}

int IndexCluster::Deserialize(const IndexMeta &meta,
                              IndexBundle::Pointer bundle,
                              CentroidList *cents) {
  if (!bundle || !cents) {
    return IndexError_InvalidArgument;
  }

  ailego::BlobWrap features = bundle->get(CLUSTER_CENTROIDS_FEATURES_NAME);
  ailego::BlobWrap indexes = bundle->get(CLUSTER_CENTROIDS_INDEXES_NAME);

  if (!features.is_valid() || !indexes.is_valid()) {
    return IndexError_InvalidArgument;
  }

  if (features.size() % meta.element_size() != 0 ||
      indexes.size() % sizeof(ItemCentroidFormat) != 0) {
    return IndexError_InvalidLength;
  }

  size_t count = features.size() / meta.element_size();
  if (indexes.size() / sizeof(ItemCentroidFormat) != count) {
    return IndexError_InvalidLength;
  }

  const ItemCentroidFormat *format =
      reinterpret_cast<const ItemCentroidFormat *>(indexes.buffer());
  std::vector<uint32_t> subitems;

  if (!GatherSubitemsCount(format, count, &subitems)) {
    return IndexError_InvalidFormat;
  }

  std::vector<Centroid *> items;
  items.reserve(count);
  cents->clear();
  cents->reserve(subitems.front());

  const uint8_t *feat = reinterpret_cast<const uint8_t *>(features.buffer());
  size_t feat_size = meta.element_size();

  for (size_t i = 0; i < count; ++i, ++format, feat += feat_size) {
    CentroidList *current = cents;

    if (format->parent != static_cast<uint32_t>(-1)) {
      if (format->parent >= items.size()) {
        return IndexError_InvalidFormat;
      }
      current = items[format->parent]->mutable_subitems();
    }
    current->emplace_back(feat, feat_size);

    // Update information
    Centroid *last_one = &(current->back());
    last_one->set_follows(static_cast<size_t>(format->follows));
    last_one->set_score(format->score);
    last_one->mutable_subitems()->reserve(subitems[i + 1]);
    items.push_back(last_one);
  }
  return 0;
}

static void SerializeToBuffers(const IndexCluster::CentroidList &cents,
                               std::string *features, std::string *indexes) {
  uint32_t parent =
      static_cast<uint32_t>(indexes->size() / sizeof(ItemCentroidFormat)) - 1;

  for (const auto &it : cents) {
    ItemCentroidFormat format{parent, 0, it.follows(), it.score(), 0};
    indexes->append(reinterpret_cast<const char *>(&format), sizeof(format));
    features->append(reinterpret_cast<const char *>(it.feature()), it.size());

    if (!it.subitems().empty()) {
      SerializeToBuffers(it.subitems(), features, indexes);
    }
  }
}

int IndexCluster::Serialize(const IndexMeta &meta, const CentroidList &cents,
                            IndexBundle::Pointer *out) {
  size_t cents_total = cents.size();

  // Check the centroids
  for (const auto &it : cents) {
    if (!it.is_matched(meta)) {
      return IndexError_Mismatch;
    }
    cents_total += it.subcount();
  }

  std::string features, indexes;
  features.reserve(cents_total * meta.element_size());
  indexes.reserve(cents_total * sizeof(ItemCentroidFormat));
  SerializeToBuffers(cents, &features, &indexes);

  std::shared_ptr<MemoryIndexBundle> bundle =
      std::make_shared<MemoryIndexBundle>();

  bundle->set(CLUSTER_CENTROIDS_FEATURES_NAME, std::move(features));
  bundle->set(CLUSTER_CENTROIDS_INDEXES_NAME, std::move(indexes));
  *out = std::move(bundle);

  return 0;
}

}  // namespace core
}  // namespace zvec
