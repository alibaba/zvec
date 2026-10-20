// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <map>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_cluster.h>
#include "ailego/pattern/defer.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class RecordingDumper : public IndexDumper {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  int append(const std::string &id, size_t bytes, size_t, uint32_t) override {
    current = id;
    sizes[id] = bytes;
    return append_status;
  }
  size_t write(const void *data, size_t bytes) override {
    max_write = std::max(max_write, bytes);
    if (short_write) return 0;
    segments[current].append(static_cast<const char *>(data), bytes);
    return bytes;
  }
  std::map<std::string, std::string> segments;
  std::map<std::string, size_t> sizes;
  std::string current;
  size_t max_write{0};
  int append_status{0};
  bool short_write{false};
};

class StoredRow : public IndexCluster::FeatureStorage {
 public:
  int read(size_t, void *out, size_t bytes) const override {
    ++reads;
    if (status) return status;
    std::memset(out, 7, bytes);
    return 0;
  }
  mutable size_t reads{0};
  int status{0};
};

class CentroidSegment : public IndexStorage::Segment {
 public:
  std::string bytes;
  mutable size_t max_fetch{0};
  bool fail{false};
  size_t data_size() const override {
    return bytes.size();
  }
  size_t capacity() const override {
    return bytes.size();
  }
  size_t padding_size() const override {
    return 0;
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t fetch(size_t offset, void *out, size_t n) const override {
    max_fetch = std::max(max_fetch, n);
    if (fail || offset > bytes.size() || n > bytes.size() - offset) return 0;
    std::memcpy(out, bytes.data() + offset, n);
    return n;
  }
  size_t read(size_t, IndexStorage::MemoryBlock &, size_t) override {
    ADD_FAILURE() << "Centroid load must not request a whole memory block";
    return 0;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<CentroidSegment>(*this);
  }
};

class CentroidStorage : public IndexStorage {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int open(const std::string &, bool) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  int flush() override {
    return 0;
  }
  int append(const std::string &, size_t) override {
    return IndexError_NotImplemented;
  }
  void refresh(uint64_t) override {}
  uint64_t check_point() const override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  bool has(const std::string &id) const override {
    return segments.count(id);
  }
  Segment::Pointer get(const std::string &id, int = -1) override {
    auto it = segments.find(id);
    return it == segments.end() ? nullptr : it->second;
  }
  std::map<std::string, std::shared_ptr<CentroidSegment>> segments;
};

TEST(CentroidSerializationMemory, ReloadUsesBoundedReadsAndOwnedPagedRows) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(0));
  IndexMeta meta(IndexMeta::DT_FP32, 8192);
  std::vector<float> row(meta.dimension(), 3);
  IndexCluster::CentroidList source;
  for (size_t i = 0; i < 512; ++i)
    source.emplace_back(row.data(), meta.element_size());
  auto dumper = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, IndexCluster::Dump(meta, source, dumper));
  IndexCluster::CentroidList().swap(source);
  auto storage = std::make_shared<CentroidStorage>();
  for (auto &segment : dumper->segments) {
    auto data = std::make_shared<CentroidSegment>();
    data->bytes = std::move(segment.second);
    storage->segments.emplace(segment.first, std::move(data));
  }
  const size_t bytes = 512 * meta.element_size();
  const size_t budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(
          bytes / ailego::kVectorPageSize + 8, true) +
      64 * ailego::kVectorPageSize +
      512 * (IndexCluster::Centroid::loaded_metadata_bytes() + 16);
  ASSERT_LT(budget, bytes);
  ASSERT_EQ(0, pool.init(budget));
  IndexCluster::CentroidList loaded;
  ASSERT_EQ(0, IndexCluster::Load(meta, storage, &loaded));
  ASSERT_EQ(512u, loaded.size());
  for (const auto &segment : storage->segments)
    EXPECT_LE(segment.second->max_fetch, 64u * 1024);
  const size_t charge = pool.used();
  EXPECT_GT(charge, 0u);
  // Failed reload must leave the published tree intact.
  storage->segments.at("IndexCluster.Centroids.Indexes")->fail = true;
  EXPECT_EQ(IndexError_ReadData, IndexCluster::Load(meta, storage, &loaded));
  ASSERT_EQ(512u, loaded.size());
  storage.reset();
  std::vector<float> actual(meta.dimension());
  for (const auto &centroid : loaded) {
    EXPECT_TRUE(centroid.feature_stored());
    ASSERT_EQ(0, centroid.read_feature(actual.data(), meta.element_size()));
    EXPECT_EQ(row, actual);
  }
}

TEST(CentroidSerializationMemory, DirectoryAdmissionFailurePreservesOutput) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(0));
  IndexMeta meta(IndexMeta::DT_FP32, 1);
  float feature = 1;
  IndexCluster::CentroidList source(512, IndexCluster::Centroid(&feature, 4));
  IndexBundle::Pointer bundle;
  ASSERT_EQ(0, IndexCluster::Serialize(meta, source, &bundle));
  ASSERT_EQ(0, pool.init(1024));
  IndexCluster::CentroidList result{IndexCluster::Centroid(&feature, 4)};
  EXPECT_EQ(IndexError_NoMemory,
            IndexCluster::Deserialize(meta, bundle, &result));
  ASSERT_EQ(1u, result.size());
  EXPECT_EQ(0u, pool.used());
}

TEST(CentroidSerializationMemory, StreamMatchesLegacyNestedWireLayout) {
  IndexMeta meta(IndexMeta::DT_FP32, 4);
  std::vector<float> row(4, 3);
  IndexCluster::CentroidList cents;
  for (size_t i = 0; i < 3; ++i) {
    cents.emplace_back(row.data(), meta.element_size());
    cents.back().set_follows(i + 5);
    cents.back().set_score(0.25 * i);
    auto *children = cents.back().mutable_subitems();
    children->emplace_back(row.data(), meta.element_size());
    children->back().mutable_subitems()->emplace_back(row.data(),
                                                      meta.element_size());
  }
  IndexBundle::Pointer bundle;
  ASSERT_EQ(0, IndexCluster::Serialize(meta, cents, &bundle));
  auto dumper = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, IndexCluster::Dump(meta, cents, dumper));
  for (const auto &segment : bundle->all()) {
    EXPECT_EQ(segment.second.size(), dumper->sizes.at(segment.first));
    EXPECT_EQ(std::string(static_cast<const char *>(segment.second.buffer()),
                          segment.second.size()),
              dumper->segments.at(segment.first));
  }
  IndexCluster::CentroidList decoded;
  ASSERT_EQ(0, IndexCluster::Deserialize(meta, bundle, &decoded));
  ASSERT_EQ(cents.size(), decoded.size());
  for (size_t i = 0; i < cents.size(); ++i) {
    EXPECT_EQ(cents[i].score(), decoded[i].score());
    EXPECT_EQ(cents[i].follows(), decoded[i].follows());
    EXPECT_EQ(2u, decoded[i].subcount());
  }
}

TEST(CentroidSerializationMemory,
     LowBudgetStreamsStoredRowsWithoutMaterializingBundle) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(8192));
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  auto storage = std::make_shared<StoredRow>();
  IndexCluster::CentroidList cents(512);
  for (auto &cent : cents)
    cent.set_feature_storage(storage, 0, meta.element_size());
  auto sentinel = std::make_shared<TrivialIndexBundle>();
  IndexBundle::Pointer output = sentinel;
  EXPECT_EQ(IndexError_NoMemory, IndexCluster::Serialize(meta, cents, &output));
  EXPECT_EQ(output, sentinel);
  EXPECT_EQ(0u, storage->reads);
  auto dumper = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, IndexCluster::Dump(meta, cents, dumper));
  EXPECT_EQ(cents.size(), storage->reads);
  EXPECT_LE(dumper->max_write, 4096u);
  EXPECT_EQ(0u, pool.used());
  storage->status = IndexError_ReadData;
  EXPECT_EQ(IndexError_ReadData, IndexCluster::Dump(meta, cents, dumper));
  storage->status = 0;
  dumper->short_write = true;
  EXPECT_EQ(IndexError_WriteData, IndexCluster::Dump(meta, cents, dumper));
}

TEST(CentroidSerializationMemory, BundleReservationFollowsSharedOwner) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(65536));
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  std::vector<float> row(64, 1);
  IndexCluster::CentroidList cents;
  for (size_t i = 0; i < 32; ++i)
    cents.emplace_back(row.data(), meta.element_size());
  IndexBundle::Pointer output;
  ASSERT_EQ(0, IndexCluster::Serialize(meta, cents, &output));
  const size_t charged = pool.used();
  EXPECT_GE(charged, cents.size() * (meta.element_size() + 32));
  auto retained = output;
  output.reset();
  EXPECT_EQ(charged, pool.used());
  const size_t input_charge = 32 * (cents.front().buffer().capacity() + 1);
  retained.reset();
  EXPECT_EQ(input_charge, pool.used());
  IndexCluster::CentroidList().swap(cents);
  EXPECT_EQ(0u, pool.used());
}
}  // namespace
}  // namespace zvec::core
