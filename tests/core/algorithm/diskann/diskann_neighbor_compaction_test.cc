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

#include <cerrno>
#include <cstring>
#include <limits>
#include <thread>
#include <tuple>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include "diskann_algorithm.h"
#include "diskann_builder_entity.h"
#include "diskann_neighbor_buffer.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#else
#include <sys/mman.h>
#endif

namespace zvec {
namespace core {
namespace {

TEST(DiskAnnNeighborBuffer, ShrinkReturnsTailPagesWithoutMovingPrefix) {
  const size_t page = ailego::MemoryHelper::PageSize();
  DiskAnnNeighborBuffer buffer;
  ASSERT_EQ(0, buffer.allocate(page * 4));
  char *base = buffer.data();
  memset(base, 0x5a, buffer.size());
  ASSERT_EQ(0, buffer.shrink(page + 1));
  EXPECT_EQ(page * 2, buffer.size());
  EXPECT_EQ(base, buffer.data());
  for (size_t i = 0; i < buffer.size(); ++i) {
    ASSERT_EQ(0x5a, base[i]);
  }

  // Inspect the released address without touching it.
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION info{};
  ASSERT_NE(0U, ::VirtualQuery(base + page * 2, &info, sizeof(info)));
  EXPECT_EQ(MEM_RESERVE, info.State);
#elif defined(__APPLE__)
  mach_vm_address_t address =
      reinterpret_cast<mach_vm_address_t>(base + page * 2);
  const auto released = address;
  mach_vm_size_t length = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  kern_return_t result = ::mach_vm_region(
      ::mach_task_self(), &address, &length, VM_REGION_BASIC_INFO_64,
      reinterpret_cast<vm_region_info_t>(&info), &info_count, &object);
  if (object != MACH_PORT_NULL)
    ::mach_port_deallocate(::mach_task_self(), object);
  EXPECT_TRUE(result == KERN_INVALID_ADDRESS ||
              (result == KERN_SUCCESS && address > released));
#else
  unsigned char residency = 0;
  errno = 0;
  int result = ::mincore(base + page * 2, page, &residency);
  EXPECT_EQ(-1, result);
  EXPECT_EQ(ENOMEM, errno);
#endif
  EXPECT_EQ(IndexError_InvalidLength, buffer.shrink(page * 3));
  EXPECT_EQ(IndexError_Overflow,
            buffer.allocate(std::numeric_limits<size_t>::max()));
  EXPECT_EQ(base, buffer.data());
  EXPECT_EQ(page * 2, buffer.size());
  ASSERT_EQ(0, buffer.shrink(0));
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0U, buffer.size());
  buffer.clear();
  ASSERT_EQ(0, buffer.allocate(page));
  EXPECT_EQ(0, buffer.data()[0]);
}

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
  int append(const std::string &id, size_t data_size, size_t padding_size,
             uint32_t crc) override {
    segments.emplace_back(id, data_size, padding_size, crc);
    return 0;
  }
  size_t write(const void *data, size_t len) override {
    if (fail_write) return 0;
    if (len != 0) bytes.append(static_cast<const char *>(data), len);
    return len;
  }
  bool fail_write{false};
  std::string bytes;
  std::vector<std::tuple<std::string, size_t, size_t, uint32_t>> segments;
};

class DiskAnnNeighborCompaction : public testing::Test {
 protected:
  IndexMeta meta{IndexMeta::DT_FP32, 16};
  DiskAnnBuilderEntity entity;
  std::shared_ptr<MultiPassIndexHolder<IndexMeta::DT_FP32>> holder;

  void prepare(size_t count, uint32_t degree = 100) {
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(16);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_TRUE(
          holder->emplace(i * 3, ailego::NumericalVector<float>(16, i)));
    }
    ASSERT_EQ(0, entity.init(meta, degree, degree, 0, 2));
    ASSERT_EQ(0, entity.reserve_space(count, holder));
    auto iter = holder->create_iterator();
    for (; iter->is_valid(); iter->next()) {
      ASSERT_EQ(0, entity.add_vector(iter->key(), iter->data()));
    }
    entity.mutable_pq_meta()->chunk_num = 1;
    entity.block_compressed_data().assign(count, 0);
  }

  std::vector<diskann_id_t> neighbors(size_t id) const {
    auto row = entity.get_neighbors(id);
    return {row.second, row.second + row.first};
  }
};

TEST_F(DiskAnnNeighborCompaction, PreservesOverlappingRowsAndReturnsPages) {
  constexpr size_t count = 8192;
  prepare(count);
  std::vector<std::vector<diskann_id_t>> expected(count);
  for (size_t id = 0; id < count; ++id) {
    // Include zero-degree nodes and dense rows to exercise overlapping moves.
    for (size_t j = 0; j < id % 101; ++j) {
      expected[id].push_back((id + j + 1) % count);
    }
    ASSERT_EQ(0, entity.set_neighbors(id, expected[id]));
  }
  const size_t before = entity.neighbor_memory_size();
  ASSERT_EQ(0, entity.compact_neighbors());
  EXPECT_LT(entity.neighbor_memory_size(), before);
  for (size_t id = 0; id < count; ++id) EXPECT_EQ(expected[id], neighbors(id));
  const size_t compacted = entity.neighbor_memory_size();
  ASSERT_EQ(0, entity.compact_neighbors());
  EXPECT_EQ(compacted, entity.neighbor_memory_size());
  EXPECT_EQ(IndexError_NoReady, entity.set_neighbors(0, {1}));
  EXPECT_EQ(IndexError_NoReady, entity.add_neighbor(0, 1));
  EXPECT_EQ(IndexError_NoReady, entity.add_vector(0, holder.get()));
  entity.release_vectors();
  for (size_t id = 0; id < count; ++id) EXPECT_EQ(expected[id], neighbors(id));
  entity.clear();
  EXPECT_EQ(0U, entity.neighbor_memory_size());
  prepare(3, 4);
  ASSERT_EQ(0, entity.set_neighbors(0, {2, 1}));
  ASSERT_EQ(0, entity.compact_neighbors());
  EXPECT_EQ((std::vector<diskann_id_t>{2, 1}), neighbors(0));
}

TEST_F(DiskAnnNeighborCompaction, ValidatesAllRowsBeforeMovingAnyEdges) {
  prepare(6, 4);
  ASSERT_EQ(0, entity.set_neighbors(0, {4, 2, 1}));
  ASSERT_EQ(0, entity.set_neighbors(5, {0, 1, 2, 3, 4}));
  const auto *first = entity.get_neighbors(0).second;
  EXPECT_EQ(IndexError_InvalidLength, entity.compact_neighbors());
  EXPECT_EQ(first, entity.get_neighbors(0).second);
  EXPECT_EQ((std::vector<diskann_id_t>{4, 2, 1}), neighbors(0));
  EXPECT_EQ((std::vector<diskann_id_t>{0, 1, 2, 3, 4}), neighbors(5));
  ASSERT_EQ(0, entity.set_neighbors(5, {0, 1, 2, 3}));
  ASSERT_EQ(0, entity.compact_neighbors());
  EXPECT_EQ((std::vector<diskann_id_t>{0, 1, 2, 3}), neighbors(5));
}

TEST_F(DiskAnnNeighborCompaction, RejectsWritesOutsideReservedRowsAndSlots) {
  prepare(2, 4);
  for (size_t i = 0; i < 5; ++i) ASSERT_EQ(0, entity.add_neighbor(0, 1));
  EXPECT_EQ(IndexError_InvalidLength, entity.add_neighbor(0, 1));
  EXPECT_EQ(IndexError_InvalidLength,
            entity.set_neighbors(0, std::vector<diskann_id_t>(6, 1)));
  EXPECT_EQ(IndexError_InvalidLength, entity.set_neighbors(2, {0}));
  EXPECT_EQ(IndexError_InvalidLength, entity.add_neighbor(2, 0));
  EXPECT_EQ(IndexError_InvalidLength, entity.add_vector(6, holder.get()));
  EXPECT_EQ(2U, entity.doc_cnt());
  EXPECT_EQ(std::vector<diskann_id_t>(5, 1), neighbors(0));
  EXPECT_TRUE(neighbors(1).empty());
}

TEST_F(DiskAnnNeighborCompaction,
       AllocatesEdgesOnDemandAndReleasesEmptyChunks) {
  constexpr size_t count = 100000;
  prepare(count);
  const size_t empty_bytes = entity.neighbor_memory_size();
  EXPECT_LT(empty_bytes, 2u * 1024u * 1024u);
  std::vector<diskann_id_t> expected;
  for (uint32_t i = 0; i < 130; ++i) {
    expected.push_back(i + 1);
    ASSERT_EQ(0, entity.add_neighbor(0, expected.back()));
    ASSERT_EQ(expected, neighbors(0));
  }
  EXPECT_EQ(IndexError_InvalidLength, entity.add_neighbor(0, 131));
  ASSERT_EQ(0, entity.set_neighbors(0, {}));
  EXPECT_EQ(empty_bytes, entity.neighbor_memory_size());
  expected.resize(50);
  for (size_t id = 0; id < count; ++id) {
    ASSERT_EQ(0, entity.set_neighbors(id, expected));
  }
  // During construction, rows with 50 edges must not retain 130 slots each.
  EXPECT_LT(entity.neighbor_memory_size(), count * 524u * 3 / 4);
  for (size_t id = 0; id < count; ++id) ASSERT_EQ(expected, neighbors(id));
}

TEST_F(DiskAnnNeighborCompaction, ConcurrentGrowthShrinkAndChunkReuse) {
  constexpr size_t count = 512;
  prepare(count);
  std::vector<std::thread> workers;
  for (size_t thread = 0; thread < 4; ++thread) {
    workers.emplace_back([&, thread]() {
      for (size_t iteration = 0; iteration < 30; ++iteration) {
        for (size_t id = thread; id < count; id += 4) {
          std::vector<diskann_id_t> row((iteration * 17 + id) % 100,
                                        static_cast<diskann_id_t>(id));
          ASSERT_EQ(0, entity.set_neighbors(id, row));
          ASSERT_EQ(0, entity.add_neighbor(id, (id + 1) % count));
          row.push_back((id + 1) % count);
          ASSERT_EQ(row, neighbors(id));
        }
      }
    });
  }
  for (auto &worker : workers) worker.join();
  for (size_t id = 0; id < count; ++id) {
    if (entity.get_neighbors(id).first > 100) {
      ASSERT_EQ(0, entity.set_neighbors(id, {1, 2, 3}));
    }
  }
  std::vector<std::vector<diskann_id_t>> expected(count);
  for (size_t id = 0; id < count; ++id) expected[id] = neighbors(id);
  ASSERT_EQ(0, entity.compact_neighbors());
  for (size_t id = 0; id < count; ++id) ASSERT_EQ(expected[id], neighbors(id));
}

class FailedNeighborEntity : public DiskAnnBuilderEntity {
 public:
  bool fail_set{false};
  bool fail_append{false};
  int set_neighbors(diskann_id_t id,
                    const std::vector<diskann_id_t> &neighbors) override {
    return fail_set ? IndexError_NoMemory
                    : DiskAnnBuilderEntity::set_neighbors(id, neighbors);
  }
  int add_neighbor(diskann_id_t id, diskann_id_t neighbor) override {
    return fail_append ? IndexError_NoMemory
                       : DiskAnnBuilderEntity::add_neighbor(id, neighbor);
  }
};

TEST(DiskAnnNeighborPoolErrors, AlgorithmPropagatesAllocationFailures) {
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_NE(nullptr, metric);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  auto entity = std::make_shared<FailedNeighborEntity>();
  ASSERT_EQ(0, entity->init(meta, 4, 10, 0, 1));
  auto vectors = std::make_shared<std::vector<float>>(6 * 16);
  ASSERT_EQ(0, entity->reserve_space(6, vectors));
  for (size_t id = 0; id < 6; ++id) {
    std::fill_n(vectors->data() + id * 16, 16, float(id));
    ASSERT_EQ(0, entity->add_vector(id, vectors->data() + id * 16));
  }
  auto context = [&]() {
    auto ctx = std::make_shared<DiskAnnContext>(meta, metric, entity);
    EXPECT_EQ(0, ctx->init(DiskAnnContext::kBuilderContext, 4, 4,
                           meta.element_size()));
    ctx->set_list_size(10);
    return ctx;
  };
  DiskAnnAlgorithm algorithm(*entity, 4);
  entity->fail_set = true;
  EXPECT_EQ(IndexError_NoMemory, algorithm.add_node(1, context().get()));
  entity->fail_set = false;
  entity->fail_append = true;
  EXPECT_EQ(IndexError_NoMemory, algorithm.add_node(1, context().get()));
  ASSERT_EQ(0, entity->set_neighbors(0, {1, 2, 3, 4, 5}));
  entity->fail_set = true;
  EXPECT_EQ(IndexError_NoMemory, algorithm.prune_node(0, context().get()));
}

TEST_F(DiskAnnNeighborCompaction, HandlesEmptyGraphsAndZeroDegreeNodes) {
  prepare(0);
  ASSERT_EQ(0, entity.compact_neighbors());
  ASSERT_EQ(0, entity.compact_neighbors());
  prepare(128);
  ASSERT_EQ(0, entity.compact_neighbors());
  for (size_t id = 0; id < 128; ++id) {
    auto row = entity.get_neighbors(id);
    EXPECT_EQ(0U, row.first);
    EXPECT_NE(nullptr, row.second);
  }
}

TEST_F(DiskAnnNeighborCompaction, DumpIsByteIdenticalAndCanRetryAfterFailure) {
  prepare(128, 8);
  // Historical build degree must still control sector packing after compaction.
  ASSERT_EQ(0, entity.set_neighbors(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  ASSERT_EQ(0, entity.set_neighbors(0, {10, 8, 6}));
  for (size_t id = 1; id < 128; ++id) {
    ASSERT_EQ(0, entity.set_neighbors(id, {(diskann_id_t)((id + 1) % 128), 0}));
  }
  auto original = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, entity.dump(holder, meta, original));
  ASSERT_EQ(0, entity.compact_neighbors());
  entity.release_vectors();
  auto failed = std::make_shared<RecordingDumper>();
  failed->fail_write = true;
  EXPECT_EQ(IndexError_WriteData, entity.dump(holder, meta, failed));
  auto packed = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, entity.dump(holder, meta, packed));
  EXPECT_EQ(original->bytes, packed->bytes);
  EXPECT_EQ(original->segments, packed->segments);
  EXPECT_EQ((std::vector<diskann_id_t>{10, 8, 6}), neighbors(0));
}

}  // namespace
}  // namespace core
}  // namespace zvec
