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

#include <cstring>
#include <string>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_mapping.h>
#if defined(_WIN32)
#include <Windows.h>
#endif

namespace zvec {
namespace core {

namespace {

constexpr uint64_t kOriginal = 1234567;

void WriteForeignIndex(const std::string &path, size_t writer_page) {
  ailego::File file;
  ASSERT_TRUE(file.create(path, writer_page * 2));
  IndexFormat::MetaHeader header;
  IndexFormat::SetupMetaHeader(
      &header, writer_page - sizeof(IndexFormat::MetaFooter), writer_page);
  ASSERT_EQ(sizeof(header), file.write(0, &header, sizeof(header)));
  // Leave room for one entry only, to force a new metadata section on append.
  std::vector<char> table(64, 0);
  IndexFormat::SegmentMeta meta{};
  meta.segment_id_offset = 58;
  meta.data_size = sizeof(uint64_t);
  meta.padding_size = writer_page - meta.data_size;
  std::memcpy(table.data(), &meta, sizeof(meta));
  std::memcpy(table.data() + meta.segment_id_offset, "first", 6);
  ASSERT_EQ(table.size(), file.write(header.meta_footer_offset - table.size(),
                                     table.data(), table.size()));
  IndexFormat::MetaFooter footer;
  IndexFormat::SetupMetaFooter(&footer);
  footer.segment_count = 1;
  footer.segments_meta_size = table.size();
  footer.segments_meta_crc =
      ailego::Crc32c::Hash(table.data(), table.size(), 0);
  footer.content_size = writer_page;
  footer.total_size = writer_page * 2;
  IndexFormat::UpdateMetaFooter(&footer, 0);
  ASSERT_EQ(sizeof(footer),
            file.write(header.meta_footer_offset, &footer, sizeof(footer)));
  ASSERT_EQ(sizeof(kOriginal),
            file.write(writer_page, &kOriginal, sizeof(kOriginal)));
  file.close();
}

size_t MappedExtent(size_t offset, size_t len) {
  const size_t page = ailego::MemoryHelper::PageSize();
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const size_t alignment = info.dwAllocationGranularity;
#else
  const size_t alignment = page;
#endif
  return ((offset % alignment + len + page - 1) / page) * page;
}

}  // namespace

TEST(IndexMappingPortability, ForeignPageAppendMetadataChainAndReopen) {
  for (size_t writer_page : {4096u, 65536u}) {
    const std::string path =
        "index_mapping_portability_" + std::to_string(writer_page) + ".tmp";
    ASSERT_NO_FATAL_FAILURE(WriteForeignIndex(path, writer_page));

    IndexMapping mapping;
    ASSERT_EQ(0, mapping.open(path, false, false));
    auto *first = mapping.map("first", false, false);
    ASSERT_NE(nullptr, first);
    EXPECT_EQ(kOriginal, *static_cast<uint64_t *>(first->data()));
    ASSERT_EQ(0, mapping.append("second", 4096));
    auto *second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, second);
    *static_cast<uint64_t *>(second->data()) = 7654321;
    second->meta()->data_size = sizeof(uint64_t);
    second->meta()->padding_size -= sizeof(uint64_t);
    second->set_dirty();
    mapping.refresh(1);
    ASSERT_EQ(0, mapping.flush());
    mapping.close();

    ASSERT_EQ(0, mapping.open(path, false, false));
    first = mapping.map("first", false, false);
    second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);
    EXPECT_EQ(kOriginal, *static_cast<uint64_t *>(first->data()));
    EXPECT_EQ(7654321u, *static_cast<uint64_t *>(second->data()));
    mapping.close();
    ASSERT_TRUE(ailego::File::Delete(path));
  }
}

TEST(IndexMappingPortability, MetadataBudgetUsesNativeMappingExtent) {
  constexpr size_t kWriterPage = 4096;
  const std::string path = "index_mapping_metadata_budget.tmp";
  ASSERT_NO_FATAL_FAILURE(WriteForeignIndex(path, kWriterPage));
  const auto cleanup = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(ailego::File::Delete(path)); });

  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(128UL << 20));
  const auto reset_pool =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(0)); });
  const size_t baseline = pool.external_used();
  const size_t host_page = ailego::MemoryHelper::PageSize();
  const size_t root_extent = MappedExtent(0, kWriterPage);
  const size_t next_extent = MappedExtent(kWriterPage * 2, host_page);
  {
    IndexMapping mapping;
    ASSERT_EQ(0, mapping.open(path, true, true));
    EXPECT_GE(pool.external_used() - baseline, root_extent);
    const size_t before_append = pool.external_used();
    ASSERT_GT(pool.available(), next_extent);
    {
      const size_t held = pool.available() - (next_extent - 1);
      ASSERT_TRUE(pool.try_charge_external(held));
      const auto release =
          ailego::ScopeGuard::Make([&] { pool.release_external(held); });
      EXPECT_EQ(IndexError_NoMemory, mapping.append("second", kWriterPage));
      ASSERT_FALSE(mapping.has("second"));
      EXPECT_EQ(kWriterPage * 2, mapping.index_size());
      EXPECT_EQ(0u, mapping.footer().next_meta_header_offset);
      ailego::File file;
      ASSERT_TRUE(file.open(path, true));
      EXPECT_EQ(kWriterPage * 2, file.size());
    }
    EXPECT_EQ(before_append, pool.external_used());
    ASSERT_EQ(0, mapping.append("second", kWriterPage));
    EXPECT_GE(pool.external_used() - before_append, next_extent);
    auto *second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, second);
    const uint64_t value = 42;
    std::memcpy(second->data(), &value, sizeof(value));
    second->meta()->data_size = sizeof(value);
    second->meta()->padding_size -= sizeof(value);
    second->set_dirty();
    mapping.refresh(1);
    ASSERT_EQ(0, mapping.flush());
  }
  EXPECT_EQ(baseline, pool.external_used());

  {
    ASSERT_GT(pool.available(), root_extent + next_extent);
    const size_t held = pool.available() - (root_extent + next_extent - 1);
    ASSERT_TRUE(pool.try_charge_external(held));
    const auto release =
        ailego::ScopeGuard::Make([&] { pool.release_external(held); });
    {
      IndexMapping mapping;
      EXPECT_EQ(IndexError_NoMemory, mapping.open(path, true, true));
      EXPECT_EQ(0u, mapping.segment_count());
    }
    EXPECT_EQ(baseline + held, pool.external_used());
  }

  {
    IndexMapping mapping;
    ASSERT_EQ(0, mapping.open(path, true, true));
    EXPECT_GE(pool.external_used() - baseline, root_extent + next_extent);
    auto *second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, second);
    EXPECT_EQ(42u, *static_cast<const uint64_t *>(second->data()));
  }
  EXPECT_EQ(baseline, pool.external_used());
}

}  // namespace core
}  // namespace zvec
