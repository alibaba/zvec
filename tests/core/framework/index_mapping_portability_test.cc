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
#elif defined(__APPLE__)
#include <sys/mman.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cerrno>
#endif

#if defined(__APPLE__)
namespace {
thread_local bool fail_next_file_mapping = false;
thread_local off_t fail_truncate_to = 0;
}  // namespace

// The File implementation is linked into this test. Fail one native file
// mapping after IndexMapping has written its new, still unlinked section.
extern "C" void *mmap(void *addr, size_t size, int prot, int flags, int fd,
                      off_t offset) {
  if (fd >= 0 && fail_next_file_mapping) {
    fail_next_file_mapping = false;
    errno = ENOMEM;
    return MAP_FAILED;
  }
  static const auto map =
      reinterpret_cast<decltype(&mmap)>(dlsym(RTLD_NEXT, "mmap"));
  return map(addr, size, prot, flags, fd, offset);
}

extern "C" int ftruncate(int fd, off_t len) {
  if (fail_truncate_to && len == fail_truncate_to) {
    fail_truncate_to = 0;
    errno = EIO;
    return -1;
  }
  static const auto truncate =
      reinterpret_cast<decltype(&ftruncate)>(dlsym(RTLD_NEXT, "ftruncate"));
  return truncate(fd, len);
}
#endif

namespace zvec {
namespace core {

namespace {

constexpr uint64_t kOriginal = 1234567;

void WriteForeignIndex(const std::string &path, size_t writer_page,
                       uint32_t first_id_offset = 58,
                       bool legacy_short_footer = false) {
  ailego::File file;
  ASSERT_TRUE(file.create(path, writer_page * 2));
  IndexFormat::MetaHeader header;
  IndexFormat::SetupMetaHeader(
      &header, writer_page - sizeof(IndexFormat::MetaFooter), writer_page);
  ASSERT_EQ(sizeof(header), file.write(0, &header, sizeof(header)));
  std::vector<char> table(first_id_offset + sizeof("first"), 0);
  IndexFormat::SegmentMeta meta{};
  meta.segment_id_offset = first_id_offset;
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
  footer.content_size = legacy_short_footer ? sizeof(kOriginal) : writer_page;
  footer.total_size = writer_page + footer.content_size;
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
    const size_t committed_size = mapping.index_size();
    mapping.close();

    ailego::File file;
    ASSERT_TRUE(file.open(path, false));
    ASSERT_TRUE(file.truncate(committed_size + writer_page));
    file.close();

    ASSERT_EQ(0, mapping.open(path, false, false));
    EXPECT_EQ(committed_size, mapping.index_size());
    ASSERT_TRUE(file.open(path, true));
    EXPECT_EQ(committed_size, file.size());
    file.close();
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

TEST(IndexMappingPortability, TailRecoveryPreservesReservedSegmentPadding) {
  constexpr size_t kWriterPage = 4096;
  constexpr uint32_t kFirstIdOffset = 2 * sizeof(IndexFormat::SegmentMeta) + 16;
  const std::string path = "index_mapping_legacy_padding.tmp";
  ASSERT_NO_FATAL_FAILURE(
      WriteForeignIndex(path, kWriterPage, kFirstIdOffset, true));
  const auto cleanup = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(ailego::File::Delete(path)); });
  ailego::File file;
  ASSERT_TRUE(file.open(path, false));
  ASSERT_TRUE(file.truncate(3 * kWriterPage));
  file.close();

  IndexMapping mapping;
  ASSERT_EQ(0, mapping.open(path, false, false));
  EXPECT_EQ(2 * kWriterPage, mapping.index_size());
  ASSERT_EQ(0, mapping.append("retry", kWriterPage));
  auto *retry = mapping.map("retry", false, false);
  ASSERT_NE(nullptr, retry);
  constexpr uint64_t kValue = 42;
  std::memcpy(retry->data(), &kValue, sizeof(kValue));
  retry->meta()->data_size = sizeof(kValue);
  retry->meta()->padding_size -= sizeof(kValue);
  retry->set_dirty();
  mapping.refresh(1);
  ASSERT_EQ(0, mapping.flush());
  mapping.close();

  ASSERT_EQ(0, mapping.open(path, false, false));
  const auto *first = mapping.map("first", false, false);
  retry = mapping.map("retry", false, false);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, retry);
  EXPECT_EQ(kOriginal, *static_cast<const uint64_t *>(first->data()));
  EXPECT_EQ(kWriterPage - sizeof(kOriginal), first->meta()->padding_size);
  EXPECT_EQ(kValue, *static_cast<const uint64_t *>(retry->data()));
}

#if defined(__APPLE__)
enum class TailRecovery { kImmediate, kReopen, kSameMapping };

static void CheckMetadataMapRecovery(TailRecovery recovery) {
  constexpr size_t kWriterPage = 4096;
  constexpr uint32_t kFirstIdOffset = 2 * sizeof(IndexFormat::SegmentMeta) + 16;
  const bool fail_rollback = recovery != TailRecovery::kImmediate;
  for (bool cow : {false, true}) {
    SCOPED_TRACE(cow);
    const std::string path = "index_mapping_failed_metadata_map.tmp";
    ASSERT_NO_FATAL_FAILURE(
        WriteForeignIndex(path, kWriterPage, kFirstIdOffset));
    const auto cleanup = ailego::ScopeGuard::Make(
        [&] { EXPECT_TRUE(ailego::File::Delete(path)); });
    IndexMapping mapping;
    ASSERT_EQ(0, mapping.open(path, cow, cow));

    const std::string split_id = "segment_id_that_requires_a_metadata_split";
    fail_next_file_mapping = true;
    if (fail_rollback) fail_truncate_to = kWriterPage * 2;
    const int append_status = mapping.append(split_id, kWriterPage);
    EXPECT_FALSE(fail_next_file_mapping);
    fail_next_file_mapping = false;
    EXPECT_EQ(0, fail_truncate_to);
    fail_truncate_to = 0;
    EXPECT_EQ(fail_rollback ? IndexError_TruncateFile : IndexError_MMapFile,
              append_status);
    EXPECT_EQ(0u, mapping.footer().next_meta_header_offset);
    EXPECT_EQ(kWriterPage * 2, mapping.index_size());
    EXPECT_FALSE(mapping.has(split_id));
    if (recovery != TailRecovery::kSameMapping) mapping.close();

    ailego::File file;
    ASSERT_TRUE(file.open(path, true));
    if (fail_rollback) {
      EXPECT_GT(file.size(), kWriterPage * 2);
    } else {
      EXPECT_EQ(kWriterPage * 2, file.size());
    }
    file.close();

    if (recovery == TailRecovery::kReopen) {
      fail_truncate_to = kWriterPage * 2;
      const int reopen_status = mapping.open(path, cow, cow);
      EXPECT_EQ(0, fail_truncate_to);
      fail_truncate_to = 0;
      EXPECT_EQ(IndexError_TruncateFile, reopen_status);
      EXPECT_EQ(0u, mapping.index_size());
    }
    if (recovery == TailRecovery::kSameMapping) {
      fail_truncate_to = kWriterPage * 2;
      EXPECT_EQ(IndexError_TruncateFile, mapping.flush());
      EXPECT_EQ(0, fail_truncate_to);
      fail_truncate_to = kWriterPage * 2;
      EXPECT_EQ(IndexError_TruncateFile, mapping.append("retry", kWriterPage));
      EXPECT_EQ(0, fail_truncate_to);
      fail_truncate_to = 0;
      EXPECT_FALSE(mapping.has("retry"));
    } else {
      ASSERT_EQ(0, mapping.open(path, cow, cow));
      ASSERT_TRUE(file.open(path, true));
      EXPECT_EQ(kWriterPage * 2, file.size());
      file.close();
    }
    // The shorter ID fits in the original directory. Any unlinked tail left
    // on disk would shift the segment past the directory's declared content.
    ASSERT_EQ(0, mapping.append("retry", kWriterPage));
    ASSERT_EQ(0u, mapping.footer().next_meta_header_offset);
    auto *retry = mapping.map("retry", false, false);
    ASSERT_NE(nullptr, retry);
    constexpr uint64_t kValue = 42;
    std::memcpy(retry->data(), &kValue, sizeof(kValue));
    retry->meta()->data_size = sizeof(kValue);
    retry->meta()->padding_size -= sizeof(kValue);
    retry->set_dirty();
    mapping.refresh(1);
    ASSERT_EQ(0, mapping.flush());
    mapping.close();

    ASSERT_EQ(0, mapping.open(path, cow, cow));
    const auto *first = mapping.map("first", false, false);
    retry = mapping.map("retry", false, false);
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, retry);
    EXPECT_EQ(kOriginal, *static_cast<const uint64_t *>(first->data()));
    EXPECT_EQ(kValue, *static_cast<const uint64_t *>(retry->data()));
  }
}

TEST(IndexMappingPortability, FailedMetadataMapRestoresFileBeforeReopen) {
  CheckMetadataMapRecovery(TailRecovery::kImmediate);
}

TEST(IndexMappingPortability, FailedRollbackTruncateIsRetriedOnOpen) {
  CheckMetadataMapRecovery(TailRecovery::kReopen);
}

TEST(IndexMappingPortability,
     FailedRollbackTruncateBlocksWritesUntilRecovered) {
  CheckMetadataMapRecovery(TailRecovery::kSameMapping);
}
#endif

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
