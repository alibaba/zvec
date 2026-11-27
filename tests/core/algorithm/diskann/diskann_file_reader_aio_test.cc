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

#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>
#include "diskann_file_reader.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

#if defined(__linux) || defined(__linux__)

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>
#include <ailego/pattern/defer.h>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>

namespace zvec::ailego {

class IoUringStagingTestPeer {
 public:
  IoUringStagingTestPeer(IoUringRing &ring, int fd) : ring_(ring) {
    current_ = this;
    ring.ring_fd_ = ::dup(fd);
    ring.sq_entries_ = sqes_.size();
    ring.sq_tail_ = &sq_tail_;
    ring.sq_ring_mask_ = &mask_;
    ring.sq_array_ = sq_array_.data();
    ring.sqes_ = sqes_.data();
    ring.cq_head_ = &cq_head_;
    ring.cq_tail_ = &cq_tail_;
    ring.cq_ring_mask_ = &mask_;
    ring.cqes_ = cqes_.data();
    ring.enter_override_ = enter;
  }

  ~IoUringStagingTestPeer() {
    ring_.teardown();
    current_ = nullptr;
  }

 private:
  static int enter(int, unsigned submitted, unsigned, unsigned) {
    auto &self = *current_;
    if (submitted == 3) {
      const auto &sqe = self.sqes_[1];
      std::memset(reinterpret_cast<void *>(sqe.addr), 0x37, sqe.len);
      self.cqes_[0].user_data = sqe.user_data;
      self.cqes_[0].res = static_cast<int32_t>(sqe.len);
      self.cq_tail_ = 1;
      return 3;
    }

    EXPECT_EQ(0u, submitted);
    self.cqes_[1].user_data = self.sqes_[0].user_data;
    self.cqes_[1].res = -EIO;
    self.cqes_[2].user_data = self.sqes_[2].user_data;
    self.cqes_[2].res = static_cast<int32_t>(self.sqes_[2].len);
    self.cq_tail_ = 3;
    return 0;
  }

  inline static thread_local IoUringStagingTestPeer *current_{nullptr};
  IoUringRing &ring_;
  unsigned sq_tail_{0};
  unsigned cq_head_{0};
  unsigned cq_tail_{0};
  unsigned mask_{3};
  std::array<unsigned, 4> sq_array_{0, 1, 2, 3};
  std::array<io_uring_sqe, 4> sqes_{};
  std::array<io_uring_cqe, 4> cqes_{};
};

}  // namespace zvec::ailego

namespace zvec {
namespace core {
int execute_io_libaio(IoBackend &ctx, int fd,
                      std::vector<AlignedRead> &read_reqs,
                      uint64_t n_retries = 0);
}  // namespace core
}  // namespace zvec

using namespace zvec::core;
namespace ailego = zvec::ailego;

namespace {

constexpr size_t kBlockSize = 512;
constexpr size_t kBlockCount = 4;

struct FakeAioState {
  std::vector<int> submit_results;
  std::vector<int> completion_results;
  std::vector<long> submit_sizes;
  std::vector<long> completion_sizes;
  std::vector<struct iocb *> submitted;
  std::vector<void *> kernel_buffers;
  int destroy_result = -EINVAL;
  size_t destroy_calls = 0;
  size_t submit_call = 0;
  size_t completion_call = 0;
  size_t completed = 0;
  size_t short_completion = std::numeric_limits<size_t>::max();
};

FakeAioState *g_fake_aio = nullptr;

int fake_io_submit(io_context_t, long nr, struct iocb *ios[]) {
  if (g_fake_aio == nullptr ||
      g_fake_aio->submit_call >= g_fake_aio->submit_results.size()) {
    return -EINVAL;
  }

  g_fake_aio->submit_sizes.push_back(nr);
  int ret = g_fake_aio->submit_results[g_fake_aio->submit_call++];
  if (ret <= 0) {
    return ret;
  }
  if (ret > nr) {
    return -EINVAL;
  }

  for (int i = 0; i < ret; ++i) {
    g_fake_aio->submitted.push_back(ios[i]);
    g_fake_aio->kernel_buffers.push_back(ios[i]->u.c.buf);
  }
  return ret;
}

int fake_io_getevents(io_context_t, long min_nr, long nr,
                      struct io_event *events, struct timespec *) {
  if (g_fake_aio == nullptr || min_nr < 0 || min_nr > nr ||
      g_fake_aio->completion_call >= g_fake_aio->completion_results.size()) {
    return -EINVAL;
  }

  g_fake_aio->completion_sizes.push_back(nr);
  int ret = g_fake_aio->completion_results[g_fake_aio->completion_call++];
  if (ret <= 0) {
    return ret;
  }
  if (ret > nr || g_fake_aio->completed + static_cast<size_t>(ret) >
                      g_fake_aio->submitted.size()) {
    return -EINVAL;
  }

  for (int i = 0; i < ret; ++i) {
    size_t completion_index = g_fake_aio->completed++;
    struct iocb *cb = g_fake_aio->submitted[completion_index];
    std::memset(cb->u.c.buf, 0xa5, cb->u.c.nbytes);
    events[i].data = cb->data;
    events[i].obj = cb;
    events[i].res = completion_index == g_fake_aio->short_completion
                        ? cb->u.c.nbytes - 1
                        : cb->u.c.nbytes;
    events[i].res2 = 0;
  }
  return ret;
}

int fake_io_setup(int, io_context_t *) {
  return -EINVAL;
}

int fake_io_destroy(io_context_t) {
  ++g_fake_aio->destroy_calls;
  return g_fake_aio->destroy_result;
}

class FakeAioGuard {
 public:
  explicit FakeAioGuard(FakeAioState *state)
      : loader_(LibAioLoader::Instance()),
        original_setup_(loader_.io_setup),
        original_submit_(loader_.io_submit),
        original_getevents_(loader_.io_getevents),
        original_destroy_(loader_.io_destroy) {
    g_fake_aio = state;
    loader_.io_setup = fake_io_setup;
    loader_.io_submit = fake_io_submit;
    loader_.io_getevents = fake_io_getevents;
    loader_.io_destroy = fake_io_destroy;
  }

  ~FakeAioGuard() {
    loader_.io_setup = original_setup_;
    loader_.io_submit = original_submit_;
    loader_.io_getevents = original_getevents_;
    loader_.io_destroy = original_destroy_;
    g_fake_aio = nullptr;
  }

 private:
  LibAioLoader &loader_;
  aio_setup_fn original_setup_;
  aio_submit_fn original_submit_;
  aio_getevents_fn original_getevents_;
  aio_destroy_fn original_destroy_;
};

class FailingAioLogger final : public ailego::Logger {
 public:
  explicit FailingAioLogger(const char *message) : message_(message) {}
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *format, va_list args) override {
    char message[512];
    std::vsnprintf(message, sizeof(message), format, args);
    if (std::strstr(message, message_)) throw std::bad_alloc();
  }

 private:
  const char *message_;
};

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_);
  }

  int fd() const {
    return fd_;
  }

  const char *path() const {
    return path_;
  }

 private:
  char path_[64] = "DiskAnnLinuxAioTest.XXXXXX";
  int fd_;
};

void *allocate_aligned(size_t size) {
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kBlockSize, size) != 0) {
    return nullptr;
  }
  std::memset(buffer, 0, size);
  return buffer;
}

std::vector<AlignedRead> make_requests(void *buffer) {
  std::vector<AlignedRead> requests;
  requests.reserve(kBlockCount);
  for (size_t i = 0; i < kBlockCount; ++i) {
    requests.emplace_back(i * kBlockSize, kBlockSize,
                          static_cast<uint8_t *>(buffer) + i * kBlockSize);
  }
  return requests;
}

std::vector<uint8_t> make_source() {
  std::vector<uint8_t> source(kBlockSize * kBlockCount);
  for (size_t block = 0; block < kBlockCount; ++block) {
    std::fill(source.begin() + block * kBlockSize,
              source.begin() + (block + 1) * kBlockSize,
              static_cast<uint8_t>(block + 1));
  }
  return source;
}

}  // namespace

TEST(DiskAnnLinuxAioTest, AccumulatesPartialSubmissionsAndCompletions) {
  void *output = allocate_aligned(kBlockSize * kBlockCount);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests = make_requests(output);

  FakeAioState state;
  state.submit_results = {2, 2};
  state.completion_results = {1, 2, 1};
  FakeAioGuard guard(&state);
  IoBackend ctx;
  ctx.aio_ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

  // An invalid fd makes any accidental pread fallback fail the test.
  EXPECT_EQ(execute_io_libaio(ctx, -1, requests), 0);
  EXPECT_EQ(state.submit_sizes, (std::vector<long>{4, 2}));
  EXPECT_EQ(state.completion_sizes, (std::vector<long>{4, 3, 1}));
  EXPECT_EQ(state.completed, kBlockCount);
  const auto *bytes = static_cast<const uint8_t *>(output);
  EXPECT_TRUE(std::all_of(bytes, bytes + kBlockSize * kBlockCount,
                          [](uint8_t value) { return value == 0xa5; }));

  std::free(output);
}

TEST(DiskAnnBufferPoolFileReaderTest,
     ReadsScatteredRequestsThroughOnePinnedPageBatch) {
  if (ailego::kVectorPageSize != DiskAnnUtil::kSectorSize) {
    GTEST_SKIP() << "DiskAnn sectors require one native buffer-pool page";
  }

  constexpr size_t kPageCount = 4;
  const size_t page_size = ailego::kVectorPageSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(kPageCount * page_size);
  for (size_t page = 0; page < kPageCount; ++page) {
    std::memset(source.data() + page * page_size, static_cast<int>(page + 1),
                page_size);
  }
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));
  ASSERT_EQ(::fsync(file.fd()), 0);

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      0, memory_pool.init(
             8 * page_size +
             ailego::VecBufferPool::metadata_bytes_for_page_count(kPageCount)));
  auto pool =
      std::make_shared<ailego::VecBufferPool>(file.path(), /*writable=*/false);
  ASSERT_EQ(pool->init(), 0);
  BufferPoolAlignedFileReader reader(pool);
  EXPECT_FALSE(reader.requires_io_context());
  reader.open(file.path());

  void *output = nullptr;
  ASSERT_EQ(::posix_memalign(&output, page_size, 4 * page_size), 0);
  ASSERT_NE(output, nullptr);
  std::memset(output, 0, 4 * page_size);
  std::vector<AlignedRead> requests;
  requests.emplace_back(3 * page_size, page_size, output);
  requests.emplace_back(page_size, 2 * page_size,
                        static_cast<char *>(output) + page_size);
  requests.emplace_back(3 * page_size, page_size,
                        static_cast<char *>(output) + 3 * page_size);

  IOContext unused{};
  ASSERT_EQ(reader.read(requests, unused), 0);
  EXPECT_EQ(std::memcmp(output, source.data() + 3 * page_size, page_size), 0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + page_size,
                        source.data() + page_size, 2 * page_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + 3 * page_size,
                        source.data() + 3 * page_size, page_size),
            0);
  EXPECT_EQ(pool->stats().miss, 3u);

  pool->page_table_.force_evict_all_loaded();
  std::free(output);
}

TEST(DiskAnnBufferPoolFileReaderTest,
     BypassesColdMissUnderPressureAndFansOutDuplicates) {
  if (ailego::kVectorPageSize != DiskAnnUtil::kSectorSize) {
    GTEST_SKIP() << "DiskAnn sectors require one native buffer-pool page";
  }

  constexpr size_t kPageCount = 4;
  const size_t page_size = ailego::kVectorPageSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(kPageCount * page_size);
  for (size_t page = 0; page < kPageCount; ++page) {
    std::memset(source.data() + page * page_size, static_cast<int>(page + 1),
                page_size);
  }
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));
  ASSERT_EQ(::fsync(file.fd()), 0);

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      0, memory_pool.init(
             2 * page_size +  // One payload page and its slab header.
             ailego::VecBufferPool::metadata_bytes_for_page_count(kPageCount)));
  auto pool =
      std::make_shared<ailego::VecBufferPool>(file.path(), /*writable=*/false);
  ASSERT_EQ(pool->init(), 0);
  BufferPoolAlignedFileReader reader(pool);
  reader.open(file.path());

  char *seed = pool->acquire_buffer(0, 10);
  ASSERT_NE(seed, nullptr);

  void *output = nullptr;
  ASSERT_EQ(::posix_memalign(&output, page_size, 4 * page_size), 0);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests;
  requests.emplace_back(page_size, 3 * page_size, output);
  requests.emplace_back(2 * page_size, page_size,
                        static_cast<char *>(output) + 3 * page_size);
  IOContext unused{};

  ASSERT_EQ(reader.read(requests, unused), 0);
  EXPECT_EQ(std::memcmp(output, source.data() + page_size, page_size), 0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + page_size,
                        source.data() + 2 * page_size, page_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + 2 * page_size,
                        source.data() + 3 * page_size, page_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + 3 * page_size,
                        source.data() + 2 * page_size, page_size),
            0);
  EXPECT_FALSE(pool->is_page_resident(1));
  EXPECT_FALSE(pool->is_page_resident(2));
  EXPECT_FALSE(pool->is_page_resident(3));
  EXPECT_EQ(pool->stats().admission_rejected, 3u);
  EXPECT_EQ(pool->stats().bypass_reads, 1u);
  EXPECT_EQ(pool->stats().bypass_bytes, 3 * page_size);
  EXPECT_EQ(pool->stats().bypass_io_requests, 1u);
  EXPECT_EQ(pool->stats().bypass_rechecks, 3u);
  EXPECT_EQ(pool->stats().bypass_cache_joins, 0u);

  // The second observation promotes these ghost entries for admission, but
  // the only cache page is still pinned. Admission failure must fall back to
  // direct I/O instead of failing the query.
  std::memset(output, 0, 4 * page_size);
  ASSERT_EQ(reader.read(requests, unused), 0);
  EXPECT_EQ(std::memcmp(output, source.data() + page_size, page_size), 0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + page_size,
                        source.data() + 2 * page_size, page_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<char *>(output) + 2 * page_size,
                        source.data() + 3 * page_size, page_size),
            0);

  pool->page_table_.release_block(0);
  pool->page_table_.force_evict_all_loaded();
  EXPECT_EQ(destroy_io_ctx(unused), 0);
  std::free(output);
}

TEST(DiskAnnBufferPoolFileReaderTest, RejectsNonPageAlignedRequests) {
  if (ailego::kVectorPageSize <= 512) {
    GTEST_SKIP() << "test requires a native page larger than 512 bytes";
  }

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(2 * ailego::kVectorPageSize, 0x5a);
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, memory_pool.init(
                   4 * ailego::kVectorPageSize +
                   ailego::VecBufferPool::metadata_bytes_for_page_count(2)));
  auto pool =
      std::make_shared<ailego::VecBufferPool>(file.path(), /*writable=*/false);
  ASSERT_EQ(pool->init(), 0);
  BufferPoolAlignedFileReader reader(pool);

  ASSERT_EQ(reader.open_from_pool(file.path()), 0);
  void *output = allocate_aligned(512);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests;
  requests.emplace_back(512, 512, output);
  IOContext unused{};
  EXPECT_EQ(reader.read(requests, unused), IndexError_InvalidArgument);
  EXPECT_EQ(unused, nullptr);
  std::free(output);
}

class DiskAnnLargeFileReuseTest : public testing::TestWithParam<bool> {
 protected:
  void check_working_set(bool previously_evicted) {
    const size_t page_size = ailego::kVectorPageSize;
    constexpr size_t file_bytes = 512UL << 20;
    constexpr size_t budget_bytes = 256UL << 20;
    TemporaryFile file;
    ASSERT_GE(file.fd(), 0);
    ASSERT_EQ(::ftruncate(file.fd(), file_bytes), 0);
    const ailego::block_id_t ids[] = {3, 50, file_bytes / page_size - 13};
    std::vector<char> page(page_size);
    for (size_t id : ids) {
      std::fill(page.begin(), page.end(), static_cast<char>(id % 127 + 1));
      ASSERT_EQ(::pwrite(file.fd(), page.data(), page.size(), id * page_size),
                static_cast<ssize_t>(page.size()));
    }
    ASSERT_EQ(::fsync(file.fd()), 0);
    auto &memory = ailego::PoolPayloadBudget::get_instance();
    ASSERT_EQ(memory.init(budget_bytes), 0);
    ASSERT_GT(memory.page_admission_reserve(), 0u);
    auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
    ASSERT_EQ(pool->init(), 0);
    ASSERT_GT(pool->file_size(), memory.capacity());
    if (previously_evicted) {
      ASSERT_NE(pool->acquire_buffer(0, 1), nullptr);
      const ailego::block_id_t seed = 0;
      pool->release_pages(&seed, 1);
      pool->page_table_.force_evict_all_loaded();
      ASSERT_TRUE(pool->has_evicted());
    }
    BufferPoolAlignedFileReader reader(pool);
    reader.open(file.path());
    void *output = nullptr;
    ASSERT_EQ(::posix_memalign(&output, page_size, 3 * page_size), 0);
    auto cleanup = ailego::ScopeGuard::Make([&] { std::free(output); });
    IOContext ctx{};
    auto io_cleanup = ailego::ScopeGuard::Make([&] {
      reader.release_io_ctx(ctx);
      pool->page_table_.force_evict_all_loaded();
    });
    std::vector<AlignedRead> requests;
    for (size_t i = 0; i < 3; ++i)
      requests.emplace_back(ids[i] * page_size, page_size,
                            static_cast<char *>(output) + i * page_size);
    for (int pass = 0; pass < (previously_evicted ? 1 : 2); ++pass) {
      std::memset(output, 0, 3 * page_size);
      if (GetParam()) {
        PendingBatch batch;
        ASSERT_EQ(reader.submit(batch, requests, ctx), 0);
        size_t completed = 0;
        while (completed < requests.size()) {
          std::vector<uint32_t> ready;
          const int count = reader.get_completed(batch, ctx, 1, ready);
          ASSERT_GT(count, 0);
          completed += ready.size();
        }
      } else {
        ASSERT_EQ(reader.read(requests, ctx), 0);
      }
      for (size_t i = 0; i < 3; ++i) {
        std::fill(page.begin(), page.end(),
                  static_cast<char>(ids[i] % 127 + 1));
        EXPECT_EQ(std::memcmp(static_cast<char *>(output) + i * page_size,
                              page.data(), page_size),
                  0);
        EXPECT_EQ(pool->is_page_resident(ids[i]), !previously_evicted);
      }
      if (!previously_evicted) EXPECT_EQ(pool->stats().bypass_bytes, 0u);
      EXPECT_LE(memory.used(), memory.capacity());
    }
  }
};

TEST_P(DiskAnnLargeFileReuseTest, RetainsSmallWorkingSetBeforeFirstEviction) {
  check_working_set(false);
}

TEST_P(DiskAnnLargeFileReuseTest, KeepsFrequencyAdmissionAfterEviction) {
  check_working_set(true);
}

INSTANTIATE_TEST_SUITE_P(ReadAndSubmit, DiskAnnLargeFileReuseTest,
                         testing::Bool());

TEST(DiskAnnLinuxAioTest, CoalescedBypassCountsOneIoRequest) {
  const size_t page_size = ailego::kVectorPageSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  ASSERT_EQ(::ftruncate(file.fd(), 512UL << 20), 0);
  auto &memory = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = memory.capacity();
  ASSERT_EQ(memory.init(256UL << 20), 0);
  const auto restore =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, memory.init(previous)); });
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(pool->init(), 0);
  ASSERT_NE(pool->acquire_buffer(0, 1), nullptr);
  const ailego::block_id_t seed = 0;
  pool->release_pages(&seed, 1);
  pool->page_table_.force_evict_all_loaded();
  ASSERT_TRUE(pool->has_evicted());

  void *output = nullptr;
  ASSERT_EQ(::posix_memalign(&output, page_size, 2 * page_size), 0);
  const auto cleanup = ailego::ScopeGuard::Make([&] { std::free(output); });
  auto *bytes = static_cast<uint8_t *>(output);
  std::vector<AlignedRead> requests{
      AlignedRead(3 * page_size, page_size, bytes),
      AlignedRead(4 * page_size, page_size, bytes + page_size)};
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(reader.open_from_pool(file.path()), 0);
  FakeAioState state;
  state.submit_results = {1};
  state.completion_results = {1};
  state.destroy_result = 0;
  FakeAioGuard guard(&state);
  IOContext ctx = new IoBackend;
  ctx->type = ailego::IOBackendType::kLibAio;
  ctx->aio_ctx = reinterpret_cast<io_context_t>(uintptr_t{1});
  const auto release = ailego::ScopeGuard::Make([&] { destroy_io_ctx(ctx); });
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> completed;
  ASSERT_EQ(2, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ((std::vector<uint32_t>{0, 1}), completed);
  EXPECT_EQ((std::vector<long>{1}), state.submit_sizes);
  EXPECT_EQ(2 * page_size, pool->stats().bypass_bytes);
  EXPECT_EQ(1u, pool->stats().bypass_io_requests);
  EXPECT_EQ(2u, pool->stats().bypass_rechecks);
  EXPECT_TRUE(std::all_of(bytes, bytes + 2 * page_size,
                          [](uint8_t byte) { return byte == 0xa5; }));
}

TEST(DiskAnnLinuxAioTest, PartialSubmissionLogFailureRetiresStaging) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  auto output = std::unique_ptr<void, decltype(&std::free)>(
      allocate_aligned(kBlockSize * kBlockCount), &std::free);
  ASSERT_NE(output, nullptr);
  auto requests = make_requests(output.get());
  auto &memory = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = memory.capacity();
  ASSERT_EQ(memory.init(4 * 1024 * 1024), 0);
  const auto restore_budget =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, memory.init(previous)); });

  auto logger = std::make_shared<FailingAioLogger>("io_submit stopped");
  auto previous_logger = ailego::LoggerBroker::Register(std::move(logger));
  const auto restore_logger = ailego::ScopeGuard::Make(
      [&] { ailego::LoggerBroker::Register(std::move(previous_logger)); });

  LinuxAlignedFileReader reader;
  ASSERT_EQ(0, reader.open_from_handle(file.path(), file.fd()));
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    FakeAioState state;
    state.submit_results = {1, -EIO};
    state.destroy_result = 0;
    FakeAioGuard guard(&state);
    IOContext ctx = new IoBackend;
    ctx->type = ailego::IOBackendType::kLibAio;
    ctx->aio_ctx = reinterpret_cast<io_context_t>(uintptr_t{1});
    const auto release = ailego::ScopeGuard::Make([&] { destroy_io_ctx(ctx); });
    PendingBatch batch;
    if (async) {
      EXPECT_THROW(reader.submit(batch, requests, ctx), std::bad_alloc);
    } else {
      EXPECT_THROW(execute_io_libaio(*ctx, file.fd(), requests),
                   std::bad_alloc);
    }
    EXPECT_EQ(2u, state.submit_call);
    EXPECT_EQ(1u, state.destroy_calls);
    EXPECT_FALSE(ctx->aio_active);
    EXPECT_EQ(nullptr, ctx->aio_batch);
    EXPECT_FALSE(ctx->aio_staging);
  }
}

TEST(DiskAnnLinuxAioTest, CacheFullGraphSubmitFallsBackWithoutCapacityWait) {
  const size_t page = ailego::kVectorPageSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  const std::vector<uint8_t> source(2 * page, 0x67);
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  auto &memory = ailego::PoolPayloadBudget::get_instance();
  auto &rss = ailego::MemoryLimitPool::get_instance();
  const size_t previous_capacity = memory.capacity();
  const auto previous_limit = rss.process_rss_limit();
  ASSERT_EQ(
      0, memory.init(2 * page +
                     ailego::VecBufferPool::metadata_bytes_for_page_count(2)));
  const auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(rss.configure_process_rss_limit(previous_limit));
    EXPECT_EQ(0, memory.init(previous_capacity));
  });
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(0, pool->init());
  ASSERT_TRUE(pool->cache_enabled());
  const ailego::block_id_t hot = 0;
  ASSERT_NE(nullptr, pool->acquire_buffer(hot, 1));
  const auto unpin =
      ailego::ScopeGuard::Make([&] { pool->release_pages(&hot, 1); });
  ASSERT_TRUE(rss.configure_process_rss_limit(uint64_t{1} << 40));
  auto output = std::unique_ptr<void, decltype(&std::free)>(
      allocate_aligned(page), &std::free);
  ASSERT_NE(nullptr, output);
  std::vector<AlignedRead> reads{{page, page, output.get()}};
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(0, reader.open_from_pool(file.path()));
  IOContext ctx = new IoBackend;
  ctx->type = ailego::IOBackendType::kLibAio;
  const auto release = ailego::ScopeGuard::Make([&] { destroy_io_ctx(ctx); });

  const auto waits = rss.stats().capacity_waits;
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, reads, ctx));
  EXPECT_TRUE(batch.used_pread);
  EXPECT_EQ(waits, rss.stats().capacity_waits);
  std::vector<uint32_t> done;
  ASSERT_EQ(1, reader.get_completed(batch, ctx, 1, done));
  EXPECT_EQ((std::vector<uint32_t>{0}), done);
  EXPECT_EQ(0, std::memcmp(output.get(), source.data() + page, page));
}

TEST(DiskAnnLinuxAioTest, OptionalStagingFallsBackWithoutCapacityWait) {
  TemporaryFile file;
  const auto source = make_source();
  ASSERT_GE(file.fd(), 0);
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  auto output = std::unique_ptr<void, decltype(&std::free)>(
      allocate_aligned(source.size()), &std::free);
  ASSERT_NE(output, nullptr);
  auto requests = make_requests(output.get());

  auto &memory = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = memory.capacity();
  const size_t page = memory.page_buffer_size();
  ASSERT_EQ(0, memory.init(16 * page));
  const auto restore =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, memory.init(previous)); });
  ailego::MemoryReservation occupied;
  occupied.add(14 * page);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const uint64_t waits = pool.stats().capacity_waits;

  FakeAioState state;
  state.submit_results = {static_cast<int>(requests.size())};
  state.completion_results = {static_cast<int>(requests.size())};
  state.destroy_result = 0;
  FakeAioGuard guard(&state);
  IOContext ctx = new IoBackend;
  ctx->type = ailego::IOBackendType::kLibAio;
  ctx->aio_ctx = reinterpret_cast<io_context_t>(uintptr_t{1});
  const auto release = ailego::ScopeGuard::Make([&] { destroy_io_ctx(ctx); });
  ASSERT_EQ(0, execute_io_libaio(*ctx, file.fd(), requests));
  EXPECT_EQ(0u, state.submit_call);
  EXPECT_EQ(waits, pool.stats().capacity_waits);
  EXPECT_EQ(0, std::memcmp(output.get(), source.data(), source.size()));
}

TEST(DiskAnnLinuxAioTest, TeardownFinishesWhenFailureLogCannotAllocate) {
  FakeAioState state;
  state.destroy_result = -EPERM;
  FakeAioGuard guard(&state);
  IOContext ctx = new IoBackend;
  ctx->type = ailego::IOBackendType::kLibAio;
  ctx->aio_ctx = reinterpret_cast<io_context_t>(uintptr_t{1});
  const auto release = ailego::ScopeGuard::Make([&] {
    state.destroy_result = 0;
    destroy_io_ctx(ctx);
  });
  auto logger = std::make_shared<FailingAioLogger>("io_destroy failed");
  auto previous_logger = ailego::LoggerBroker::Register(std::move(logger));
  const auto restore = ailego::ScopeGuard::Make(
      [&] { ailego::LoggerBroker::Register(std::move(previous_logger)); });
  EXPECT_EQ(IndexError_Runtime, destroy_io_ctx(ctx));
  EXPECT_EQ(1u, state.destroy_calls);
  EXPECT_EQ(nullptr, ctx);
}

TEST(DiskAnnLinuxAioTest, FailedWaitRetriesOnlyUndeliveredReads) {
  TemporaryFile file;
  const auto source = make_source();
  ASSERT_GE(file.fd(), 0);
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  auto output = std::unique_ptr<void, decltype(&std::free)>(
      allocate_aligned(source.size()), &std::free);
  ASSERT_NE(output, nullptr);
  auto requests = make_requests(output.get());
  auto &memory = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = memory.capacity();
  ASSERT_EQ(memory.init(4 * 1024 * 1024), 0);
  const auto restore =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, memory.init(previous)); });
  FakeAioState state;
  state.submit_results = {static_cast<int>(requests.size())};
  state.completion_results = {1, -EINTR, -EIO};
  state.destroy_result = 0;
  FakeAioGuard guard(&state);
  IOContext ctx = new IoBackend;
  ctx->type = ailego::IOBackendType::kLibAio;
  ctx->aio_ctx = reinterpret_cast<io_context_t>(uintptr_t{1});
  const auto release = ailego::ScopeGuard::Make([&] { destroy_io_ctx(ctx); });
  LinuxAlignedFileReader reader;
  ASSERT_EQ(0, reader.open_from_handle(file.path(), file.fd()));
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> completed;
  ASSERT_EQ(1, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ((std::vector<uint32_t>{0}), completed);

  ASSERT_EQ(3, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ((std::vector<uint32_t>{1, 2, 3}), completed);
  EXPECT_EQ(1u, state.destroy_calls);
  EXPECT_FALSE(batch.has_async_requests());
  EXPECT_EQ(0, reader.get_completed(batch, ctx, 1, completed));
  const auto *bytes = static_cast<const uint8_t *>(output.get());
  EXPECT_TRUE(std::all_of(bytes, bytes + kBlockSize,
                          [](uint8_t byte) { return byte == 0xa5; }));
  EXPECT_EQ(0, std::memcmp(bytes + kBlockSize, source.data() + kBlockSize,
                           source.size() - kBlockSize));
}

TEST(DiskAnnLinuxAioTest, FailedCancellationKeepsChargedStagingIsolated) {
  TemporaryFile file;
  auto source = make_source();
  ASSERT_GE(file.fd(), 0);
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  auto output =
      std::unique_ptr<void, decltype(&std::free)>(allocate_aligned(source.size()),
                                                 &std::free);
  ASSERT_NE(nullptr, output);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_EQ(0, pool.init(previous)); });
  const size_t baseline = pool.external_used();

  {
    std::memset(output.get(), 0, source.size());
    auto requests = make_requests(output.get());
    FakeAioState state;
    state.submit_results = {static_cast<int>(requests.size())};
    state.completion_results = {-EIO};
    state.destroy_result = -EPERM;
    FakeAioGuard guard(&state);
    IOContext ctx = new IoBackend;
    ctx->type = ailego::IOBackendType::kLibAio;
    ctx->aio_ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));
    const auto cleanup = ailego::ScopeGuard::Make([&] {
      state.destroy_result = 0;
      EXPECT_EQ(0, destroy_io_ctx(ctx));
    });
    LinuxAlignedFileReader reader;
    ASSERT_EQ(0, reader.open_from_handle(file.path(), file.fd()));
    PendingBatch batch;
    std::vector<uint32_t> completed;
    ASSERT_EQ(0, reader.submit(batch, requests, ctx));
    EXPECT_EQ(IndexError_Runtime,
              reader.get_completed(batch, ctx, requests.size(), completed));
    ASSERT_TRUE(ctx->aio_quarantined);
    ASSERT_TRUE(ctx->aio_staging);
    ASSERT_EQ(requests.size(), state.kernel_buffers.size());
    for (size_t i = 0; i < requests.size(); ++i) {
      EXPECT_NE(requests[i].buf, state.kernel_buffers[i]);
      std::memset(state.kernel_buffers[i], 0xa5, requests[i].len);
    }
    const auto *data = static_cast<const uint8_t *>(output.get());
    EXPECT_TRUE(std::all_of(data, data + source.size(),
                            [](uint8_t byte) { return byte == 0; }));
    EXPECT_GT(pool.external_used(), baseline);
    EXPECT_EQ(0, reader.read(requests, ctx));
    EXPECT_EQ(1u, state.submit_call);
    EXPECT_EQ(0, std::memcmp(output.get(), source.data(), source.size()));
  }
  EXPECT_EQ(baseline, pool.external_used());
}

TEST(DiskAnnLinuxAioTest, FailedRingCompletionRetriesUndeliveredReads) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  const auto source = make_source();
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  constexpr uint64_t kTailSize = 127;
  ASSERT_EQ(0, ::ftruncate(file.fd(), 2 * kBlockSize + kTailSize));
  auto output =
      std::unique_ptr<void, decltype(&std::free)>(allocate_aligned(source.size()),
                                                 &std::free);
  ASSERT_NE(nullptr, output);

  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto old_capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_EQ(0, pool.init(old_capacity)); });
  const size_t before = pool.external_used();
  IoBackend backend;
  backend.type = ailego::IOBackendType::kIoUring;
  ailego::IoUringStagingTestPeer ring(backend.ring, file.fd());
  IOContext ctx = &backend;
  LinuxAlignedFileReader reader(::dup(file.fd()));
  auto requests = make_requests(output.get());
  requests.resize(3);
  requests[2].expected_len = kTailSize;
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  ASSERT_FALSE(batch.used_pread);
  std::vector<uint32_t> completed;
  ASSERT_EQ(1, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ((std::vector<uint32_t>{1}), completed);
  ASSERT_EQ(2, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ((std::vector<uint32_t>{0, 2}), completed);
  EXPECT_EQ(0, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_FALSE(backend.ring.has_pending_reads());
  EXPECT_EQ(before, pool.external_used());
  const auto *bytes = static_cast<const uint8_t *>(output.get());
  EXPECT_EQ(0, std::memcmp(bytes, source.data(), kBlockSize));
  EXPECT_TRUE(std::all_of(bytes + kBlockSize, bytes + 2 * kBlockSize,
                          [](uint8_t value) { return value == 0x37; }));
  EXPECT_EQ(0, std::memcmp(bytes + 2 * kBlockSize,
                           source.data() + 2 * kBlockSize, kTailSize));
  EXPECT_TRUE(std::all_of(bytes + 2 * kBlockSize + kTailSize,
                          bytes + 3 * kBlockSize,
                          [](uint8_t value) { return value == 0; }));
}

TEST(DiskAnnLinuxAioTest, DrainsPartialSubmissionBeforePreadFallback) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source = make_source();
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));

  void *output = allocate_aligned(source.size());
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests = make_requests(output);

  FakeAioState state;
  state.submit_results = {2, -EAGAIN};
  state.completion_results = {1, 1};
  FakeAioGuard guard(&state);
  IoBackend ctx;
  ctx.aio_ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

  EXPECT_EQ(execute_io_libaio(ctx, file.fd(), requests), 0);
  EXPECT_EQ(state.submit_sizes, (std::vector<long>{4, 2}));
  EXPECT_EQ(state.completion_sizes, (std::vector<long>{2, 1}));
  EXPECT_EQ(state.completed, 2u);
  EXPECT_EQ(std::memcmp(output, source.data(), source.size()), 0);

  std::free(output);
}

TEST(DiskAnnLinuxAioTest, DrainsAllCompletionsBeforePreadFallback) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source = make_source();
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));

  void *output = allocate_aligned(source.size());
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests = make_requests(output);

  FakeAioState state;
  state.submit_results = {4};
  state.completion_results = {1, 1, 2};
  state.short_completion = 0;
  FakeAioGuard guard(&state);
  IoBackend ctx;
  ctx.aio_ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

  EXPECT_EQ(execute_io_libaio(ctx, file.fd(), requests), 0);
  EXPECT_EQ(state.completion_sizes, (std::vector<long>{4, 3, 2}));
  EXPECT_EQ(state.completed, kBlockCount);
  EXPECT_EQ(std::memcmp(output, source.data(), source.size()), 0);

  std::free(output);
}

#endif  // __linux__
