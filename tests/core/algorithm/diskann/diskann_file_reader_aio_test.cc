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
#include "diskann_file_reader.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

#if defined(__linux) || defined(__linux__)

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>

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
  int destroy_result = -EIO;
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
  if (g_fake_aio == nullptr || min_nr > nr ||
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

int fake_io_destroy(io_context_t) {
  return g_fake_aio->destroy_result;
}

class FakeAioGuard {
 public:
  explicit FakeAioGuard(FakeAioState *state)
      : loader_(LibAioLoader::Instance()),
        original_submit_(loader_.io_submit),
        original_getevents_(loader_.io_getevents),
        original_destroy_(loader_.io_destroy) {
    g_fake_aio = state;
    loader_.io_submit = fake_io_submit;
    loader_.io_getevents = fake_io_getevents;
    loader_.io_destroy = fake_io_destroy;
  }

  ~FakeAioGuard() {
    loader_.io_submit = original_submit_;
    loader_.io_getevents = original_getevents_;
    loader_.io_destroy = original_destroy_;
    g_fake_aio = nullptr;
  }

 private:
  LibAioLoader &loader_;
  aio_submit_fn original_submit_;
  aio_getevents_fn original_getevents_;
  aio_destroy_fn original_destroy_;
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

  auto &memory_pool = ::zvec::ailego::PoolPayloadBudget::get_instance();
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

  auto &memory_pool = ::zvec::ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      0, memory_pool.init(
             2 * page_size +
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

  auto &memory_pool = ::zvec::ailego::PoolPayloadBudget::get_instance();
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

TEST(DiskAnnLinuxAioTest, FailedCancellationRetainsOnlyKernelStaging) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  const auto source = make_source();
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));
  std::unique_ptr<void, decltype(&std::free)> output(
      allocate_aligned(source.size()), &std::free);
  ASSERT_NE(output, nullptr);
  std::memset(output.get(), 0, source.size());
  auto requests = make_requests(output.get());
  FakeAioState state;
  state.submit_results = {4};
  state.completion_results = {-EIO};
  FakeAioGuard guard(&state);
  LinuxAlignedFileReader reader(::dup(file.fd()));
  IOContext ctx = new IoBackend();
  ctx->type = ailego::IOBackendType::kLibAio;
  ctx->aio_ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  ASSERT_NE(nullptr, ctx->aio_staging);
  std::vector<uint32_t> completed;
  EXPECT_EQ(IndexError_Runtime, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_TRUE(completed.empty());
  EXPECT_TRUE(ctx->aio_quarantined);
  ASSERT_NE(nullptr, ctx->aio_staging);
  ASSERT_EQ(kBlockCount, state.kernel_buffers.size());
  for (size_t i = 0; i < kBlockCount; ++i) {
    EXPECT_NE(requests[i].buf, state.kernel_buffers[i]);
    std::memset(state.kernel_buffers[i], 0xa5, kBlockSize);
  }
  EXPECT_EQ(0, reader.submit(batch, requests, ctx));
  EXPECT_TRUE(batch.used_pread);
  EXPECT_EQ(0, std::memcmp(output.get(), source.data(), source.size()));
  EXPECT_EQ(1u, state.submit_call);
  state.destroy_result = 0;
  EXPECT_EQ(0, destroy_io_ctx(ctx));
}

TEST(DiskAnnLinuxAioTest, RealAioRingAndStagingReturnTheirBudget) {
  auto &loader = LibAioLoader::Instance();
  if (!loader.load()) GTEST_SKIP() << "libaio is unavailable";
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto reset = ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(0)); });
  auto context = std::make_unique<IoBackend>();
  context->type = ailego::IOBackendType::kLibAio;
  ASSERT_EQ(0, loader.setup_budgeted(16, context->aio_ctx, context->aio_memory));
  EXPECT_GE(context->aio_memory.bytes, pool.page_buffer_size());

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  const auto source = make_source();
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));
  std::unique_ptr<void, decltype(&std::free)> output(
      allocate_aligned(source.size()), &std::free);
  ASSERT_NE(output, nullptr);
  auto requests = make_requests(output.get());
  LinuxAlignedFileReader reader(::dup(file.fd()));
  const size_t external_before = pool.external_used();
  auto *ctx = context.get();
  {
    PendingBatch batch;
    ASSERT_EQ(0, reader.submit(batch, requests, ctx));
    ASSERT_FALSE(batch.used_pread);
    const size_t in_flight = pool.external_used();
    EXPECT_GT(in_flight, external_before);
    std::vector<uint32_t> completed;
    const int result = reader.get_completed(batch, ctx, kBlockCount, completed);
    EXPECT_EQ(static_cast<int>(kBlockCount), result);
    EXPECT_LT(pool.external_used(), in_flight);
  }
  EXPECT_EQ(external_before, pool.external_used());
  EXPECT_EQ(0, std::memcmp(output.get(), source.data(), source.size()));
  EXPECT_EQ(0, loader.destroy_budgeted(ctx->aio_ctx, ctx->aio_memory));
  EXPECT_EQ(nullptr, ctx->aio_ctx);
  EXPECT_EQ(0u, ctx->aio_memory.bytes);
}

#endif  // __linux__
