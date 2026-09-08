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
#include <gtest/gtest.h>
#include "diskann_file_reader.h"

namespace zvec {
namespace core {
int execute_io_libaio(io_context_t &ctx, int fd,
                      std::vector<AlignedRead> &read_reqs,
                      uint64_t n_retries = 0);
}  // namespace core
}  // namespace zvec

using namespace zvec::core;

namespace {

constexpr size_t kBlockSize = 512;
constexpr size_t kBlockCount = 4;

struct FakeAioState {
  std::vector<int> submit_results;
  std::vector<int> completion_results;
  std::vector<long> submit_sizes;
  std::vector<long> completion_sizes;
  std::vector<struct iocb *> submitted;
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
  }
  return ret;
}

int fake_io_getevents(io_context_t, long min_nr, long nr,
                      struct io_event *events, struct timespec *) {
  if (g_fake_aio == nullptr || min_nr != nr ||
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

class FakeAioGuard {
 public:
  explicit FakeAioGuard(FakeAioState *state)
      : loader_(LibAioLoader::Instance()),
        original_submit_(loader_.io_submit),
        original_getevents_(loader_.io_getevents) {
    g_fake_aio = state;
    loader_.io_submit = fake_io_submit;
    loader_.io_getevents = fake_io_getevents;
  }

  ~FakeAioGuard() {
    loader_.io_submit = original_submit_;
    loader_.io_getevents = original_getevents_;
    g_fake_aio = nullptr;
  }

 private:
  LibAioLoader &loader_;
  aio_submit_fn original_submit_;
  aio_getevents_fn original_getevents_;
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
  io_context_t ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

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
  io_context_t ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

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
  io_context_t ctx = reinterpret_cast<io_context_t>(static_cast<uintptr_t>(1));

  EXPECT_EQ(execute_io_libaio(ctx, file.fd(), requests), 0);
  EXPECT_EQ(state.completion_sizes, (std::vector<long>{4, 3, 2}));
  EXPECT_EQ(state.completed, kBlockCount);
  EXPECT_EQ(std::memcmp(output, source.data(), source.size()), 0);

  std::free(output);
}
