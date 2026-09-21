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

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <ailego/io/libaio_loader.h>
#include <gtest/gtest.h>

namespace {

class WorkerAioContext {
 public:
  ~WorkerAioContext() {
    if (ctx_ == nullptr || LibAioLoader::Instance().io_destroy(ctx_) != 0) {
      std::_Exit(EXIT_FAILURE);
    }
    std::fputs("libaio context destroyed at shutdown\n", stderr);
  }

  io_context_t ctx_{nullptr};
};

// Mirror GlobalResource: its thread pool is constructed before libaio is
// loaded, but its workers retain AIO contexts until shutdown.
class ShutdownWorker {
 public:
  void start() {
    auto ready = ready_.get_future();
    worker_ = std::thread([this] {
      // Keep cleanup on the worker stack: on musl, libstdc++ can delete its
      // TLS destructor key before ShutdownWorker joins this thread. A
      // thread_local destructor would then be skipped by the runtime.
      WorkerAioContext context;
      ready_.set_value(LibAioLoader::Instance().io_setup(1, &context.ctx_));
      stop_.get_future().wait();
    });
    if (ready.get() != 0) {
      std::_Exit(EXIT_FAILURE);
    }
  }

  ~ShutdownWorker() {
    stop_.set_value();
    worker_.join();
  }

 private:
  std::promise<int> ready_;
  std::promise<void> stop_;
  std::thread worker_;
};

[[noreturn]] void exit_with_worker_context() {
  // Registration order matters: the old loader destructor runs before the
  // worker destructor and unloads the code needed by its io_destroy().
  static ShutdownWorker worker;
  if (!LibAioLoader::Instance().load()) {
    std::_Exit(EXIT_FAILURE);
  }
  worker.start();
  std::exit(EXIT_SUCCESS);
}

}  // namespace

TEST(DiskAnnLibAioDeathTest, WorkerContextOutlivesStaticDestructors) {
  // Check the optional dependency without constructing the loader singleton.
  void *handle = dlopen("libaio.so.1", RTLD_LAZY);
  if (handle == nullptr) {
    handle = dlopen("libaio.so.1t64", RTLD_LAZY);
  }
  if (handle == nullptr) {
    GTEST_SKIP() << "libaio is not installed";
  }
  dlclose(handle);

  // Re-exec to keep singleton initialization independent of other tests.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(exit_with_worker_context(), ::testing::ExitedWithCode(0),
              "libaio context destroyed at shutdown");
}
