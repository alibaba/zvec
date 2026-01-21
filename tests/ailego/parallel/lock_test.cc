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

#include <mutex>
#include <ailego/parallel/lock.h>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_pool.h>

using namespace zvec;

TEST(SpinMutex, General) {
  ailego::SpinMutex mutex;
  { std::unique_lock<ailego::SpinMutex> signal_lock1(mutex); }
  { std::lock_guard<ailego::SpinMutex> signal_lock2(mutex); }

  ailego::SpinMutex mutex2;
  int result = std::try_lock(mutex, mutex2);
  if (result == -1) {
    mutex.unlock();
    mutex2.unlock();
  }
}

TEST(WriteLock, General) {
  ailego::SharedMutex mutex;
  ailego::WriteLock wrlock(mutex);
  { std::unique_lock<ailego::WriteLock> signal_lock1(wrlock); }
  { std::lock_guard<ailego::WriteLock> signal_lock2(wrlock); }
}

TEST(ReadLock, General) {
  ailego::SharedMutex mutex;
  ailego::ReadLock rdlock(mutex);
  { std::unique_lock<ailego::ReadLock> signal_lock1(rdlock); }
  { std::lock_guard<ailego::ReadLock> signal_lock2(rdlock); }
}

TEST(Mutex, General) {
  ailego::ThreadPool pool;
  std::mutex mutex;

  int count = 0;
  for (int i = 0; i < 2000; ++i) {
    pool.execute([&]() {
      std::lock_guard<std::mutex> lock(mutex);
      ++count;
    });
  }
  pool.wait_finish();
  EXPECT_EQ(2000, count);
}

class NoLockTest {
 public:
  virtual void open() {
    no_lock_opened_ = true;
  }
  virtual int read(volatile int *count) {
    if (!no_lock_opened_) {
      return -1;
    }
    (*count)++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  virtual int write(volatile int *count) {
    if (!no_lock_opened_) {
      return -1;
    }
    (*count)++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  virtual void close() {
    no_lock_opened_ = false;
  }

 private:
  volatile bool no_lock_opened_{false};
};

class AtomicLockTest : public NoLockTest {
 public:
  void open() override {
    opened_.store(true);
    NoLockTest::open();
  }
  int read(volatile int *count) override {
    AILEGO_SAFE_ACCESS(-1);
    return NoLockTest::read(count);
  }
  int write(volatile int *count) override {
    AILEGO_SAFE_ACCESS(-1);
    return NoLockTest::write(count);
  }
  void close() override {
    AILEGO_SAFE_CLOSE;
    NoLockTest::close();
  }

 private:
  mutable std::atomic<int> counter_{0};
  std::atomic<bool> opened_{false};
};

void test_lock(NoLockTest &test_obj) {
  ailego::ThreadPool pool;
  test_obj.open();

  auto start = std::chrono::system_clock::now();

  const int kTestCount = 10000;
  volatile int count = 0;
  for (int i = 0; i < kTestCount; ++i) {
    pool.execute([&]() {
      test_obj.read(&count);
      test_obj.write(&count);
    });
  }
  test_obj.close();
  pool.wait_finish();

  auto stop = std::chrono::system_clock::now();
  std::chrono::nanoseconds time_used = stop - start;
  std::cout << "use: " << time_used.count() / 1000 << " us" << std::endl;

  std::cout << "count: " << count << std::endl;
  EXPECT_LE(count, kTestCount * 2);
}

TEST(CloseLock, Perf) {
  std::cout << "NoLockTest" << std::endl;
  NoLockTest test_obj;
  test_lock(test_obj);

  std::cout << "AtomicLockTest" << std::endl;
  AtomicLockTest test_obj3;
  test_lock(test_obj3);
}
