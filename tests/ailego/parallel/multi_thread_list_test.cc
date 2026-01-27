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

#include <iostream>
#include <memory>
#include <zvec/ailego/parallel/thread_pool.h>

#define private public
#include <ailego/parallel/multi_thread_list.h>
#undef private

#include <gtest/gtest.h>

using namespace zvec;
using namespace zvec::ailego;
using namespace std;

struct Item {
  uint32_t a_;
  std::string b_;
  Item() {};
  Item(uint32_t a, std::string b) : a_(a), b_(b) {}
};

MultiThreadList<Item> mt_queue(100);

void producer(uint32_t i) {
  Item item{i, std::to_string(i)};
  mt_queue.produce(item);
  return;
}
void consumer(uint32_t i, uint32_t *result) {
  Item item;
  while (mt_queue.consume(&item)) {
    *result += item.a_;
  }
}
void producer_done(uint32_t i) {
  Item item{i, std::to_string(i)};
  EXPECT_EQ(false, mt_queue.produce(item));
  return;
}

TEST(MultiThreadListTest, General) {
  int times = 100;
  while (times--) {
    cout << "================================" << endl;
    cout << "times: " << times << endl;

    mt_queue.reset();

    ailego::ThreadPool producer_pool;
    ailego::ThreadPool consumer_pool;
    ailego::ThreadPool producer_done_pool;

    uint32_t num_of_consumer = 100;
    uint32_t num_of_producer = 100;
    uint32_t num_of_producer_done = 100;

    uint32_t consumer_results[num_of_consumer];
    memset(consumer_results, 0, sizeof(uint32_t) * num_of_consumer);

    for (uint32_t i = 0; i < num_of_consumer; i++) {
      consumer_pool.execute(consumer, i + 1, &consumer_results[i]);
    }

    for (uint32_t i = 0; i < num_of_producer; i++) {
      producer_pool.execute(producer, i + 1);
    }

    producer_pool.wait_finish();
    mt_queue.done();
    consumer_pool.wait_finish();

    // produce after queue done
    for (uint32_t i = 0; i < num_of_producer_done; i++) {
      producer_done_pool.execute(producer_done, i + 1);
    }
    producer_done_pool.wait_finish();

    uint32_t total = 0;
    for (uint32_t i = 0; i < num_of_consumer; i++) {
      cout << consumer_results[i] << " ";
      total += consumer_results[i];
    }
    cout << endl;

    EXPECT_EQ(total, 5050);
  }
}

TEST(MultiThreadListTest, FullQueueQuit) {
  mt_queue.reset();

  ailego::ThreadPool producer_pool;

  uint32_t num_of_producer = 1000;

  for (uint32_t i = 1; i <= num_of_producer; i++) {
    producer_pool.execute(producer, i);
  }

  mt_queue.done();
  producer_pool.wait_finish();
}

TEST(MultiThreadListTest, ConsumeStopResume) {
  mt_queue.reset();

  ailego::ThreadPool producer_pool;
  ailego::ThreadPool consumer_pool;

  uint32_t num_of_consumer = 100;
  uint32_t num_of_producer = 100;

  uint32_t consumer_results[2 * num_of_consumer];
  memset(consumer_results, 0, sizeof(uint32_t) * 2 * num_of_consumer);

  for (uint32_t i = 0; i < num_of_consumer; i++) {
    consumer_pool.execute(consumer, i + 1, &consumer_results[i]);
  }

  for (uint32_t i = 0; i < num_of_producer; i++) {
    producer_pool.execute(producer, i + 1);
  }

  producer_pool.wait_finish();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  mt_queue.stop_consume();
  consumer_pool.wait_finish();

  uint32_t total = 0;
  for (uint32_t i = 0; i < num_of_consumer; i++) {
    cout << consumer_results[i] << " ";
    total += consumer_results[i];
  }
  cout << endl;

  cout << "mt queue size: " << mt_queue.list_.size() << endl;

  EXPECT_EQ(total, 5050);

  for (uint32_t i = num_of_producer; i < 2 * num_of_producer; i++) {
    producer_pool.execute(producer, i + 1);
  }

  mt_queue.resume_consume();

  for (uint32_t i = num_of_producer; i < 2 * num_of_consumer; i++) {
    consumer_pool.execute(consumer, i + 1, &consumer_results[i]);
  }

  producer_pool.wait_finish();
  mt_queue.done();
  consumer_pool.wait_finish();

  total = 0;
  for (uint32_t i = num_of_consumer; i < 2 * num_of_consumer; i++) {
    cout << consumer_results[i] << " ";
    total += consumer_results[i];
  }
  cout << endl;

  cout << "mt queue size: " << mt_queue.list_.size() << endl;

  EXPECT_EQ(total, 15050);
}

struct MoveableItem {
  uint32_t a_;
  std::string b_;
  MoveableItem() {};
  MoveableItem(uint32_t a, std::string b) : a_(a), b_(b) {}

  MoveableItem(const MoveableItem &) = delete;
  MoveableItem &operator=(const MoveableItem &) = delete;

  MoveableItem(MoveableItem &&) = default;
  MoveableItem &operator=(MoveableItem &&) = default;
};

MultiThreadList<MoveableItem> mt_moveable_queue(100);

void producer_moveable(uint32_t i) {
  MoveableItem item{i, std::to_string(i)};
  mt_moveable_queue.produce(std::move(item));
  return;
}
void consumer_moveable(uint32_t i, uint32_t *result) {
  MoveableItem item;
  while (mt_moveable_queue.consume(&item)) {
    *result += item.a_;
  }
}
void producer_moveable_done(uint32_t i) {
  MoveableItem item{i, std::to_string(i)};
  EXPECT_EQ(false, mt_moveable_queue.produce(std::move(item)));
  return;
}

TEST(MultiThreadListTest, General_Moveable) {
  int times = 100;
  while (times--) {
    cout << "================================" << endl;
    cout << "times: " << times << endl;

    mt_moveable_queue.reset();

    ailego::ThreadPool producer_pool;
    ailego::ThreadPool consumer_pool;
    ailego::ThreadPool producer_done_pool;

    uint32_t num_of_consumer = 100;
    uint32_t num_of_producer = 100;
    uint32_t num_of_producer_done = 100;

    uint32_t consumer_results[num_of_consumer];
    memset(consumer_results, 0, sizeof(uint32_t) * num_of_consumer);

    for (uint32_t i = 0; i < num_of_consumer; i++) {
      consumer_pool.execute(consumer_moveable, i + 1, &consumer_results[i]);
    }

    for (uint32_t i = 0; i < num_of_producer; i++) {
      producer_pool.execute(producer_moveable, i + 1);
    }

    producer_pool.wait_finish();
    mt_moveable_queue.done();
    consumer_pool.wait_finish();

    // produce after queue done
    for (uint32_t i = 0; i < num_of_producer_done; i++) {
      producer_done_pool.execute(producer_moveable_done, i + 1);
    }
    producer_done_pool.wait_finish();

    uint32_t total = 0;
    for (uint32_t i = 0; i < num_of_consumer; i++) {
      cout << consumer_results[i] << " ";
      total += consumer_results[i];
    }
    cout << endl;

    EXPECT_EQ(total, 5050);
  }
}
