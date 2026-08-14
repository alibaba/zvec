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

#include <algorithm>
#include <atomic>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>

using namespace zvec;
using namespace zvec::core;

class BufferStorageWriteTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Initialize the memory limit pool with 64MB - enough for all tests.
    ailego::MemoryLimitPool::get_instance().init(64 * 1024UL * 1024UL);
  }

  void SetUp() override {
    file_path_ = "buffer_storage_write_test_dir/test_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this));
    ailego::File::Delete(file_path_);
    ailego::File::MakePath("buffer_storage_write_test_dir");
  }

  void TearDown() override {
    ailego::File::Delete(file_path_);
  }

  // Open BufferStorage in writable mode (create_if_missing=true)
  IndexStorage::Pointer OpenWritable() {
    auto storage = IndexFactory::CreateStorage("BufferStorage");
    if (!storage) return nullptr;
    ailego::Params params;
    storage->init(params);
    if (storage->open(file_path_, true) != 0) return nullptr;
    return storage;
  }

  // Open BufferStorage in read-only mode
  IndexStorage::Pointer OpenReadOnly() {
    auto storage = IndexFactory::CreateStorage("BufferStorage");
    if (!storage) return nullptr;
    ailego::Params params;
    storage->init(params);
    if (storage->open(file_path_, false) != 0) return nullptr;
    return storage;
  }

  std::string file_path_;
};

// ===== Basic Write Tests =====

// Test: Create new index via BufferStorage, append segment, write data, read
// back
TEST_F(BufferStorageWriteTest, WriteBasicCreateAndWrite) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  std::string data = "Hello BufferStorage Write!";
  EXPECT_EQ(data.size(), seg->write(0, data.data(), data.size()));

  // Verify data via fetch
  std::vector<char> buf(data.size());
  EXPECT_EQ(data.size(), seg->fetch(0, buf.data(), buf.size()));
  EXPECT_EQ(data, std::string(buf.data(), buf.size()));

  // data_size should reflect the written bytes
  EXPECT_EQ(data.size(), seg->data_size());
  EXPECT_EQ(0, storage->close());
}

// Test: Write at non-zero offset within the segment
TEST_F(BufferStorageWriteTest, WriteAtNonZeroOffset) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 8192));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  // First write at offset 0
  std::string first = "AAAA";
  EXPECT_EQ(first.size(), seg->write(0, first.data(), first.size()));

  // Second write at offset 100
  std::string second = "BBBB";
  EXPECT_EQ(second.size(), seg->write(100, second.data(), second.size()));

  // data_size should be max(first.end, second.end) = 104
  EXPECT_EQ(104u, seg->data_size());

  // Verify both writes
  std::vector<char> buf1(first.size());
  EXPECT_EQ(first.size(), seg->fetch(0, buf1.data(), buf1.size()));
  EXPECT_EQ(first, std::string(buf1.data(), buf1.size()));

  std::vector<char> buf2(second.size());
  EXPECT_EQ(second.size(), seg->fetch(100, buf2.data(), buf2.size()));
  EXPECT_EQ(second, std::string(buf2.data(), buf2.size()));

  EXPECT_EQ(0, storage->close());
}

// Test: Write to multiple independent segments
TEST_F(BufferStorageWriteTest, WriteMultipleSegments) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg_a", 4096));
  ASSERT_EQ(0, storage->append("seg_b", 4096));
  ASSERT_EQ(0, storage->append("seg_c", 4096));

  auto seg_a = storage->get("seg_a");
  auto seg_b = storage->get("seg_b");
  auto seg_c = storage->get("seg_c");
  ASSERT_TRUE(seg_a);
  ASSERT_TRUE(seg_b);
  ASSERT_TRUE(seg_c);

  std::string da = "data_for_a";
  std::string db = "data_for_b_longer";
  std::string dc = "c";

  EXPECT_EQ(da.size(), seg_a->write(0, da.data(), da.size()));
  EXPECT_EQ(db.size(), seg_b->write(0, db.data(), db.size()));
  EXPECT_EQ(dc.size(), seg_c->write(0, dc.data(), dc.size()));

  // Verify independently
  std::vector<char> buf(db.size());
  EXPECT_EQ(da.size(), seg_a->fetch(0, buf.data(), da.size()));
  EXPECT_EQ(da, std::string(buf.data(), da.size()));

  EXPECT_EQ(db.size(), seg_b->fetch(0, buf.data(), db.size()));
  EXPECT_EQ(db, std::string(buf.data(), db.size()));

  EXPECT_EQ(dc.size(), seg_c->fetch(0, buf.data(), dc.size()));
  EXPECT_EQ(dc, std::string(buf.data(), dc.size()));

  EXPECT_EQ(0, storage->close());
}

// Test: Overwrite existing data at the same offset
TEST_F(BufferStorageWriteTest, WriteOverwrite) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  std::string first = "XXXXXXXX";
  EXPECT_EQ(first.size(), seg->write(0, first.data(), first.size()));

  std::string second = "YYYYYYYY";
  EXPECT_EQ(second.size(), seg->write(0, second.data(), second.size()));

  // Second write should overwrite
  std::vector<char> buf(second.size());
  EXPECT_EQ(second.size(), seg->fetch(0, buf.data(), buf.size()));
  EXPECT_EQ(second, std::string(buf.data(), buf.size()));

  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, WriteBatchSamePagePersists) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("batch_seg", 8192));
  auto segment = storage->get("batch_seg");
  ASSERT_TRUE(segment);

  const size_t base_in_page = segment->data_offset() % ailego::kVectorPageSize;
  const size_t offset = base_in_page + 128 <= ailego::kVectorPageSize
                            ? 0
                            : ailego::kVectorPageSize - base_in_page;
  const uint32_t payload = 0x12345678u;
  const uint32_t published_count = 7;
  const IndexStorage::SegmentData writes[] = {
      {offset + 64, sizeof(payload), &payload},
      {offset, sizeof(published_count), &published_count},
  };
  ASSERT_TRUE(segment->write_batch(writes, 2));
  EXPECT_EQ(offset + 64 + sizeof(payload), segment->data_size());

  uint32_t got_count = 0;
  uint32_t got_payload = 0;
  EXPECT_EQ(sizeof(got_count),
            segment->fetch(offset, &got_count, sizeof(got_count)));
  EXPECT_EQ(sizeof(got_payload),
            segment->fetch(offset + 64, &got_payload, sizeof(got_payload)));
  EXPECT_EQ(published_count, got_count);
  EXPECT_EQ(payload, got_payload);

  ASSERT_EQ(0, storage->close());
  storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  segment = storage->get("batch_seg");
  ASSERT_TRUE(segment);
  got_count = 0;
  got_payload = 0;
  EXPECT_EQ(sizeof(got_count),
            segment->fetch(offset, &got_count, sizeof(got_count)));
  EXPECT_EQ(sizeof(got_payload),
            segment->fetch(offset + 64, &got_payload, sizeof(got_payload)));
  EXPECT_EQ(published_count, got_count);
  EXPECT_EQ(payload, got_payload);
  EXPECT_EQ(0, storage->close());
}

// ===== Boundary / Error Tests =====

// Test: Write exceeding segment capacity returns 0
TEST_F(BufferStorageWriteTest, WriteExceedsCapacity) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  // Append a small segment (page-aligned, so at least 4096 bytes capacity)
  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  size_t cap = seg->capacity();
  ASSERT_GT(cap, 0u);

  // Write at an offset that causes overflow: offset + len > capacity
  std::vector<char> big_data(cap + 1, 'Z');
  EXPECT_EQ(0u, seg->write(0, big_data.data(), big_data.size()));

  // Write at offset that exceeds capacity
  std::string small = "small";
  EXPECT_EQ(0u, seg->write(cap + 1, small.data(), small.size()));

  EXPECT_EQ(0, storage->close());
}

// Test: Write with zero length (edge case)
TEST_F(BufferStorageWriteTest, WriteZeroLength) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  // Writing zero bytes should succeed (no-op but valid)
  EXPECT_EQ(0u, seg->write(0, "x", 0));
  EXPECT_EQ(0u, seg->data_size());

  EXPECT_EQ(0, storage->close());
}

// ===== Persistence Tests =====

// Test: Write, flush, close, reopen, verify data persisted
TEST_F(BufferStorageWriteTest, WriteFlushReopenVerify) {
  std::string data = "Persistent data that survives close/reopen";

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("persist_seg", 8192));
    auto seg = storage->get("persist_seg");
    ASSERT_TRUE(seg);
    EXPECT_EQ(data.size(), seg->write(0, data.data(), data.size()));
    EXPECT_EQ(0, storage->flush());
    EXPECT_EQ(0, storage->close());
  }

  // Reopen in read-only mode and verify
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("persist_seg");
    ASSERT_TRUE(seg);
    EXPECT_EQ(data.size(), seg->data_size());

    std::vector<char> buf(data.size());
    EXPECT_EQ(data.size(), seg->fetch(0, buf.data(), buf.size()));
    EXPECT_EQ(data, std::string(buf.data(), buf.size()));
    EXPECT_EQ(0, storage->close());
  }
}

// Test: Multiple write-flush cycles persist all data
TEST_F(BufferStorageWriteTest, WriteMultipleFlushCycles) {
  std::string data1 = "first_write";
  std::string data2 = "second_write_longer";

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);

    // First write + flush
    EXPECT_EQ(data1.size(), seg->write(0, data1.data(), data1.size()));
    EXPECT_EQ(0, storage->flush());

    // Second write at a different offset + flush
    EXPECT_EQ(data2.size(), seg->write(200, data2.data(), data2.size()));
    EXPECT_EQ(0, storage->flush());
    EXPECT_EQ(0, storage->close());
  }

  // Verify persistence
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);

    std::vector<char> buf1(data1.size());
    EXPECT_EQ(data1.size(), seg->fetch(0, buf1.data(), buf1.size()));
    EXPECT_EQ(data1, std::string(buf1.data(), buf1.size()));

    std::vector<char> buf2(data2.size());
    EXPECT_EQ(data2.size(), seg->fetch(200, buf2.data(), buf2.size()));
    EXPECT_EQ(data2, std::string(buf2.data(), buf2.size()));

    EXPECT_EQ(0, storage->close());
  }
}

// Test: Close without explicit flush still persists (close_index does flush)
TEST_F(BufferStorageWriteTest, WriteCloseWithoutExplicitFlush) {
  std::string data = "should_persist_on_close";

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    EXPECT_EQ(data.size(), seg->write(0, data.data(), data.size()));
    // No explicit flush - close should handle it
    EXPECT_EQ(0, storage->close());
  }

  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    std::vector<char> buf(data.size());
    EXPECT_EQ(data.size(), seg->fetch(0, buf.data(), buf.size()));
    EXPECT_EQ(data, std::string(buf.data(), buf.size()));
    EXPECT_EQ(0, storage->close());
  }
}

// ===== Read-Only Behavior =====

// Test: Write to read-only storage is a silent no-op (returns len)
TEST_F(BufferStorageWriteTest, WriteReadOnlyNoOp) {
  // First create an index file with a segment
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    std::string init_data = "initial";
    seg->write(0, init_data.data(), init_data.size());
    EXPECT_EQ(0, storage->flush());
    EXPECT_EQ(0, storage->close());
  }

  // Open read-only and attempt write
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);

    std::string new_data = "overwrite_attempt";
    // Should return len (silent no-op)
    EXPECT_EQ(new_data.size(), seg->write(0, new_data.data(), new_data.size()));

    // Data should remain unchanged (still "initial")
    std::vector<char> buf(7);
    EXPECT_EQ(7u, seg->fetch(0, buf.data(), 7));
    EXPECT_EQ("initial", std::string(buf.data(), 7));

    EXPECT_EQ(0, storage->close());
  }
}

// ===== Resize Tests =====

// Test: Resize increases data_size without writing
TEST_F(BufferStorageWriteTest, ResizeGrow) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  EXPECT_EQ(0u, seg->data_size());
  size_t new_size = seg->resize(512);
  EXPECT_EQ(512u, new_size);
  EXPECT_EQ(512u, seg->data_size());
  EXPECT_EQ(seg->capacity() - 512, seg->padding_size());

  EXPECT_EQ(0, storage->close());
}

// Test: Resize shrinks data_size
TEST_F(BufferStorageWriteTest, ResizeShrink) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  // Write to grow data_size to 100
  std::vector<char> buf(100, 'X');
  seg->write(0, buf.data(), buf.size());
  EXPECT_EQ(100u, seg->data_size());

  // Resize to smaller
  size_t new_size = seg->resize(50);
  EXPECT_EQ(50u, new_size);
  EXPECT_EQ(50u, seg->data_size());

  EXPECT_EQ(0, storage->close());
}

// Test: Resize beyond capacity is clamped
TEST_F(BufferStorageWriteTest, ResizeBeyondCapacityClamped) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  size_t cap = seg->capacity();
  size_t result = seg->resize(cap + 1000);
  EXPECT_EQ(cap, result);
  EXPECT_EQ(cap, seg->data_size());
  EXPECT_EQ(0u, seg->padding_size());

  EXPECT_EQ(0, storage->close());
}

// ===== CRC Tests =====

// Test: update_data_crc reflects in data_crc() getter
TEST_F(BufferStorageWriteTest, UpdateDataCrc) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  uint32_t new_crc = 0xDEADBEEF;
  seg->update_data_crc(new_crc);
  EXPECT_EQ(new_crc, seg->data_crc());

  EXPECT_EQ(0, storage->close());
}

// Test: CRC persists after flush and reopen
TEST_F(BufferStorageWriteTest, UpdateDataCrcPersistence) {
  uint32_t crc_val = 0x12345678;
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    std::string data = "crc_test_data";
    seg->write(0, data.data(), data.size());
    seg->update_data_crc(crc_val);
    EXPECT_EQ(0, storage->flush());
    EXPECT_EQ(0, storage->close());
  }

  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    EXPECT_EQ(crc_val, seg->data_crc());
    EXPECT_EQ(0, storage->close());
  }
}

// ===== Concurrency Tests =====

// Test: Multiple threads writing to different segments concurrently
TEST_F(BufferStorageWriteTest, ConcurrentWriteDifferentSegments) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  const int kNumSegments = 8;
  for (int i = 0; i < kNumSegments; ++i) {
    ASSERT_EQ(0, storage->append("seg_" + std::to_string(i), 16384));
  }

  std::vector<std::thread> threads;
  std::atomic<int> errors{0};

  for (int i = 0; i < kNumSegments; ++i) {
    threads.emplace_back([&, i]() {
      auto seg = storage->get("seg_" + std::to_string(i));
      if (!seg) {
        errors.fetch_add(1);
        return;
      }
      // Each thread writes its own pattern to its own segment
      std::vector<char> data(1024, static_cast<char>('A' + i));
      for (int j = 0; j < 10; ++j) {
        size_t offset = j * 1024;
        if (seg->write(offset, data.data(), data.size()) != data.size()) {
          errors.fetch_add(1);
        }
      }
    });
  }

  for (auto &t : threads) t.join();
  EXPECT_EQ(0, errors.load());

  // Verify each segment's data
  for (int i = 0; i < kNumSegments; ++i) {
    auto seg = storage->get("seg_" + std::to_string(i));
    ASSERT_TRUE(seg);
    // Last write was at offset 9*1024, so data_size >= 10*1024
    EXPECT_GE(seg->data_size(), 10u * 1024u);

    std::vector<char> buf(1024);
    seg->fetch(0, buf.data(), 1024);
    EXPECT_EQ(buf[0], static_cast<char>('A' + i));
  }

  EXPECT_EQ(0, storage->close());
}

// Test: Multiple threads writing to the same segment at different offsets
TEST_F(BufferStorageWriteTest, ConcurrentWriteSameSegment) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  // Need large enough segment for all threads
  ASSERT_EQ(0, storage->append("shared_seg", 65536));
  auto seg = storage->get("shared_seg");
  ASSERT_TRUE(seg);

  const int kNumThreads = 8;
  const size_t kChunkSize = 256;
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      // Each thread writes to its own non-overlapping region
      size_t offset = i * kChunkSize * 10;
      std::vector<char> data(kChunkSize, static_cast<char>('A' + i));
      for (int j = 0; j < 10; ++j) {
        if (seg->write(offset + j * kChunkSize, data.data(), data.size()) !=
            data.size()) {
          errors.fetch_add(1);
        }
      }
    });
  }

  for (auto &t : threads) t.join();
  EXPECT_EQ(0, errors.load());

  // Verify each thread's region
  for (int i = 0; i < kNumThreads; ++i) {
    size_t offset = i * kChunkSize * 10;
    std::vector<char> buf(kChunkSize);
    seg->fetch(offset, buf.data(), kChunkSize);
    for (size_t b = 0; b < kChunkSize; ++b) {
      EXPECT_EQ(buf[b], static_cast<char>('A' + i))
          << "Mismatch at thread " << i << " byte " << b;
    }
  }

  EXPECT_EQ(0, storage->close());
}

// Test: Concurrent writers + flush (simulates real workload)
TEST_F(BufferStorageWriteTest, ConcurrentWriteWithFlush) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 65536));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  std::atomic<bool> stop{false};
  std::atomic<int> write_errors{0};

  // Writer threads
  std::vector<std::thread> writers;
  for (int i = 0; i < 4; ++i) {
    writers.emplace_back([&, i]() {
      std::vector<char> data(128, static_cast<char>('0' + i));
      int iter = 0;
      while (!stop.load(std::memory_order_relaxed) && iter < 100) {
        size_t offset = (i * 128 + (iter % 10) * 128) % 4096;
        if (seg->write(offset, data.data(), data.size()) != data.size()) {
          write_errors.fetch_add(1);
        }
        ++iter;
      }
    });
  }

  // Flush thread
  std::thread flusher([&]() {
    for (int i = 0; i < 5; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      storage->flush();
    }
    stop.store(true);
  });

  for (auto &w : writers) w.join();
  flusher.join();

  EXPECT_EQ(0, write_errors.load());
  EXPECT_EQ(0, storage->close());
}

// ===== Append + Write Integration =====

// Test: Append multiple segments then write to each
TEST_F(BufferStorageWriteTest, AppendThenWriteSequence) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  for (int i = 0; i < 5; ++i) {
    std::string seg_name = "seg_" + std::to_string(i);
    ASSERT_EQ(0, storage->append(seg_name, 4096));
    auto seg = storage->get(seg_name);
    ASSERT_TRUE(seg);

    std::string data = "content_of_segment_" + std::to_string(i);
    EXPECT_EQ(data.size(), seg->write(0, data.data(), data.size()));
  }

  // Verify all segments have correct data
  for (int i = 0; i < 5; ++i) {
    std::string seg_name = "seg_" + std::to_string(i);
    auto seg = storage->get(seg_name);
    ASSERT_TRUE(seg);
    std::string expected = "content_of_segment_" + std::to_string(i);
    std::vector<char> buf(expected.size());
    EXPECT_EQ(expected.size(), seg->fetch(0, buf.data(), buf.size()));
    EXPECT_EQ(expected, std::string(buf.data(), buf.size()));
  }

  EXPECT_EQ(0, storage->close());
}

// Test: Write to a segment, append another, write to both, verify all
TEST_F(BufferStorageWriteTest, InterleavedAppendAndWrite) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  // Append and write first segment
  ASSERT_EQ(0, storage->append("seg1", 4096));
  auto seg1 = storage->get("seg1");
  ASSERT_TRUE(seg1);
  std::string d1 = "first_data";
  EXPECT_EQ(d1.size(), seg1->write(0, d1.data(), d1.size()));

  // Append second segment (triggers flush_index internally)
  ASSERT_EQ(0, storage->append("seg2", 4096));
  auto seg2 = storage->get("seg2");
  ASSERT_TRUE(seg2);
  std::string d2 = "second_data";
  EXPECT_EQ(d2.size(), seg2->write(0, d2.data(), d2.size()));

  // Re-get seg1 (pointer stability) and write more
  auto seg1_again = storage->get("seg1");
  ASSERT_TRUE(seg1_again);
  std::string d1_extra = "extra";
  EXPECT_EQ(d1_extra.size(),
            seg1_again->write(d1.size(), d1_extra.data(), d1_extra.size()));

  // Verify all data
  std::vector<char> buf(d1.size() + d1_extra.size());
  EXPECT_EQ(buf.size(), seg1_again->fetch(0, buf.data(), buf.size()));
  EXPECT_EQ(d1 + d1_extra, std::string(buf.data(), buf.size()));

  std::vector<char> buf2(d2.size());
  EXPECT_EQ(d2.size(), seg2->fetch(0, buf2.data(), buf2.size()));
  EXPECT_EQ(d2, std::string(buf2.data(), buf2.size()));

  EXPECT_EQ(0, storage->close());
}

// ===== Large Write Tests =====

// Test: Fill entire segment capacity with data
TEST_F(BufferStorageWriteTest, WriteLargeBuffer) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  // Request 16KB segment (will be page-aligned)
  ASSERT_EQ(0, storage->append("big_seg", 16384));
  auto seg = storage->get("big_seg");
  ASSERT_TRUE(seg);

  size_t cap = seg->capacity();
  ASSERT_GE(cap, 16384u);

  // Fill with a pattern
  std::vector<char> data(cap);
  std::iota(data.begin(), data.end(), static_cast<char>(0));
  EXPECT_EQ(cap, seg->write(0, data.data(), data.size()));
  EXPECT_EQ(cap, seg->data_size());
  EXPECT_EQ(0u, seg->padding_size());

  // Verify a portion
  std::vector<char> verify(1024);
  EXPECT_EQ(1024u, seg->fetch(0, verify.data(), 1024));
  EXPECT_EQ(0, std::memcmp(data.data(), verify.data(), 1024));

  EXPECT_EQ(0, storage->close());
}

// Test: Large write persistence across close/reopen
TEST_F(BufferStorageWriteTest, WriteLargeBufferPersistence) {
  const size_t kSize = 8192;
  std::vector<char> data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    data[i] = static_cast<char>(i % 256);
  }

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("large_seg", kSize));
    auto seg = storage->get("large_seg");
    ASSERT_TRUE(seg);
    EXPECT_EQ(kSize, seg->write(0, data.data(), data.size()));
    EXPECT_EQ(0, storage->close());
  }

  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("large_seg");
    ASSERT_TRUE(seg);
    EXPECT_EQ(kSize, seg->data_size());

    std::vector<char> buf(kSize);
    EXPECT_EQ(kSize, seg->fetch(0, buf.data(), kSize));
    EXPECT_EQ(0, std::memcmp(data.data(), buf.data(), kSize));
    EXPECT_EQ(0, storage->close());
  }
}

// ===== Refresh / Checkpoint Tests =====

// Test: refresh() updates checkpoint and marks dirty
TEST_F(BufferStorageWriteTest, RefreshCheckpoint) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->append("seg1", 4096));

  storage->refresh(42);
  EXPECT_EQ(0, storage->flush());

  // After flush the check_point should be >= 42
  EXPECT_GE(storage->check_point(), 42u);

  // Increasing checkpoint
  storage->refresh(100);
  EXPECT_EQ(0, storage->flush());
  EXPECT_GE(storage->check_point(), 100u);

  EXPECT_EQ(0, storage->close());
}

// ===== Duplicate / Error Handling =====

// Test: Appending a duplicate segment ID returns error
TEST_F(BufferStorageWriteTest, AppendDuplicateSegment) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("dup_seg", 4096));
  // Second append with same ID should fail
  EXPECT_NE(0, storage->append("dup_seg", 4096));

  EXPECT_EQ(0, storage->close());
}

// Test: Appending a zero-size segment returns error
TEST_F(BufferStorageWriteTest, AppendZeroSize) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  EXPECT_NE(0, storage->append("zero_seg", 0));

  EXPECT_EQ(0, storage->close());
}

// ===== Code Review Issue Tests =====
// The following tests target specific bugs/races found during code review.

// PR#414 Issue: data_size concurrent race on same segment.
// Multiple threads calling write() with different offsets should not corrupt
// the (data_size, padding_size) pair. Their sum must equal capacity when
// observed after all writers quiesce (individual unsynchronized reads during
// concurrent writes may appear torn, which is expected).
TEST_F(BufferStorageWriteTest, CR_DataSizePaddingSizeInvariant) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 8192));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);
  const size_t cap = seg->capacity();

  const int kNumThreads = 8;
  const int kIters = 200;
  std::atomic<int> write_failures{0};
  std::vector<std::thread> threads;

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      char buf[64];
      std::memset(buf, 'A' + i, sizeof(buf));
      for (int j = 0; j < kIters; ++j) {
        // Write at various offsets within capacity to exercise data_size growth
        size_t offset = ((i * 64) + j * 7) % (cap - 64);
        if (seg->write(offset, buf, sizeof(buf)) != sizeof(buf)) {
          write_failures.fetch_add(1);
        }
      }
    });
  }

  for (auto &t : threads) t.join();
  EXPECT_EQ(0, write_failures.load());
  // After all writers stop, the invariant MUST hold
  EXPECT_EQ(cap, seg->data_size() + seg->padding_size());
  EXPECT_GT(seg->data_size(), 0u);
  EXPECT_EQ(0, storage->close());
}

// PR#414 Issue: Concurrent write() + resize() on same segment.
// meta_mtx_ must serialize so that (data_size, padding_size) stays consistent.
// The invariant is verified after all threads stop (reads without meta_mtx_
// during concurrent mutation may observe a torn pair, which is expected).
TEST_F(BufferStorageWriteTest, CR_ConcurrentWriteAndResize) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 8192));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);
  const size_t cap = seg->capacity();

  std::atomic<bool> stop{false};
  std::atomic<int> write_failures{0};

  // Writer thread: grows data_size by writing at increasing offsets
  std::thread writer([&]() {
    char buf[128];
    std::memset(buf, 'W', sizeof(buf));
    for (int j = 0; j < 300 && !stop.load(std::memory_order_relaxed); ++j) {
      size_t offset = j % (cap - 128);
      if (seg->write(offset, buf, sizeof(buf)) != sizeof(buf)) {
        write_failures.fetch_add(1);
      }
    }
  });

  // Resizer thread: constantly resizes
  std::thread resizer([&]() {
    for (int j = 0; j < 300 && !stop.load(std::memory_order_relaxed); ++j) {
      size_t new_size = (j * 37) % cap;
      seg->resize(new_size);
    }
    stop.store(true);
  });

  writer.join();
  resizer.join();

  EXPECT_EQ(0, write_failures.load());
  // After quiescence, invariant must hold
  EXPECT_EQ(cap, seg->data_size() + seg->padding_size());
  EXPECT_EQ(0, storage->close());
}

// Chain-split bug: Many appends exhaust segment_meta capacity, triggering
// chain split. After reopen, ALL segments must be findable.
// (Tests fix for reserve()-induced dangling pointer in append_segment.)
TEST_F(BufferStorageWriteTest, CR_ChainSplitAllSegmentsAccessible) {
  const int kNumSegments =
      50;  // Enough to trigger chain split with default 4096 meta capacity

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);

    for (int i = 0; i < kNumSegments; ++i) {
      std::string name = "chain_seg_" + std::to_string(i);
      ASSERT_EQ(0, storage->append(name, 4096))
          << "Failed to append segment " << i;
      auto seg = storage->get(name);
      ASSERT_TRUE(seg) << "Failed to get segment " << name
                       << " right after append";
      // Write a marker so we can verify on reopen
      std::string marker = "marker_" + std::to_string(i);
      EXPECT_EQ(marker.size(), seg->write(0, marker.data(), marker.size()));
    }
    EXPECT_EQ(0, storage->flush());
    EXPECT_EQ(0, storage->close());
  }

  // Reopen and verify ALL segments are present and readable
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    for (int i = 0; i < kNumSegments; ++i) {
      std::string name = "chain_seg_" + std::to_string(i);
      auto seg = storage->get(name);
      ASSERT_TRUE(seg) << "Segment " << name
                       << " missing after reopen (chain-split bug?)";
      std::string expected = "marker_" + std::to_string(i);
      std::vector<char> buf(expected.size());
      EXPECT_EQ(expected.size(), seg->fetch(0, buf.data(), buf.size()));
      EXPECT_EQ(expected, std::string(buf.data(), buf.size()))
          << "Data mismatch for " << name;
    }
    EXPECT_EQ(0, storage->close());
  }
}

// mapping_shard_id bug: Multiple BufferStorage instances opened on the
// same thread must work correctly (the old thread_local shard_id would
// map them to the same shard, causing potential conflicts).
TEST_F(BufferStorageWriteTest, CR_MultipleInstancesSameThread) {
  std::string path2 = file_path_ + "_second";
  ailego::File::Delete(path2);

  auto storage1 = OpenWritable();
  ASSERT_TRUE(storage1);

  // Open a second independent BufferStorage instance
  auto storage2 = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_TRUE(storage2);
  ailego::Params params;
  storage2->init(params);
  ASSERT_EQ(0, storage2->open(path2, true));

  // Append and write to both concurrently from the SAME thread
  ASSERT_EQ(0, storage1->append("seg_a", 4096));
  ASSERT_EQ(0, storage2->append("seg_b", 4096));

  auto seg_a = storage1->get("seg_a");
  auto seg_b = storage2->get("seg_b");
  ASSERT_TRUE(seg_a);
  ASSERT_TRUE(seg_b);

  std::string da = "instance_one_data";
  std::string db = "instance_two_data";
  EXPECT_EQ(da.size(), seg_a->write(0, da.data(), da.size()));
  EXPECT_EQ(db.size(), seg_b->write(0, db.data(), db.size()));

  // Verify data isolation
  std::vector<char> buf1(da.size());
  EXPECT_EQ(da.size(), seg_a->fetch(0, buf1.data(), buf1.size()));
  EXPECT_EQ(da, std::string(buf1.data(), buf1.size()));

  std::vector<char> buf2(db.size());
  EXPECT_EQ(db.size(), seg_b->fetch(0, buf2.data(), buf2.size()));
  EXPECT_EQ(db, std::string(buf2.data(), buf2.size()));

  EXPECT_EQ(0, storage1->close());
  EXPECT_EQ(0, storage2->close());
  ailego::File::Delete(path2);
}

// Cross-page read/write: Write data spanning page boundaries (4KB pages),
// then read back via both fetch() and read(MemoryBlock&) to verify the
// cross-page buffer allocation path. (Tests fix for UAF in cross-page read.)
TEST_F(BufferStorageWriteTest, CR_CrossPageWriteAndRead) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  // Segment large enough to span multiple pages
  ASSERT_EQ(0, storage->append("cross_page_seg", 16384));
  auto seg = storage->get("cross_page_seg");
  ASSERT_TRUE(seg);

  // Write 5000 bytes starting at offset 2000, which crosses the first
  // page boundary at 4096 (relative to segment data start in the file).
  const size_t kWriteOffset = 2000;
  const size_t kWriteLen = 5000;
  std::vector<char> write_data(kWriteLen);
  for (size_t i = 0; i < kWriteLen; ++i) {
    write_data[i] = static_cast<char>((i * 7 + 13) % 256);
  }
  EXPECT_EQ(kWriteLen, seg->write(kWriteOffset, write_data.data(), kWriteLen));

  // Read back via fetch (uses read_range internally for cross-page)
  std::vector<char> fetch_buf(kWriteLen);
  EXPECT_EQ(kWriteLen, seg->fetch(kWriteOffset, fetch_buf.data(), kWriteLen));
  EXPECT_EQ(write_data, fetch_buf);

  // Read back via read(MemoryBlock&) - exercises the cross-page alloc path on
  // 4K hosts and the writable-pool snapshot path on large-page hosts.
  {
    IndexStorage::MemoryBlock mb;
    EXPECT_EQ(kWriteLen, seg->read(kWriteOffset, mb, kWriteLen));
    EXPECT_EQ(0, std::memcmp(write_data.data(), mb.data(), kWriteLen));
  }

  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ImmutableReadPinsWritablePageWithoutCopy) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("immutable_read_seg", 8192));
  auto segment = storage->get("immutable_read_seg");
  ASSERT_TRUE(segment);
  const size_t page_offset = segment->data_offset() % ailego::kVectorPageSize;
  const size_t offset = page_offset + 256 <= ailego::kVectorPageSize
                            ? 0
                            : ailego::kVectorPageSize - page_offset;
  const std::string expected = "immutable-vector-bytes";
  ASSERT_EQ(expected.size(),
            segment->write(offset, expected.data(), expected.size()));

  IndexStorage::MemoryBlock snapshot;
  ASSERT_EQ(expected.size(), segment->read(offset, snapshot, expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, snapshot.type_);

  IndexStorage::MemoryBlock pinned;
  ASSERT_EQ(expected.size(),
            segment->read_immutable(offset, pinned, expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, pinned.type_);
  EXPECT_TRUE(pinned.buffer_pool_handle_owner_);
  EXPECT_EQ(expected, std::string(static_cast<const char *>(pinned.data()),
                                  expected.size()));

  IndexStorage::MemoryBlock borrowed;
  ASSERT_EQ(expected.size(), segment->read_borrowed_immutable(offset, borrowed,
                                                              expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, borrowed.type_);
  EXPECT_FALSE(borrowed.buffer_pool_handle_owner_);

  borrowed.reset();
  pinned.reset();
  snapshot.reset();
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ImmutableReadCopiesCrossPageRange) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  const size_t page_size = ailego::kVectorPageSize;
  ASSERT_EQ(0, storage->append("immutable_cross_page", 3 * page_size));
  auto segment = storage->get("immutable_cross_page");
  ASSERT_TRUE(segment);

  const size_t aligned_offset =
      (page_size - segment->data_offset() % page_size) % page_size;
  const size_t offset = aligned_offset + page_size - 128;
  std::vector<char> expected(3072);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<char>((i * 17 + 11) % 251);
  }
  ASSERT_EQ(expected.size(),
            segment->write(offset, expected.data(), expected.size()));

  IndexStorage::MemoryBlock snapshot;
  ASSERT_EQ(expected.size(),
            segment->read_immutable(offset, snapshot, expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, snapshot.type_);
  EXPECT_EQ(0, std::memcmp(expected.data(), snapshot.data(), expected.size()));
  snapshot.reset();
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ImmutableBatchReadUsesWritableCache) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);
  const size_t page_size = ailego::kVectorPageSize;
  ASSERT_EQ(0, storage->append("immutable_batch", 3 * page_size));
  auto segment = storage->get("immutable_batch");
  ASSERT_TRUE(segment);

  EXPECT_FALSE(segment->prefer_borrowed_batch_for(
      std::max<size_t>(1, page_size / 8) - 1));
  EXPECT_TRUE(
      segment->prefer_borrowed_batch_for(std::max<size_t>(1, page_size / 8)));

  std::vector<char> payload(3 * page_size);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>((i * 31 + 7) % 251);
  }
  ASSERT_EQ(payload.size(), segment->write(0, payload.data(), payload.size()));

  const size_t aligned_offset =
      (page_size - segment->data_offset() % page_size) % page_size;
  const size_t cross_offset = aligned_offset + page_size - 64;
  IndexStorage::MemoryBlock blocks[2];
  IndexStorage::Segment::BorrowedRead reads[] = {
      {segment.get(), aligned_offset, 128, &blocks[0]},
      {segment.get(), cross_offset, 128, &blocks[1]},
  };
  ASSERT_TRUE(segment->read_borrowed_batch_immutable(reads, 2));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks[0].type_);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_MMAP, blocks[1].type_);
  EXPECT_EQ(0, std::memcmp(payload.data() + aligned_offset, blocks[0].data(),
                           reads[0].length));
  EXPECT_EQ(0, std::memcmp(payload.data() + cross_offset, blocks[1].data(),
                           reads[1].length));
  blocks[0].reset();
  blocks[1].reset();
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ReadOnlyBatchFallsBackWhenPinsExceedBudget) {
  constexpr size_t kSegmentBytes = 68UL * 1024UL * 1024UL;
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("batch_pressure", kSegmentBytes));
    auto segment = storage->get("batch_pressure");
    ASSERT_TRUE(segment);
    const char marker = 'Z';
    ASSERT_EQ(1U, segment->write(kSegmentBytes - 1, &marker, 1));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto segment = storage->get("batch_pressure");
  ASSERT_TRUE(segment);

  const size_t page_size = ailego::kVectorPageSize;
  const size_t first =
      (page_size - segment->data_offset() % page_size) % page_size;
  const size_t count = (segment->data_size() - first) / page_size;
  ASSERT_GT(count * page_size, 64UL * 1024UL * 1024UL);

  // Repeat the final (necessarily cold) page to verify fallback is deduplicated
  // by page rather than issued once per vector occurrence.
  constexpr size_t kDuplicateTailReads = 2;
  std::vector<IndexStorage::MemoryBlock> blocks(count + kDuplicateTailReads);
  std::vector<IndexStorage::Segment::BorrowedRead> reads;
  reads.reserve(blocks.size());
  for (size_t i = 0; i < count; ++i) {
    reads.emplace_back(segment.get(), first + i * page_size, 1, &blocks[i]);
  }
  for (size_t i = 0; i < kDuplicateTailReads; ++i) {
    reads.emplace_back(segment.get(), first + (count - 1) * page_size, 1,
                       &blocks[count + i]);
  }

  auto pool = storage->vec_buffer_pool();
  ASSERT_TRUE(pool);
  const auto before = pool->stats();
  ASSERT_TRUE(segment->read_borrowed_batch_immutable(reads.data(),
                                                      reads.size()));
  size_t cached_unique = 0;
  size_t bypassed_unique = 0;
  for (size_t i = 0; i < count; ++i) {
    const auto &block = blocks[i];
    if (block.type_ == IndexStorage::MemoryBlock::MBT_BUFFERPOOL) {
      ++cached_unique;
    } else {
      EXPECT_EQ(IndexStorage::MemoryBlock::MBT_MMAP, block.type_);
      ++bypassed_unique;
    }
    ASSERT_NE(nullptr, block.data());
    EXPECT_EQ(0, *static_cast<const unsigned char *>(block.data()));
  }
  EXPECT_GT(cached_unique, 0U);
  EXPECT_GT(bypassed_unique, 0U);
  for (size_t i = 0; i < kDuplicateTailReads; ++i) {
    EXPECT_EQ(IndexStorage::MemoryBlock::MBT_MMAP, blocks[count + i].type_);
    ASSERT_NE(nullptr, blocks[count + i].data());
    EXPECT_EQ(0,
              *static_cast<const unsigned char *>(blocks[count + i].data()));
  }
  const auto after = pool->stats();
  EXPECT_EQ(bypassed_unique, after.bypass_reads - before.bypass_reads);
  blocks.clear();
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ReadOnlyBatchPreservesSharedCacheReserve) {
  constexpr size_t kSegmentBytes = 64UL * 1024UL * 1024UL;
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("batch_reserve", kSegmentBytes));
    auto segment = storage->get("batch_reserve");
    ASSERT_TRUE(segment);
    const char marker = 'R';
    ASSERT_EQ(1U, segment->write(kSegmentBytes - 1, &marker, 1));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto segment = storage->get("batch_reserve");
  ASSERT_TRUE(segment);

  const size_t page_size = ailego::kVectorPageSize;
  const size_t first =
      (page_size - segment->data_offset() % page_size) % page_size;
  constexpr size_t kSharedReserveBytes = 4UL * 1024UL * 1024UL;
  const size_t available_before =
      ailego::MemoryLimitPool::get_instance().available();
  ASSERT_GT(available_before, kSharedReserveBytes);
  const size_t count =
      (available_before - kSharedReserveBytes) / page_size + 1;
  ASSERT_LT(first + count * page_size, segment->data_size());

  std::vector<IndexStorage::MemoryBlock> blocks(count);
  std::vector<IndexStorage::Segment::BorrowedRead> reads;
  reads.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    reads.emplace_back(segment.get(), first + i * page_size, 1, &blocks[i]);
  }

  ASSERT_TRUE(segment->read_borrowed_batch_immutable(reads.data(),
                                                      reads.size()));
  EXPECT_GE(ailego::MemoryLimitPool::get_instance().available() + page_size,
            kSharedReserveBytes);
  blocks.clear();
  EXPECT_EQ(0, storage->close());
}

// Repeated legacy pointer reads from a writable cached page must reuse the
// pinned page. Retaining a separate 4K-aligned snapshot for every read grows
// memory until close() and makes long Optimize workloads consume gigabytes.
TEST_F(BufferStorageWriteTest, CR_WritableLegacyPointerReadReusesCachedPage) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("legacy_pointer_seg", 8192));
  auto seg = storage->get("legacy_pointer_seg");
  ASSERT_TRUE(seg);

  constexpr size_t kReadLen = 64;
  const size_t data_offset_in_page =
      seg->data_offset() % ailego::kVectorPageSize;
  const size_t read_offset =
      (ailego::kVectorPageSize - data_offset_in_page) % ailego::kVectorPageSize;
  ASSERT_LE(read_offset + kReadLen, seg->capacity());

  std::vector<char> expected(kReadLen, 'P');
  ASSERT_EQ(kReadLen,
            seg->write(read_offset, expected.data(), expected.size()));

  const void *first = nullptr;
  ASSERT_EQ(kReadLen, seg->read(read_offset, &first, kReadLen));
  ASSERT_NE(nullptr, first);
  EXPECT_EQ(0, std::memcmp(expected.data(), first, expected.size()));

  for (size_t i = 0; i < 128; ++i) {
    const void *again = nullptr;
    ASSERT_EQ(kReadLen, seg->read(read_offset, &again, kReadLen));
    EXPECT_EQ(first, again);
  }

  EXPECT_EQ(0, storage->close());
}

// Dirty flag race: write() after flush_index() must re-set the dirty flag.
// If the write lands between CAS(dirty, false) and the end of flush,
// the next flush must still persist it. Verified by close→reopen→read.
TEST_F(BufferStorageWriteTest, CR_DirtyFlagNotLostAfterFlush) {
  std::string early_data = "early";
  std::string late_data = "late_write_after_flush";

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);

    // Write and flush
    EXPECT_EQ(early_data.size(),
              seg->write(0, early_data.data(), early_data.size()));
    EXPECT_EQ(0, storage->flush());

    // Write again AFTER flush - dirty flag must be re-set
    EXPECT_EQ(late_data.size(),
              seg->write(100, late_data.data(), late_data.size()));
    // Close without explicit flush (close_index will flush)
    EXPECT_EQ(0, storage->close());
  }

  // Reopen and verify the late write persisted
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);

    std::vector<char> buf(late_data.size());
    EXPECT_EQ(late_data.size(), seg->fetch(100, buf.data(), buf.size()));
    EXPECT_EQ(late_data, std::string(buf.data(), buf.size()));
    EXPECT_EQ(0, storage->close());
  }
}

// Stress test: Concurrent flush + write interleaving to expose dirty flag
// races. All writes that return successfully MUST be visible after final
// close+reopen.
TEST_F(BufferStorageWriteTest, CR_ConcurrentFlushWriteDirtyFlagStress) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 65536));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  // Track the highest offset+len successfully written
  std::atomic<size_t> max_committed_end{0};
  std::atomic<bool> stop{false};

  // Writer: writes sequentially increasing offsets
  std::thread writer([&]() {
    char pattern[64];
    std::memset(pattern, 'P', sizeof(pattern));
    for (int i = 0; i < 500 && !stop.load(std::memory_order_relaxed); ++i) {
      size_t offset = i * 64;
      if (offset + 64 > 65536) break;
      if (seg->write(offset, pattern, 64) == 64) {
        // Update max committed end
        size_t end = offset + 64;
        size_t cur = max_committed_end.load(std::memory_order_relaxed);
        while (end > cur) {
          if (max_committed_end.compare_exchange_weak(
                  cur, end, std::memory_order_relaxed)) {
            break;
          }
        }
      }
    }
  });

  // Flusher: repeatedly flushes to trigger the CAS(dirty, false) path
  std::thread flusher([&]() {
    for (int i = 0; i < 50; ++i) {
      storage->flush();
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    stop.store(true);
  });

  writer.join();
  flusher.join();

  size_t final_data_size = seg->data_size();
  EXPECT_GE(final_data_size, max_committed_end.load());
  EXPECT_EQ(0, storage->close());
}

// Pointer stability after append: WrappedSegment obtained BEFORE a new
// append must still work correctly AFTER the append (unordered_map address
// stability guarantee). This tests the fix for reserve()-based invalidation.
TEST_F(BufferStorageWriteTest, CR_PointerStabilityAcrossAppend) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg_first", 4096));
  auto seg_first = storage->get("seg_first");
  ASSERT_TRUE(seg_first);

  // Write initial data
  std::string initial = "before_append";
  EXPECT_EQ(initial.size(),
            seg_first->write(0, initial.data(), initial.size()));

  // Append many more segments (may trigger internal rehash/resize)
  for (int i = 0; i < 20; ++i) {
    ASSERT_EQ(0, storage->append("new_seg_" + std::to_string(i), 4096));
  }

  // The original segment handle must still be valid and writable
  std::string after = "_after_appends";
  EXPECT_EQ(after.size(),
            seg_first->write(initial.size(), after.data(), after.size()));

  // Verify full data
  std::string expected = initial + after;
  std::vector<char> buf(expected.size());
  EXPECT_EQ(expected.size(), seg_first->fetch(0, buf.data(), buf.size()));
  EXPECT_EQ(expected, std::string(buf.data(), buf.size()));

  EXPECT_EQ(0, storage->close());
}

// update_data_crc concurrent with write: CRC update must be serialized
// with data_size changes via meta_mtx_. Invariant verified post-quiescence.
TEST_F(BufferStorageWriteTest, CR_ConcurrentWriteAndCrcUpdate) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);

  ASSERT_EQ(0, storage->append("seg1", 8192));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);
  const size_t cap = seg->capacity();

  std::atomic<bool> stop{false};
  std::atomic<int> write_failures{0};

  // Writer thread
  std::thread writer([&]() {
    char buf[128];
    std::memset(buf, 'X', sizeof(buf));
    for (int i = 0; i < 500 && !stop.load(std::memory_order_relaxed); ++i) {
      size_t offset = (i * 128) % (cap - 128);
      if (seg->write(offset, buf, sizeof(buf)) != sizeof(buf)) {
        write_failures.fetch_add(1);
      }
    }
  });

  // CRC updater thread: concurrently updates CRC
  std::thread crc_updater([&]() {
    for (int i = 0; i < 500 && !stop.load(std::memory_order_relaxed); ++i) {
      seg->update_data_crc(static_cast<uint32_t>(i));
    }
    stop.store(true);
  });

  writer.join();
  crc_updater.join();

  EXPECT_EQ(0, write_failures.load());
  // After all threads stop, invariant must hold
  EXPECT_EQ(cap, seg->data_size() + seg->padding_size());
  // CRC should have been updated (last writer wins)
  // Just verify it doesn't crash and the value is readable
  (void)seg->data_crc();
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, CR_ConcurrentSamePageReadSeesWholeWrite) {
  auto storage = OpenWritable();
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->append("seg1", 2 * ailego::kVectorPageSize));
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);

  constexpr size_t kPayloadSize = 256;
  const size_t page_offset = seg->data_offset() % ailego::kVectorPageSize;
  const size_t offset = page_offset + kPayloadSize <= ailego::kVectorPageSize
                            ? 0
                            : ailego::kVectorPageSize - page_offset;
  std::vector<char> pattern_a(kPayloadSize, static_cast<char>(0x55));
  std::vector<char> pattern_b(kPayloadSize, static_cast<char>(0xAA));
  ASSERT_EQ(kPayloadSize,
            seg->write(offset, pattern_a.data(), pattern_a.size()));

  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  std::thread writer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (size_t i = 0; i < 10000; ++i) {
      const auto &pattern = (i & 1U) == 0 ? pattern_b : pattern_a;
      if (seg->write(offset, pattern.data(), pattern.size()) != kPayloadSize) {
        failed.store(true, std::memory_order_release);
        return;
      }
    }
  });
  std::thread reader([&] {
    std::vector<char> observed(kPayloadSize);
    start.store(true, std::memory_order_release);
    for (size_t i = 0; i < 10000; ++i) {
      if (seg->fetch(offset, observed.data(), observed.size()) !=
              kPayloadSize ||
          (observed != pattern_a && observed != pattern_b)) {
        failed.store(true, std::memory_order_release);
        return;
      }
    }
  });
  writer.join();
  reader.join();
  EXPECT_FALSE(failed.load(std::memory_order_acquire));
  EXPECT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, CR_MemoryBlockOutlivesReadOnlyStorage) {
  const std::string expected = "lifetime";
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    ASSERT_EQ(expected.size(), seg->write(0, expected.data(), expected.size()));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  IndexStorage::MemoryBlock block;
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    ASSERT_EQ(expected.size(), seg->read(0, block, expected.size()));
    ASSERT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
    ASSERT_EQ(0, storage->close());
    seg.reset();
    storage.reset();
    EXPECT_EQ(0, std::memcmp(expected.data(), block.data(), expected.size()));
  }
  block.reset();
}

TEST_F(BufferStorageWriteTest, CR_BorrowedReadAvoidsOwningHandle) {
  const std::string expected = "borrowed";
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    ASSERT_EQ(expected.size(), seg->write(0, expected.data(), expected.size()));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto seg = storage->get("seg1");
  ASSERT_TRUE(seg);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(expected.size(), seg->read_borrowed(0, block, expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
  EXPECT_EQ(nullptr, block.buffer_pool_handle_owner_);
  EXPECT_NE(nullptr, block.buffer_pool_handle_);
  EXPECT_EQ(0, std::memcmp(expected.data(), block.data(), expected.size()));

  block.reset();
  ASSERT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, BorrowedReadUsesOneColdPageLoadSequence) {
  const std::string expected = "borrowed pressure fallback";
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto segment = storage->get("seg1");
    ASSERT_TRUE(segment);
    ASSERT_EQ(expected.size(),
              segment->write(0, expected.data(), expected.size()));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto segment = storage->get("seg1");
  ASSERT_TRUE(segment);

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t external_charge = pool.available();
  ASSERT_GT(external_charge, 0u);
  ASSERT_TRUE(pool.try_charge_external(external_charge));
  const uint64_t high_watermark_hits_before = pool.stats().high_watermark_hits;

  IndexStorage::MemoryBlock block;
  ASSERT_EQ(expected.size(), segment->read_borrowed(0, block, expected.size()));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
  EXPECT_EQ(0, std::memcmp(expected.data(), block.data(), expected.size()));
  // get_single_page() performs one initial attempt plus at most 50 retries.
  EXPECT_LE(pool.stats().high_watermark_hits - high_watermark_hits_before, 51u);

  pool.release_external(external_charge);
  ASSERT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, ReadOnlyPrefetchPreservesCachePriority) {
  const size_t page_size = ailego::kVectorPageSize;
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("hot", 3 * page_size));
    auto segment = storage->get("hot");
    ASSERT_TRUE(segment);
    ASSERT_EQ(3 * page_size, segment->resize(3 * page_size));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto segment = storage->get("hot");
  ASSERT_TRUE(segment);
  auto pool = storage->vec_buffer_pool();
  ASSERT_TRUE(pool);

  const size_t first_page = segment->data_offset() / page_size;
  segment->prefetch(0, 1, IndexStorage::Segment::CachePriority::kHigh);
  EXPECT_TRUE(pool->is_page_resident(first_page));
  EXPECT_EQ(ailego::VecBufferPool::kHighPriority,
            pool->page_table_.eviction_priority(first_page));

  // A later, colder hint must not demote an already protected page.
  segment->prefetch(0, 1, IndexStorage::Segment::CachePriority::kNormal);
  EXPECT_EQ(ailego::VecBufferPool::kHighPriority,
            pool->page_table_.eviction_priority(first_page));

  const size_t second_page_offset =
      page_size - (segment->data_offset() % page_size);
  const size_t second_page =
      (segment->data_offset() + second_page_offset) / page_size;
  segment->prefetch(second_page_offset, 1,
                    IndexStorage::Segment::CachePriority::kNormal);
  EXPECT_TRUE(pool->is_page_resident(second_page));
  EXPECT_EQ(ailego::VecBufferPool::kNormalPriority,
            pool->page_table_.eviction_priority(second_page));

  ASSERT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, BatchBorrowedReadAcrossSegments) {
  const size_t page_size = ailego::kVectorPageSize;
  std::vector<char> payload_a(3 * page_size);
  std::vector<char> payload_b(2 * page_size);
  for (size_t i = 0; i < payload_a.size(); ++i) {
    payload_a[i] = static_cast<char>((i * 17 + 3) % 251);
  }
  for (size_t i = 0; i < payload_b.size(); ++i) {
    payload_b[i] = static_cast<char>((i * 29 + 11) % 251);
  }

  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg_a", payload_a.size()));
    ASSERT_EQ(0, storage->append("seg_b", payload_b.size()));
    auto seg_a = storage->get("seg_a");
    auto seg_b = storage->get("seg_b");
    ASSERT_TRUE(seg_a);
    ASSERT_TRUE(seg_b);
    ASSERT_EQ(payload_a.size(),
              seg_a->write(0, payload_a.data(), payload_a.size()));
    ASSERT_EQ(payload_b.size(),
              seg_b->write(0, payload_b.data(), payload_b.size()));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto storage = OpenReadOnly();
  ASSERT_TRUE(storage);
  auto seg_a = storage->get("seg_a");
  auto seg_b = storage->get("seg_b");
  ASSERT_TRUE(seg_a);
  ASSERT_TRUE(seg_b);

  const size_t a_page_aligned_offset =
      (page_size - seg_a->data_offset() % page_size) % page_size;
  const size_t a_cross_page_offset =
      (2 * page_size - 32 - seg_a->data_offset() % page_size) % page_size;
  const size_t b_page_aligned_offset =
      (page_size - seg_b->data_offset() % page_size) % page_size;
  ASSERT_LE(a_page_aligned_offset + 64, payload_a.size());
  ASSERT_LE(a_cross_page_offset + 128, payload_a.size());
  ASSERT_LE(b_page_aligned_offset + 96, payload_b.size());

  IndexStorage::MemoryBlock blocks[4];
  IndexStorage::Segment::BorrowedRead reads[] = {
      {seg_a.get(), a_page_aligned_offset, 64, &blocks[0]},
      {seg_a.get(), a_page_aligned_offset + 8, 32, &blocks[1]},
      {seg_a.get(), a_cross_page_offset, 128, &blocks[2]},
      {seg_b.get(), b_page_aligned_offset, 96, &blocks[3]},
  };
  ASSERT_TRUE(seg_a->read_borrowed_batch(reads, 4));

  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks[0].type_);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks[1].type_);
  // Cross-page reads are served from the reused thread-local scratch arena as
  // non-owning views (ZVEC_CROSS_ARENA default on); the reassembled bytes are
  // still validated below.
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_MMAP, blocks[2].type_);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks[3].type_);
  EXPECT_EQ(0, std::memcmp(payload_a.data() + a_page_aligned_offset,
                           blocks[0].data(), 64));
  EXPECT_EQ(0, std::memcmp(payload_a.data() + a_page_aligned_offset + 8,
                           blocks[1].data(), 32));
  EXPECT_EQ(0, std::memcmp(payload_a.data() + a_cross_page_offset,
                           blocks[2].data(), 128));
  EXPECT_EQ(0, std::memcmp(payload_b.data() + b_page_aligned_offset,
                           blocks[3].data(), 96));

  for (auto &block : blocks) {
    block.reset();
  }
  ASSERT_EQ(0, storage->close());
}

TEST_F(BufferStorageWriteTest, CR_ReadOnlyMetadataPressureFallsBackToBypass) {
  const std::string expected = "bypass";
  {
    auto storage = OpenWritable();
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->append("seg1", 4096));
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    ASSERT_EQ(expected.size(), seg->write(0, expected.data(), expected.size()));
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
  }

  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(ailego::kVectorPageSize));
  {
    auto storage = OpenReadOnly();
    ASSERT_TRUE(storage);
    auto seg = storage->get("seg1");
    ASSERT_TRUE(seg);
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(expected.size(), seg->read(0, block, expected.size()));
    EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
    EXPECT_EQ(0, std::memcmp(expected.data(), block.data(), expected.size()));
    ASSERT_EQ(0, storage->close());
  }
  ASSERT_EQ(0, pool.init(64UL * 1024UL * 1024UL));
}
