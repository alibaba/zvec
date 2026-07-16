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

#include <atomic>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

static std::string iter_test_path = "test_iterator_collection";

class IteratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll *
                                                       1024ll);
    FileHelper::RemoveDirectory(iter_test_path);
  }

  void TearDown() override {
    FileHelper::RemoveDirectory(iter_test_path);
  }
};

// Test 1: Basic iteration — insert 100 docs, iterate, verify count + PK
TEST_F(IteratorTest, BasicIteration) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  auto collection = std::move(result.value());

  // Insert 100 docs
  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  auto insert_result = collection->Insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  // Flush to ensure all docs are in persist segments
  collection->Flush();

  // Create iterator
  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value()) << iter_result.error().message();
  auto iter = iter_result.value();

  // Iterate and collect PKs + verify scalar fields
  std::set<std::string> pks;
  int count = 0;
  while (true) {
    auto r = iter->Next();
    if (!r.has_value()) {
      FAIL() << "Iterator error: " << r.error().message();
    }
    if (r.value() == nullptr) {
      break;  // EOF
    }
    auto doc = r.value();
    pks.insert(doc->pk());
    // Verify scalar field extraction (int32 field exists in
    // TestHelper::CreateNormalSchema)
    auto int32_val = doc->get<int32_t>("int32");
    EXPECT_TRUE(int32_val.has_value())
        << "int32 field missing for doc " << doc->pk();
    count++;
  }

  EXPECT_EQ(count, N) << "Expected " << N << " docs, got " << count;
  EXPECT_EQ(pks.size(), N) << "Expected " << N << " unique PKs";

  iter->Close();
  collection->Destroy();
}

// Test 2: Empty collection — iterator should immediately return EOF
TEST_F(IteratorTest, EmptyCollection) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // No docs inserted, just create iterator
  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Next should return EOF immediately
  auto r = iter->Next();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r.value(), nullptr) << "Expected EOF on empty collection";

  collection->Destroy();
}

// Test 3: Deleted docs are filtered out
TEST_F(IteratorTest, DeletedDocsFiltered) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // Insert 50 docs
  const int N = 50;
  std::vector<Doc> docs;
  std::vector<std::string> pks_to_delete;
  for (int i = 0; i < N; i++) {
    auto doc = TestHelper::CreateDoc(i, *schema);
    docs.push_back(doc);
    if (i % 2 == 0) {
      pks_to_delete.push_back(doc.pk());
    }
  }
  auto insert_result = collection->Insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  // Delete every other doc
  auto delete_result = collection->Delete(pks_to_delete);
  ASSERT_TRUE(delete_result.has_value());

  collection->Flush();

  // Create iterator
  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Iterate
  std::set<std::string> deleted_set(pks_to_delete.begin(), pks_to_delete.end());
  int count = 0;
  while (true) {
    auto r = iter->Next();
    if (!r.has_value()) {
      FAIL() << "Iterator error: " << r.error().message();
    }
    if (r.value() == nullptr) break;

    auto pk = r.value()->pk();
    EXPECT_EQ(deleted_set.count(pk), 0)
        << "Deleted doc " << pk << " should not appear in iteration";
    count++;
  }

  // Should have N - deleted_count docs
  EXPECT_EQ(count, N - static_cast<int>(pks_to_delete.size()));

  collection->Destroy();
}

// Test 4: Iterator after Close() returns error
TEST_F(IteratorTest, CloseThenNext) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // Insert a few docs
  std::vector<Doc> docs;
  for (int i = 0; i < 5; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  auto insert_result = collection->Insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  collection->Flush();

  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Close iterator
  iter->Close();

  // Next() after Close should return error
  auto r = iter->Next();
  EXPECT_FALSE(r.has_value()) << "Expected error after Close()";

  collection->Destroy();
}

// Test 5: Iterator with include_vector=true — verify vector fields are present
TEST_F(IteratorTest, IncludeVector) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // Insert 10 docs
  const int N = 10;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  auto insert_result = collection->Insert(docs);
  ASSERT_TRUE(insert_result.has_value());
  collection->Flush();

  // Create iterator with include_vector=true (default)
  IteratorOptions iter_opts;
  iter_opts.include_vector_ = true;
  auto iter_result = collection->CreateIterator(iter_opts);
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->Next();
    if (!r.has_value()) FAIL() << r.error().message();
    if (r.value() == nullptr) break;

    auto doc = r.value();
    // Verify PK
    EXPECT_FALSE(doc->pk().empty());

    // Verify vector field exists (dense_fp32 is in
    // TestHelper::CreateNormalSchema)
    auto vec = doc->get<std::vector<float>>("dense_fp32");
    EXPECT_TRUE(vec.has_value())
        << "dense_fp32 vector missing for doc " << doc->pk();
    if (vec.has_value()) {
      EXPECT_EQ(vec->size(), 128) << "dense_fp32 dimension should be 128";
    }

    count++;
  }

  EXPECT_EQ(count, N);
  collection->Destroy();
}

// Test 6: Iterator with include_vector=false — verify no vector fields
TEST_F(IteratorTest, ExcludeVector) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  std::vector<Doc> docs;
  for (int i = 0; i < 5; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  collection->Insert(docs);
  collection->Flush();

  IteratorOptions iter_opts;
  iter_opts.include_vector_ = false;
  auto iter_result = collection->CreateIterator(iter_opts);
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->Next();
    if (!r.has_value()) FAIL() << r.error().message();
    if (r.value() == nullptr) break;

    auto doc = r.value();
    // Scalar field should be present
    auto int32_val = doc->get<int32_t>("int32");
    EXPECT_TRUE(int32_val.has_value());

    // Vector field should NOT be present (include_vector=false)
    auto vec = doc->get<std::vector<float>>("dense_fp32");
    EXPECT_FALSE(vec.has_value())
        << "Vector should not be present with include_vector=false";

    count++;
  }

  EXPECT_EQ(count, 5);
  collection->Destroy();
}

// Test 7: Scalar type mapping — every scalar/array Arrow type extracted
// correctly. CreateNormalSchema covers 8 base types + 8 array types.
TEST_F(IteratorTest, ScalarTypeMapping) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // doc_id = 7 → deterministic values (see TestHelper::CreateDoc).
  const uint64_t kId = 7;
  std::vector<Doc> docs{TestHelper::CreateDoc(kId, *schema)};
  ASSERT_TRUE(collection->Insert(docs).has_value());
  collection->Flush();

  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  auto r = iter->Next();
  ASSERT_TRUE(r.has_value());
  ASSERT_NE(r.value(), nullptr);
  auto doc = r.value();

  // ── base scalar types ──
  EXPECT_EQ(doc->get<int32_t>("int32").value_or(-1), (int32_t)kId);
  EXPECT_EQ(doc->get<int64_t>("int64").value_or(-1), (int64_t)kId);
  EXPECT_EQ(doc->get<uint32_t>("uint32").value_or(0), (uint32_t)kId);
  EXPECT_EQ(doc->get<uint64_t>("uint64").value_or(0), (uint64_t)kId);
  EXPECT_FLOAT_EQ(doc->get<float>("float").value_or(-1), (float)kId);
  EXPECT_DOUBLE_EQ(doc->get<double>("double").value_or(-1), (double)kId);
  EXPECT_EQ(doc->get<std::string>("string").value_or(""),
            "value_" + std::to_string(kId));
  EXPECT_EQ(doc->get<bool>("bool").value_or(true), kId % 10 == 0);

  // ── array types (each element == kId, length 10) ──
  auto a_i32 = doc->get<std::vector<int32_t>>("array_int32");
  ASSERT_TRUE(a_i32.has_value());
  EXPECT_EQ(a_i32->size(), 10u);
  EXPECT_EQ((*a_i32)[0], (int32_t)kId);

  auto a_i64 = doc->get<std::vector<int64_t>>("array_int64");
  ASSERT_TRUE(a_i64.has_value());
  EXPECT_EQ((*a_i64)[0], (int64_t)kId);

  auto a_u32 = doc->get<std::vector<uint32_t>>("array_uint32");
  ASSERT_TRUE(a_u32.has_value());
  EXPECT_EQ((*a_u32)[0], (uint32_t)kId);

  auto a_u64 = doc->get<std::vector<uint64_t>>("array_uint64");
  ASSERT_TRUE(a_u64.has_value());
  EXPECT_EQ((*a_u64)[0], (uint64_t)kId);

  auto a_f = doc->get<std::vector<float>>("array_float");
  ASSERT_TRUE(a_f.has_value());
  EXPECT_FLOAT_EQ((*a_f)[0], (float)kId);

  auto a_d = doc->get<std::vector<double>>("array_double");
  ASSERT_TRUE(a_d.has_value());
  EXPECT_DOUBLE_EQ((*a_d)[0], (double)kId);

  auto a_b = doc->get<std::vector<bool>>("array_bool");
  ASSERT_TRUE(a_b.has_value());
  EXPECT_EQ(a_b->size(), 10u);

  auto a_s = doc->get<std::vector<std::string>>("array_string");
  ASSERT_TRUE(a_s.has_value());
  EXPECT_EQ((*a_s)[0], "value_" + std::to_string(kId));

  collection->Destroy();
}

// Test 8: Integration — 1000 docs, verify count + PK + scalar + vector values.
TEST_F(IteratorTest, Integration1000Docs) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 1000;
  std::vector<Doc> docs;
  docs.reserve(N);
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  ASSERT_TRUE(collection->Insert(docs).has_value());
  collection->Flush();

  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  std::set<std::string> seen_pks;
  while (true) {
    auto r = iter->Next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    auto doc = r.value();

    // PK format is "pk_<id>" (TestHelper::MakePK); derive id back.
    std::string pk = doc->pk();
    seen_pks.insert(pk);

    // int32 field == the doc's id; verify vector value matches id + 0.1.
    auto id32 = doc->get<int32_t>("int32");
    ASSERT_TRUE(id32.has_value());
    uint64_t id = static_cast<uint64_t>(*id32);

    auto vec = doc->get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vec.has_value()) << "vector missing for " << pk;
    EXPECT_EQ(vec->size(), 128u);
    EXPECT_FLOAT_EQ((*vec)[0], float(id + 0.1));

    // scalar string value matches id.
    EXPECT_EQ(doc->get<std::string>("string").value_or(""),
              "value_" + std::to_string(id));
    count++;
  }

  EXPECT_EQ(count, N);
  EXPECT_EQ(seen_pks.size(), (size_t)N);
  collection->Destroy();
}

// Test 9: Concurrency — docs inserted after iterator creation are not visible.
TEST_F(IteratorTest, ConcurrentInsertNotVisible) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 500;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(docs).has_value());
  collection->Flush();

  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Consume one doc to establish the snapshot, then insert concurrently.
  auto first = iter->Next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);

  std::atomic<bool> writer_failed{false};
  std::thread writer([&]() {
    std::vector<Doc> more;
    for (int i = N; i < N + 200; i++) {
      more.push_back(TestHelper::CreateDoc(i, *schema));
    }
    if (!collection->Insert(more).has_value()) writer_failed = true;
    collection->Flush();
  });

  int count = 1;  // already consumed one
  while (true) {
    auto r = iter->Next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count++;
  }
  writer.join();

  EXPECT_FALSE(writer_failed);
  // Snapshot was taken at creation → only the original N are visible.
  EXPECT_EQ(count, N);

  // A fresh iterator sees all N + 200.
  auto iter2 = collection->CreateIterator().value();
  int count2 = 0;
  while (true) {
    auto r = iter2->Next();
    ASSERT_TRUE(r.has_value());
    if (r.value() == nullptr) break;
    count2++;
  }
  EXPECT_EQ(count2, N + 200);

  collection->Destroy();
}

// Test 10: Concurrency — Optimize during iteration must not crash.
TEST_F(IteratorTest, ConcurrentOptimizeNoCrash) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 500;
  // Insert in two batches with a flush between, so Optimize has >1 segment.
  std::vector<Doc> b1, b2;
  for (int i = 0; i < N / 2; i++)
    b1.push_back(TestHelper::CreateDoc(i, *schema));
  for (int i = N / 2; i < N; i++)
    b2.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(b1).has_value());
  collection->Flush();
  ASSERT_TRUE(collection->Insert(b2).has_value());
  collection->Flush();

  auto iter_result = collection->CreateIterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Kick off Optimize concurrently (destroys old segments after compaction).
  std::thread optimizer([&]() { collection->Optimize(); });

  int count = 0;
  while (true) {
    auto r = iter->Next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    EXPECT_FALSE(r.value()->pk().empty());
    count++;
    // Slow the consumer slightly so Optimize overlaps with iteration.
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  optimizer.join();

  // Snapshot isolation: all N docs remain visible despite concurrent Optimize.
  EXPECT_EQ(count, N);
  collection->Destroy();
}

// Test 11: Performance — iterate 100k docs; memory should stay bounded (one
// batch at a time). Reports elapsed time; asserts correctness of the count.
TEST_F(IteratorTest, Performance100k) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 100000;
  const int kBatch = 1000;  // max write batch size is 1024 (constants.h)
  for (int start = 0; start < N; start += kBatch) {
    std::vector<Doc> docs;
    docs.reserve(kBatch);
    for (int i = start; i < start + kBatch; i++) {
      docs.push_back(TestHelper::CreateDoc(i, *schema));
    }
    auto ins = collection->Insert(docs);
    ASSERT_TRUE(ins.has_value())
        << "insert failed at start=" << start << ": " << ins.error().message();
  }
  collection->Flush();

  // include_vector=false to isolate scan+scalar throughput.
  IteratorOptions iter_opts;
  iter_opts.include_vector_ = false;
  auto iter = collection->CreateIterator(iter_opts).value();

  auto t0 = std::chrono::steady_clock::now();
  int count = 0;
  while (true) {
    auto r = iter->Next();
    ASSERT_TRUE(r.has_value());
    if (r.value() == nullptr) break;
    count++;
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  EXPECT_EQ(count, N);
  std::cout << "[perf] iterated " << count << " docs (no vector) in " << ms
            << " ms (" << (ms > 0 ? count / ms : count) << " docs/ms)"
            << std::endl;

  collection->Destroy();
}
