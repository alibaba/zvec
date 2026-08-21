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
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
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

// Basic iteration — insert 100 docs, iterate, verify count + PK
TEST_F(IteratorTest, BasicIteration) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  auto collection = std::move(result.value());

  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  auto insert_result = collection->insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value()) << iter_result.error().message();
  auto iter = iter_result.value();

  std::set<std::string> pks;
  int count = 0;
  while (true) {
    auto r = iter->next();
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

  iter->close();
  collection->destroy();
}

// Empty collection — iterator should immediately return EOF
TEST_F(IteratorTest, EmptyCollection) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  auto r = iter->next();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r.value(), nullptr) << "Expected EOF on empty collection";

  iter->close();
  collection->destroy();
}

// Deleted docs are filtered out
TEST_F(IteratorTest, DeletedDocsFiltered) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

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
  auto insert_result = collection->insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  auto delete_result = collection->delete_(pks_to_delete);
  ASSERT_TRUE(delete_result.has_value());

  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  std::set<std::string> deleted_set(pks_to_delete.begin(), pks_to_delete.end());
  int count = 0;
  while (true) {
    auto r = iter->next();
    if (!r.has_value()) {
      FAIL() << "Iterator error: " << r.error().message();
    }
    if (r.value() == nullptr) break;

    auto pk = r.value()->pk();
    EXPECT_EQ(deleted_set.count(pk), 0)
        << "Deleted doc " << pk << " should not appear in iteration";
    count++;
  }

  EXPECT_EQ(count, N - static_cast<int>(pks_to_delete.size()));

  iter->close();
  collection->destroy();
}

// Iterator after Close() returns error
TEST_F(IteratorTest, CloseThenNext) {
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
  auto insert_result = collection->insert(docs);
  ASSERT_TRUE(insert_result.has_value());

  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  iter->close();

  auto r = iter->next();
  EXPECT_FALSE(r.has_value()) << "Expected error after Close()";

  collection->destroy();
}

// Iterator with include_vector=true — verify vector fields are present
TEST_F(IteratorTest, IncludeVector) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 10;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  auto insert_result = collection->insert(docs);
  ASSERT_TRUE(insert_result.has_value());
  collection->flush();

  IteratorOptions iter_opts;
  iter_opts.include_vector_ = true;
  auto iter_result = collection->create_iterator(iter_opts);
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->next();
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
  iter->close();
  collection->destroy();
}

// Iterator with include_vector=false — verify no vector fields
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
  collection->insert(docs);
  collection->flush();

  IteratorOptions iter_opts;
  iter_opts.include_vector_ = false;
  auto iter_result = collection->create_iterator(iter_opts);
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->next();
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
  iter->close();
  collection->destroy();
}

// Only the requested output fields are materialized.
TEST_F(IteratorTest, OutputFieldsSelection) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 20;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  collection->insert(docs);
  collection->flush();

  // Request only the "int32" scalar field.
  IteratorOptions iter_opts;
  iter_opts.output_fields_ = std::vector<std::string>{"int32"};
  iter_opts.include_vector_ = false;
  auto iter_result = collection->create_iterator(iter_opts);
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    auto doc = r.value();
    // Requested field present; non-requested scalar fields absent.
    EXPECT_TRUE(doc->has("int32"));
    EXPECT_FALSE(doc->has("string"));
    EXPECT_FALSE(doc->has("array_int32"));
    count++;
  }
  EXPECT_EQ(count, N);

  iter->close();
  collection->destroy();
}

TEST_F(IteratorTest, InvalidOutputFieldsRejected) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // Unknown field is rejected.
  {
    IteratorOptions iter_opts;
    iter_opts.output_fields_ = std::vector<std::string>{"no_such_field"};
    auto r = collection->create_iterator(iter_opts);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), StatusCode::INVALID_ARGUMENT);
  }
  // Duplicate field is rejected.
  {
    IteratorOptions iter_opts;
    iter_opts.output_fields_ = std::vector<std::string>{"int32", "int32"};
    auto r = collection->create_iterator(iter_opts);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), StatusCode::INVALID_ARGUMENT);
  }
  // Vector field names are rejected (output_fields accepts scalar fields
  // only; vectors are controlled by include_vector_).
  {
    IteratorOptions iter_opts;
    iter_opts.output_fields_ = std::vector<std::string>{"dense_fp32"};
    auto r = collection->create_iterator(iter_opts);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  collection->destroy();
}

// Scalar type mapping — every scalar/array Arrow type in CreateNormalSchema
// (8 base types + 8 array types).
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
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  auto r = iter->next();
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

  iter->close();
  collection->destroy();
}

// Integration — 1000 docs, verify count + PK + scalar + vector values.
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
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  std::set<std::string> seen_pks;
  while (true) {
    auto r = iter->next();
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
  iter->close();
  collection->destroy();
}

// Concurrency — docs inserted after iterator creation are not visible.
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
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Consume one doc to establish the snapshot, then insert concurrently.
  auto first = iter->next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);

  std::atomic<bool> writer_failed{false};
  std::thread writer([&]() {
    std::vector<Doc> more;
    for (int i = N; i < N + 200; i++) {
      more.push_back(TestHelper::CreateDoc(i, *schema));
    }
    // No flush here: the new docs stay in the writing segment, invisible
    // to the open iterator; the fresh iterator below still sees them
    // because create_iterator seals the writing segment itself.
    if (!collection->insert(more).has_value()) writer_failed = true;
  });

  int count = 1;  // already consumed one
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count++;
  }
  writer.join();

  EXPECT_FALSE(writer_failed);
  // Snapshot was taken at creation → only the original N are visible.
  EXPECT_EQ(count, N);

  // A fresh iterator sees all N + 200.
  auto iter2 = collection->create_iterator().value();
  int count2 = 0;
  while (true) {
    auto r = iter2->next();
    ASSERT_TRUE(r.has_value());
    if (r.value() == nullptr) break;
    count2++;
  }
  EXPECT_EQ(count2, N + 200);

  iter->close();
  iter2->close();
  collection->destroy();
}

// Concurrency — optimize is rejected while an iterator is open and
// succeeds after it is closed.
TEST_F(IteratorTest, OptimizeRejectedWhileIteratorOpen) {
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
  ASSERT_TRUE(collection->insert(b1).has_value());
  collection->flush();
  ASSERT_TRUE(collection->insert(b2).has_value());
  collection->flush();

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  // Optimize is rejected while the iterator is open.
  auto optimize_status = collection->optimize();
  EXPECT_FALSE(optimize_status.ok());

  // The full snapshot stays readable.
  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    EXPECT_FALSE(r.value()->pk().empty());
    count++;
  }
  EXPECT_EQ(count, N);
  iter->close();

  // After the iterator is closed, Optimize succeeds and data is intact.
  ASSERT_TRUE(collection->optimize().ok());
  auto iter2 = collection->create_iterator().value();
  int count2 = 0;
  while (true) {
    auto r = iter2->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count2++;
  }
  EXPECT_EQ(count2, N);
  iter2->close();
  collection->destroy();
}

// Schema changes are rejected while an iterator is open (flush is not)
// and succeed again after it is closed.
TEST_F(IteratorTest, DdlRejectedWhileIteratorOpen) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter = collection->create_iterator().value();

  auto index_params = std::make_shared<HnswIndexParams>(MetricType::IP);
  auto create_status = collection->create_index("dense_fp32", index_params);
  ASSERT_FALSE(create_status.ok());
  EXPECT_EQ(create_status.code(), StatusCode::FAILED_PRECONDITION);
  EXPECT_FALSE(collection->drop_index("dense_fp32").ok());
  auto new_field =
      std::make_shared<FieldSchema>("added_int32", DataType::INT32, false);
  EXPECT_FALSE(collection->add_column(new_field, "int32", {}).ok());
  EXPECT_FALSE(collection->drop_column("int32").ok());
  EXPECT_FALSE(
      collection->alter_column("int32", "int32_renamed", nullptr, {}).ok());
  // flush is not blocked by open iterators (it only touches the writing
  // segment, never the snapshot).
  EXPECT_TRUE(collection->flush().ok());

  // The rejected operations left the snapshot untouched.
  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count++;
  }
  EXPECT_EQ(count, N);
  iter->close();

  // After the iterator is closed, DDL succeeds.
  ASSERT_TRUE(collection->create_index("dense_fp32", index_params).ok());
  collection->destroy();
}

// create_iterator is rejected while optimize runs instead of blocking on
// maintenance_mtx_ (iterators and maintenance ops are mutually exclusive).
TEST_F(IteratorTest, CreateIteratorRejectedWhileOptimizeRunning) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  // Two flushed chunks give optimize more than one segment to compact.
  const int N = 10000;
  for (int base = 0; base < N; base += N / 2) {
    for (int batch = 0; batch < N / 2; batch += 1000) {
      std::vector<Doc> docs;
      for (int i = base + batch; i < base + batch + 1000; i++)
        docs.push_back(TestHelper::CreateDoc(i, *schema));
      ASSERT_TRUE(collection->insert(docs).has_value());
    }
    collection->flush();
  }

  std::atomic<bool> optimize_done{false};
  std::atomic<bool> optimize_ok{false};
  std::thread optimizer([&] {
    optimize_ok.store(collection->optimize().ok());
    optimize_done.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // While optimize is still running, create_iterator must fail fast
  // instead of blocking on maintenance_mtx_. If optimize finished early
  // on a fast machine, the admission window simply never opened.
  if (!optimize_done.load()) {
    LOG_INFO("admission window active: optimize still holds maintenance_mtx_");
    auto r = collection->create_iterator();
    if (r.has_value()) {
      // Optimize finished between the check above and this call.
      r.value()->close();
    } else {
      EXPECT_EQ(r.error().code(), StatusCode::FAILED_PRECONDITION);
    }
  }

  optimizer.join();
  ASSERT_TRUE(optimize_ok.load());

  auto iter_result = collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value());
  iter_result.value()->close();
  collection->destroy();
}

// The public close() and destroy() fail fast while an iterator is open.
TEST_F(IteratorTest, CloseAndDestroyRejectedWhileIteratorOpen) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter = collection->create_iterator().value();

  EXPECT_FALSE(collection->close().ok());
  EXPECT_FALSE(collection->destroy().ok());

  // The full snapshot stays readable.
  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count++;
  }
  EXPECT_EQ(count, N);
  iter->close();

  // After the iterator is closed, destroy succeeds.
  ASSERT_TRUE(collection->destroy().ok());
}

// The destructor cannot report errors, so unlike the public close() it
// waits for open iterators instead of rejecting.
TEST_F(IteratorTest, DestructorWaitsForIterators) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter = collection->create_iterator().value();

  std::atomic<bool> destroyed{false};
  std::thread destroyer([&] {
    collection.reset();  // last reference -> destructor waits
    destroyed.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(destroyed.load());  // still waiting for the iterator

  // The iterator keeps working while the destructor is waiting.
  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    count++;
  }
  EXPECT_EQ(count, N);
  iter->close();

  destroyer.join();
  EXPECT_TRUE(destroyed.load());

  // The destructor closed (but did not destroy) the collection.
  auto reopened = Collection::Open(iter_test_path, options);
  ASSERT_TRUE(reopened.has_value());
  reopened.value()->destroy();
}

TEST_F(IteratorTest, CloseThenCreateIteratorRejected) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  std::vector<Doc> docs;
  for (int i = 0; i < 10; i++)
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  ASSERT_TRUE(collection->close().ok());

  auto iter_result = collection->create_iterator();
  EXPECT_FALSE(iter_result.has_value());
}

// Destroying an unclosed iterator releases the active-iterator slot too.
TEST_F(IteratorTest, IteratorResetReleasesSlot) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  std::vector<Doc> docs;
  for (int i = 0; i < 10; i++)
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  {
    auto iter = collection->create_iterator().value();
    ASSERT_TRUE(iter->next().has_value());
    EXPECT_FALSE(collection->optimize().ok());  // rejected while alive
  }  // iter destroyed here (no explicit close)

  EXPECT_TRUE(collection->optimize().ok());  // slot released by destructor
  collection->destroy();
}

TEST_F(IteratorTest, MultipleIteratorsCoexist) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 100;
  std::vector<Doc> docs;
  for (int i = 0; i < N; i++) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->insert(docs).has_value());
  collection->flush();

  auto iter1 = collection->create_iterator().value();
  auto iter2 = collection->create_iterator().value();

  auto count_all = [](const DocIterator::Ptr &it) {
    int count = 0;
    while (true) {
      auto r = it->next();
      EXPECT_TRUE(r.has_value());
      if (!r.has_value() || r.value() == nullptr) break;
      count++;
    }
    return count;
  };
  EXPECT_EQ(count_all(iter1), N);
  EXPECT_EQ(count_all(iter2), N);

  iter1->close();
  // One iterator is still open → exclusive operations stay rejected.
  EXPECT_FALSE(collection->optimize().ok());

  iter2->close();
  ASSERT_TRUE(collection->optimize().ok());
  collection->destroy();
}

// Performance — iterate 100k docs; reports elapsed time, asserts count.
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
    auto ins = collection->insert(docs);
    ASSERT_TRUE(ins.has_value())
        << "insert failed at start=" << start << ": " << ins.error().message();
  }
  collection->flush();

  // include_vector=false to isolate scan+scalar throughput.
  IteratorOptions iter_opts;
  iter_opts.include_vector_ = false;
  auto iter = collection->create_iterator(iter_opts).value();

  auto t0 = std::chrono::steady_clock::now();
  int count = 0;
  while (true) {
    auto r = iter->next();
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

  iter->close();
  collection->destroy();
}

// Windowed materialization on the Parquet store: one ReadNext returns a
// 10000-row row group, spanning several 4096-row windows.
TEST_F(IteratorTest, ParquetLargeRowGroupWindowedMaterialization) {
  auto schema = TestHelper::CreateNormalSchema();
  CollectionOptions options;
  options.read_only_ = false;
  // Parquet forward store (mmap disabled selects the buffer-pool store).
  options.enable_mmap_ = false;

  auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto collection = std::move(result.value());

  const int N = 10000;  // single row group spanning multiple windows
  const int kBatch = 1000;
  for (int base = 0; base < N; base += kBatch) {
    std::vector<Doc> docs;
    docs.reserve(kBatch);
    for (int i = base; i < base + kBatch; i++) {
      docs.push_back(TestHelper::CreateDoc(i, *schema));
    }
    ASSERT_TRUE(collection->insert(docs).has_value());
  }
  collection->flush();

  auto iter_result = collection->create_iterator();  // include_vector = true
  ASSERT_TRUE(iter_result.has_value());
  auto iter = iter_result.value();

  int count = 0;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    auto doc = r.value();

    // Vector values must stay aligned with the scalar id across every
    // window boundary (window misalignment is the failure mode here).
    auto id32 = doc->get<int32_t>("int32");
    ASSERT_TRUE(id32.has_value());
    uint64_t id = static_cast<uint64_t>(*id32);
    auto vec = doc->get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vec.has_value()) << "vector missing for id " << id;
    EXPECT_FLOAT_EQ((*vec)[0], float(id + 0.1))
        << "window misalignment at id " << id;
    count++;
  }
  EXPECT_EQ(count, N);

  iter->close();
  collection->destroy();
}

// Read-only collection iteration — must work without flushing (the
// read-only path reads the writing segment directly).
TEST_F(IteratorTest, ReadOnlyCollectionIteration) {
  auto schema = TestHelper::CreateNormalSchema();
  const int N = 50;

  {
    CollectionOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = true;
    auto result = Collection::CreateAndOpen(iter_test_path, *schema, options);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = std::move(result.value());
    std::vector<Doc> docs;
    for (int i = 0; i < N; i++) {
      docs.push_back(TestHelper::CreateDoc(i, *schema));
    }
    ASSERT_TRUE(collection->insert(docs).has_value());
    collection->flush();
  }  // writable collection closed (dtor flushes + releases lock)

  CollectionOptions ro_options;
  ro_options.read_only_ = true;
  auto ro_result = Collection::Open(iter_test_path, ro_options);
  ASSERT_TRUE(ro_result.has_value()) << ro_result.error().message();
  auto ro_collection = std::move(ro_result.value());

  auto iter_result = ro_collection->create_iterator();
  ASSERT_TRUE(iter_result.has_value()) << iter_result.error().message();
  auto iter = iter_result.value();

  int count = 0;
  std::set<std::string> pks;
  while (true) {
    auto r = iter->next();
    ASSERT_TRUE(r.has_value()) << r.error().message();
    if (r.value() == nullptr) break;
    pks.insert(r.value()->pk());
    count++;
  }
  EXPECT_EQ(count, N);
  EXPECT_EQ(pks.size(), N);
  // No Destroy(): a read-only collection cannot be destroyed; TearDown cleans
  // up.
}
