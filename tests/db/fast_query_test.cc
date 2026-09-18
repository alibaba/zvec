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
#include <cmath>
#include <limits>
#include <random>
#include <thread>
#include <utility>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"

using namespace zvec;

TEST(FastQueryTest, NativeTopkMatchesQueryBounds) {
  const std::string path = "test_fast_query_topk_bounds";
  FileHelper::RemoveDirectory(path);
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  CollectionSchema schema("fast_topk_bounds");
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, uint32_t{8}, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(created);
  ASSERT_TRUE(created.value()->close().ok());
  auto opened = Collection::Open(path, CollectionOptions{true, true});
  ASSERT_TRUE(opened);
  auto reader = std::move(opened.value());
  std::vector<float> vector(8, 0.0f);
  SearchQuery query;
  query.target_.field_name_ = "vector";
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  for (int topk : {-1, std::numeric_limits<int>::min(), 100001,
                   std::numeric_limits<int>::max()}) {
    query.topk_ = topk;
    auto normal = reader->query(query);
    auto fast = reader->fast_query("vector", vector.data(), nullptr, topk);
    ASSERT_FALSE(normal);
    ASSERT_FALSE(fast);
    EXPECT_EQ(normal.error().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fast.error().code(), normal.error().code());
    EXPECT_EQ(fast.error().message(), normal.error().message());
  }
  // Native query allows zero; the public Python API requires a positive int.
  for (int topk : {0, 1, 100000}) {
    query.topk_ = topk;
    auto normal = reader->query(query);
    auto fast = reader->fast_query("vector", vector.data(), nullptr, topk);
    ASSERT_TRUE(normal);
    ASSERT_TRUE(fast);
    EXPECT_TRUE(normal->empty());
    EXPECT_EQ(fast->ids.size(), static_cast<size_t>(topk));
    for (auto id : fast->ids) EXPECT_EQ(id, -1);
  }
  // Zero must not bypass the supplied input metadata validation.
  auto invalid = reader->fast_query("vector", vector.data(), nullptr, 0, false,
                                    DataType::VECTOR_FP32, 7);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_TRUE(reader->close().ok());
  reader.reset();
  FileHelper::RemoveDirectory(path);
}

TEST(FastQueryTest, ConcurrentFieldsFromFirstQueryThroughClose) {
  const std::string path = "test_fast_query_concurrent_fields";
  FileHelper::RemoveDirectory(path);
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  CollectionSchema schema("fast_query_fields");
  schema.add_field(std::make_shared<FieldSchema>(
      "flat", DataType::VECTOR_FP32, uint32_t{32}, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  schema.add_field(std::make_shared<FieldSchema>(
      "graph", DataType::VECTOR_FP32, uint32_t{64}, false,
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 64)));
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(created) << created.error().message();
  auto writer = std::move(created.value());
  std::mt19937 rng(761);
  std::normal_distribution<float> normal;
  auto make_vector = [&](size_t dimension) {
    std::vector<float> vector(dimension);
    for (auto &value : vector) value = normal(rng);
    return vector;
  };
  std::vector<Doc> docs;
  for (int i = 0; i < 128; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    doc.set<std::vector<float>>("flat", make_vector(32));
    doc.set<std::vector<float>>("graph", make_vector(64));
    docs.push_back(std::move(doc));
  }
  auto inserted = writer->insert(docs);
  ASSERT_TRUE(inserted) << inserted.error().message();
  for (const auto &status : inserted.value()) ASSERT_TRUE(status.ok());
  ASSERT_TRUE(writer->optimize(OptimizeOptions{1}).ok());

  struct Request {
    std::string field;
    std::vector<float> vector;
    QueryParams::Ptr params;
    int topk;
    FastQueryResult expected;
  };
  std::vector<Request> requests{
      {"flat", make_vector(32), nullptr, 1, {}},
      {"graph", make_vector(64), std::make_shared<HnswQueryParams>(32), 7, {}},
      {"graph", make_vector(64), std::make_shared<HnswQueryParams>(64), 13, {}},
      {"graph", make_vector(64), nullptr, 17, {}}};
  // Compute expectations on the writer so the reader below has never searched.
  for (auto &request : requests) {
    SearchQuery query;
    query.topk_ = request.topk;
    query.target_.field_name_ = request.field;
    query.target_.query_params_ = request.params;
    query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(request.vector.data()),
                    request.vector.size() * sizeof(float)));
    auto expected = writer->query(query);
    ASSERT_TRUE(expected) << expected.error().message();
    ASSERT_EQ(expected->size(), request.topk);
    for (const auto &doc : expected.value()) {
      request.expected.ids.push_back(std::stoll(doc->pk()));
      request.expected.scores.push_back(doc->score());
    }
  }
  ASSERT_TRUE(writer->close().ok());
  writer.reset();
  auto opened = Collection::Open(path, CollectionOptions{true, true});
  ASSERT_TRUE(opened) << opened.error().message();
  auto reader = std::move(opened.value());

  std::atomic<bool> start{false}, closing{false}, stop{false}, failed{false};
  std::atomic<size_t> completed{0};
  std::vector<std::thread> threads;
  for (size_t worker = 0; worker < requests.size(); ++worker) {
    threads.emplace_back([&, worker] {
      while (!start.load()) std::this_thread::yield();
      for (size_t repeat = 0; !stop.load(); ++repeat) {
        const auto &request = requests[(worker + repeat) % requests.size()];
        auto result = reader->fast_query(
            request.field, request.vector.data(), request.params, request.topk,
            true, DataType::VECTOR_FP32, request.vector.size());
        if (!result) {
          EXPECT_TRUE(closing.load()) << result.error().message();
          EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
          EXPECT_NE(result.error().message().find("closed"), std::string::npos);
          failed.store(true);
          break;
        }
        EXPECT_EQ(result->ids, request.expected.ids);
        for (size_t i = 0; i < result->scores.size(); ++i) {
          EXPECT_NEAR(result->scores[i], request.expected.scores[i], 1e-4f);
        }
        ++completed;
      }
    });
  }
  start.store(true);
  while (completed.load() < 128 && !failed.load()) std::this_thread::yield();
  closing.store(true);
  const auto closed = reader->close();
  stop.store(true);
  for (auto &thread : threads) thread.join();
  EXPECT_TRUE(closed.ok()) << closed.message();
  auto after_close = reader->fast_query("flat", requests[0].vector.data());
  ASSERT_FALSE(after_close);
  EXPECT_NE(after_close.error().message().find("closed"), std::string::npos);
  reader.reset();
  FileHelper::RemoveDirectory(path);
}

TEST(FastQueryTest, PreservesOrdinalsAfterReopenAndCompaction) {
  const std::string path = "test_fast_query_identity_doc_ids";
  FileHelper::RemoveDirectory(path);
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  CollectionSchema schema("fast_query_identity");
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, uint32_t{32}, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(created) << created.error().message();
  auto writer = std::move(created.value());
  std::vector<Doc> docs;
  for (int i = 0; i < 16; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    doc.set<std::vector<float>>("vector", std::vector<float>(32, i));
    docs.push_back(std::move(doc));
  }
  auto inserted = writer->insert(docs);
  ASSERT_TRUE(inserted) << inserted.error().message();
  for (const auto &status : inserted.value()) ASSERT_TRUE(status.ok());
  ASSERT_TRUE(writer->optimize(OptimizeOptions{1}).ok());
  ASSERT_TRUE(writer->close().ok());
  writer.reset();

  // Reopen with identity IDs, a delete filter, then compacted IDs with gaps.
  for (int phase = 0; phase < 3; ++phase) {
    SCOPED_TRACE(phase);
    if (phase != 0) {
      auto reopened = Collection::Open(path, CollectionOptions{});
      ASSERT_TRUE(reopened) << reopened.error().message();
      writer = std::move(reopened.value());
      if (phase == 1) {
        // Keep ID 0 so the identity check must also detect internal gaps.
        auto deleted = writer->delete_({"7", "9"});
        ASSERT_TRUE(deleted) << deleted.error().message();
        for (const auto &status : deleted.value()) ASSERT_TRUE(status.ok());
      } else {
        ASSERT_TRUE(writer->optimize(OptimizeOptions{1}).ok());
      }
      ASSERT_TRUE(writer->close().ok());
      writer.reset();
    }

    auto opened = Collection::Open(path, CollectionOptions{true, true});
    ASSERT_TRUE(opened) << opened.error().message();
    auto reader = std::move(opened.value());
    std::vector<float> vector(32, 4.1f);
    SearchQuery query;
    query.topk_ = 20;
    query.target_.field_name_ = "vector";
    query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));
    auto expected = reader->query(query);
    ASSERT_TRUE(expected) << expected.error().message();
    ASSERT_EQ(expected->size(), phase == 0 ? 16 : 14);
    for (bool scores : {false, true}) {
      auto actual = reader->fast_query("vector", vector.data(), nullptr,
                                       query.topk_, scores);
      ASSERT_TRUE(actual) << actual.error().message();
      ASSERT_EQ(actual->ids.size(), query.topk_);
      for (size_t i = 0; i < actual->ids.size(); ++i) {
        if (i < expected->size()) {
          EXPECT_EQ(actual->ids[i], std::stoll(expected.value()[i]->pk()));
          if (scores) {
            EXPECT_FLOAT_EQ(actual->scores[i], expected.value()[i]->score());
          }
        } else {
          EXPECT_EQ(actual->ids[i], -1);
          if (scores) EXPECT_TRUE(std::isnan(actual->scores[i]));
        }
      }
    }
    ASSERT_TRUE(reader->close().ok());
  }
  FileHelper::RemoveDirectory(path);
}

TEST(FastQueryTest, ReadsRefineParametersOnEveryCall) {
  const std::string path = "test_fast_search_refine_scale";
  FileHelper::RemoveDirectory(path);
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  CollectionSchema schema("fast_search");
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, uint32_t{32}, false,
      std::make_shared<FlatIndexParams>(MetricType::L2, QuantizeType::INT8)));
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(created) << created.error().message();
  auto writer = std::move(created.value());
  std::mt19937 rng(712);
  std::normal_distribution<float> normal;
  std::vector<Doc> docs;
  for (int i = 0; i < 512; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    std::vector<float> vector(32);
    for (auto &v : vector) v = normal(rng);
    doc.set<std::vector<float>>("vector", vector);
    docs.push_back(std::move(doc));
  }
  auto inserted = writer->insert(docs);
  ASSERT_TRUE(inserted) << inserted.error().message();
  for (const auto &status : inserted.value()) ASSERT_TRUE(status.ok());
  ASSERT_TRUE(writer->optimize(OptimizeOptions{1}).ok());
  ASSERT_TRUE(writer->close().ok());
  auto opened = Collection::Open(path, CollectionOptions{true, true});
  ASSERT_TRUE(opened) << opened.error().message();
  auto reader = std::move(opened.value());
  auto param = std::make_shared<FlatQueryParams>(true);
  for (float scale : {1.0f, 3.0f, 7.0f, 1.0f}) {
    param->set_scale_factor(scale);
    for (int repeat = 0; repeat < 10; ++repeat) {
      std::vector<float> vector(32);
      for (auto &v : vector) v = normal(rng);
      SearchQuery query;
      query.topk_ = 10;
      query.target_.field_name_ = "vector";
      query.target_.set_vector(
          std::string(reinterpret_cast<const char *>(vector.data()),
                      vector.size() * sizeof(float)));
      query.target_.query_params_ =
          std::make_shared<FlatQueryParams>(true, scale);
      auto expected = reader->query(query);
      ASSERT_TRUE(expected) << expected.error().message();
      auto actual =
          reader->fast_query("vector", vector.data(), param, 10, true);
      ASSERT_TRUE(actual) << actual.error().message();
      if (repeat == 0) {
        auto checked = reader->fast_query("vector", vector.data(), param, 10,
                                          true, DataType::VECTOR_FP32, 32);
        ASSERT_TRUE(checked) << checked.error().message();
        EXPECT_EQ(actual->ids, checked->ids);
        EXPECT_EQ(actual->scores, checked->scores);
        // Wrong or partially supplied metadata must fail before vector reads.
        for (const auto &[type, dimension] :
             {std::pair{DataType::VECTOR_FP64, 32U},
              std::pair{DataType::VECTOR_FP32, 31U},
              std::pair{DataType::VECTOR_FP32, 0U},
              std::pair{DataType::UNDEFINED, 32U}}) {
          auto invalid = reader->fast_query("vector", vector.data(), param, 10,
                                            true, type, dimension);
          ASSERT_FALSE(invalid);
          EXPECT_EQ(StatusCode::INVALID_ARGUMENT, invalid.error().code());
        }
      }
      ASSERT_EQ(actual->ids.size(), expected->size());
      for (size_t i = 0; i < expected->size(); ++i) {
        EXPECT_EQ(actual->ids[i], std::stoll(expected.value()[i]->pk()));
        EXPECT_FLOAT_EQ(actual->scores[i], expected.value()[i]->score());
      }
    }
  }
  // Simultaneous calls must not share mutable topk/refiner parameters, even
  // when one caller requests defaults and another switches refinement off.
  const std::vector<int> topks{1, 17, 3, 10};
  const std::vector<QueryParams::Ptr> params{
      nullptr, std::make_shared<FlatQueryParams>(false),
      std::make_shared<FlatQueryParams>(true, 3.0f),
      std::make_shared<FlatQueryParams>(true, 7.0f)};
  std::vector<std::vector<float>> queries(4, std::vector<float>(32));
  std::vector<FastQueryResult> expected;
  for (size_t i = 0; i < queries.size(); ++i) {
    for (auto &value : queries[i]) value = normal(rng);
    auto result = reader->fast_query("vector", queries[i].data(), params[i],
                                     topks[i], true);
    ASSERT_TRUE(result) << result.error().message();
    expected.push_back(std::move(result.value()));
  }
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (size_t worker = 0; worker < queries.size(); ++worker) {
    threads.emplace_back([&, worker] {
      while (!start.load()) std::this_thread::yield();
      for (size_t repeat = 0; repeat < 40; ++repeat) {
        const size_t i = (worker + repeat) % queries.size();
        auto result = reader->fast_query("vector", queries[i].data(), params[i],
                                         topks[i], true);
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(result->ids, expected[i].ids);
        EXPECT_EQ(result->scores, expected[i].scores);
      }
    });
  }
  start.store(true);
  for (auto &thread : threads) thread.join();

  ASSERT_TRUE(reader->close().ok());
  reader.reset();
  FileHelper::RemoveDirectory(path);
}
