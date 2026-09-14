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
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/query_params.h>
#include <zvec/db/reranker.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

namespace zvec {
namespace {

#if defined(__ANDROID__) || \
    (defined(__APPLE__) && (TARGET_OS_IOS || TARGET_OS_SIMULATOR))
static_assert(DISKANN_SUPPORTED == 1,
              "Android and iOS must compile the DiskAnn mobile contract");
#endif

#if DISKANN_SUPPORTED

constexpr char kCollectionPath[] = "diskann_mobile_collection";
constexpr char kFp32Field[] = "dense_fp32";
constexpr char kFp16Field[] = "dense_fp16";
constexpr char kDynamicField[] = "dense_dynamic";
constexpr char kGroupByField[] = "dense_group_by";
constexpr size_t kDimension = 16;
constexpr uint64_t kDocCount = 48;

std::vector<float> MakeFp32Vector(uint64_t doc_id) {
  std::vector<float> result(kDimension);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<float>(((doc_id + 3) * (i + 5)) % 23) / 23.0F +
                static_cast<float>(doc_id) * 0.01F;
  }
  return result;
}

std::vector<float16_t> MakeFp16Vector(uint64_t doc_id) {
  auto fp32 = MakeFp32Vector(doc_id);
  std::vector<float16_t> result;
  result.reserve(fp32.size());
  for (float value : fp32) {
    result.emplace_back(value);
  }
  return result;
}

CollectionSchema::Ptr MakeSchema(MetricType metric, bool include_fp16 = false,
                                 bool include_dynamic = false,
                                 bool include_group_by = false) {
  auto schema = std::make_shared<CollectionSchema>("diskann_mobile");
  schema->set_max_doc_count_per_segment(1000);
  EXPECT_TRUE(schema
                  ->add_field(std::make_shared<FieldSchema>(
                      "category", DataType::INT32, false))
                  .ok());
  EXPECT_TRUE(schema
                  ->add_field(std::make_shared<FieldSchema>(
                      "name", DataType::STRING, false))
                  .ok());

  auto diskann = std::make_shared<DiskAnnIndexParams>(metric, 16, 32, 2);
  EXPECT_TRUE(
      schema
          ->add_field(std::make_shared<FieldSchema>(
              kFp32Field, DataType::VECTOR_FP32, kDimension, false, diskann))
          .ok());
  if (include_group_by) {
    EXPECT_TRUE(schema
                    ->add_field(std::make_shared<FieldSchema>(
                        kGroupByField, DataType::VECTOR_FP32, kDimension, false,
                        std::make_shared<FlatIndexParams>(metric)))
                    .ok());
  }
  if (include_fp16) {
    EXPECT_TRUE(schema
                    ->add_field(std::make_shared<FieldSchema>(
                        kFp16Field, DataType::VECTOR_FP16, kDimension, false,
                        diskann->clone()))
                    .ok());
  }
  if (include_dynamic) {
    EXPECT_TRUE(
        schema
            ->add_field(std::make_shared<FieldSchema>(
                kDynamicField, DataType::VECTOR_FP32, kDimension, false))
            .ok());
  }
  return schema;
}

Doc MakeDoc(uint64_t doc_id, bool include_fp16 = false,
            bool include_dynamic = false, bool include_group_by = false,
            std::string pk = "") {
  Doc doc;
  doc.set_pk(pk.empty() ? "pk_" + std::to_string(doc_id) : std::move(pk));
  doc.set<int32_t>("category", static_cast<int32_t>(doc_id % 4));
  doc.set<std::string>("name", "name_" + std::to_string(doc_id));
  doc.set<std::vector<float>>(kFp32Field, MakeFp32Vector(doc_id));
  if (include_group_by) {
    doc.set<std::vector<float>>(kGroupByField, MakeFp32Vector(doc_id));
  }
  if (include_fp16) {
    doc.set<std::vector<float16_t>>(kFp16Field, MakeFp16Vector(doc_id));
  }
  if (include_dynamic) {
    doc.set<std::vector<float>>(kDynamicField, MakeFp32Vector(doc_id + 7));
  }
  return doc;
}

std::vector<Doc> MakeDocs(uint64_t begin, uint64_t end,
                          bool include_fp16 = false,
                          bool include_dynamic = false,
                          bool include_group_by = false) {
  std::vector<Doc> docs;
  docs.reserve(end - begin);
  for (uint64_t doc_id = begin; doc_id < end; ++doc_id) {
    docs.emplace_back(
        MakeDoc(doc_id, include_fp16, include_dynamic, include_group_by));
  }
  return docs;
}

SearchQuery MakeFp32Query(uint64_t doc_id, const std::string &field,
                          int topk = 5) {
  auto vector = field == kDynamicField ? MakeFp32Vector(doc_id + 7)
                                       : MakeFp32Vector(doc_id);
  SearchQuery query;
  query.topk_ = topk;
  query.target_.field_name_ = field;
  query.target_.query_params_ = std::make_shared<DiskAnnQueryParams>(32);
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  return query;
}

SearchQuery MakeFp16Query(uint64_t doc_id, int topk = 5) {
  auto vector = MakeFp16Vector(doc_id);
  SearchQuery query;
  query.topk_ = topk;
  query.target_.field_name_ = kFp16Field;
  query.target_.query_params_ = std::make_shared<DiskAnnQueryParams>(32);
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float16_t)));
  return query;
}

SearchQuery MakeFlatQuery(uint64_t doc_id, int topk = 5) {
  auto vector = MakeFp32Vector(doc_id);
  SearchQuery query;
  query.topk_ = topk;
  query.target_.field_name_ = kGroupByField;
  query.target_.query_params_ = std::make_shared<FlatQueryParams>();
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  return query;
}

std::vector<std::string> SortedPks(const DocPtrList &docs) {
  std::vector<std::string> pks;
  pks.reserve(docs.size());
  for (const auto &doc : docs) {
    if (doc != nullptr) {
      pks.emplace_back(doc->pk());
    }
  }
  std::sort(pks.begin(), pks.end());
  return pks;
}

bool FetchContainsPk(const Result<DocPtrMap> &result, const std::string &pk) {
  if (!result.has_value() || result->size() != 1) {
    return false;
  }
  auto it = result->find(pk);
  return it != result->end() && it->second != nullptr;
}

::testing::AssertionResult WriteSucceeded(const Result<WriteResults> &result,
                                          size_t expected_count) {
  if (!result.has_value()) {
    return ::testing::AssertionFailure() << result.error().message();
  }
  if (result->size() != expected_count) {
    return ::testing::AssertionFailure()
           << "expected " << expected_count << " write results, got "
           << result->size();
  }
  for (size_t i = 0; i < result->size(); ++i) {
    if (!result->at(i).ok()) {
      return ::testing::AssertionFailure()
             << "write " << i << " failed: " << result->at(i).message();
    }
  }
  return ::testing::AssertionSuccess();
}

class DiskAnnMobileCollectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::FileHelper::RemoveDirectory(kCollectionPath);
  }

  void TearDown() override {
    ailego::FileHelper::RemoveDirectory(kCollectionPath);
  }

  static CollectionOptions Options(bool read_only = false) {
    return CollectionOptions{read_only, true, 32 * 1024 * 1024};
  }
};

TEST_F(DiskAnnMobileCollectionTest, PublicCollectionApiLifecycle) {
  auto schema = MakeSchema(MetricType::L2, false, true);
  auto options = Options();
  auto create_result =
      Collection::CreateAndOpen(kCollectionPath, *schema, options);
  ASSERT_TRUE(create_result.has_value()) << create_result.error().message();
  auto collection = std::move(create_result.value());

  auto path_result = collection->path();
  ASSERT_TRUE(path_result.has_value()) << path_result.error().message();
  EXPECT_EQ(*path_result, kCollectionPath);
  auto schema_result = collection->schema();
  ASSERT_TRUE(schema_result.has_value()) << schema_result.error().message();
  EXPECT_EQ(*schema_result, *schema);
  auto options_result = collection->options();
  ASSERT_TRUE(options_result.has_value()) << options_result.error().message();
  EXPECT_EQ(*options_result, options);
  auto empty_stats = collection->stats();
  ASSERT_TRUE(empty_stats.has_value()) << empty_stats.error().message();
  EXPECT_EQ(empty_stats->doc_count, 0u);

  auto docs = MakeDocs(0, 32, false, true);
  ASSERT_TRUE(WriteSucceeded(collection->insert(docs), docs.size()));
  ASSERT_TRUE(collection->flush().ok());
  auto flushed_stats = collection->stats();
  ASSERT_TRUE(flushed_stats.has_value()) << flushed_stats.error().message();
  ASSERT_EQ(flushed_stats->index_completeness[kFp32Field], 0);
  ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());
  auto optimized_stats = collection->stats();
  ASSERT_TRUE(optimized_stats.has_value()) << optimized_stats.error().message();
  ASSERT_EQ(optimized_stats->index_completeness[kFp32Field], 1);

  auto fetch = collection->fetch(
      {"pk_8"}, std::vector<std::string>{"category", "name"}, false);
  ASSERT_TRUE(fetch.has_value()) << fetch.error().message();
  ASSERT_TRUE(FetchContainsPk(fetch, "pk_8"));
  EXPECT_TRUE(fetch->at("pk_8")->has("category"));
  EXPECT_TRUE(fetch->at("pk_8")->has("name"));
  EXPECT_FALSE(fetch->at("pk_8")->has(kFp32Field));

  std::vector<Doc> update_docs{MakeDoc(100, false, true, false, "pk_0")};
  ASSERT_TRUE(
      WriteSucceeded(collection->update(update_docs), update_docs.size()));

  std::vector<Doc> upsert_docs{MakeDoc(101, false, true, false, "pk_1"),
                               MakeDoc(32, false, true)};
  ASSERT_TRUE(
      WriteSucceeded(collection->upsert(upsert_docs), upsert_docs.size()));
  ASSERT_TRUE(WriteSucceeded(collection->delete_({"pk_2"}), 1));
  ASSERT_TRUE(collection->delete_by_filter("category = 3").ok());
  auto deleted_fetch = collection->fetch({"pk_2", "pk_3"});
  ASSERT_TRUE(deleted_fetch.has_value()) << deleted_fetch.error().message();
  ASSERT_EQ(deleted_fetch->size(), 2u);
  auto deleted_pk2 = deleted_fetch->find("pk_2");
  auto deleted_pk3 = deleted_fetch->find("pk_3");
  ASSERT_NE(deleted_pk2, deleted_fetch->end());
  ASSERT_NE(deleted_pk3, deleted_fetch->end());
  EXPECT_EQ(deleted_pk2->second, nullptr);
  EXPECT_EQ(deleted_pk3->second, nullptr);

  auto added_field =
      std::make_shared<FieldSchema>("category_copy", DataType::INT32, false);
  ASSERT_TRUE(collection->add_column(added_field, "category").ok());
  ASSERT_TRUE(
      collection->alter_column("category_copy", "category_renamed").ok());
  ASSERT_TRUE(collection->drop_column("category_renamed").ok());

  auto dynamic_index =
      std::make_shared<DiskAnnIndexParams>(MetricType::L2, 16, 32, 2);
  ASSERT_TRUE(collection->create_index(kDynamicField, dynamic_index).ok());
  ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());
  auto dynamic_result = collection->query(MakeFp32Query(8, kDynamicField, 32));
  ASSERT_TRUE(dynamic_result.has_value()) << dynamic_result.error().message();
  ASSERT_FALSE(dynamic_result->empty());
  ASSERT_TRUE(collection->drop_index(kDynamicField).ok());

  ASSERT_TRUE(collection->flush().ok());
  collection.reset();

  auto reopen_result = Collection::Open(kCollectionPath, options);
  ASSERT_TRUE(reopen_result.has_value()) << reopen_result.error().message();
  collection = std::move(reopen_result.value());
  auto primary_result = collection->query(MakeFp32Query(8, kFp32Field));
  ASSERT_TRUE(primary_result.has_value()) << primary_result.error().message();
  ASSERT_FALSE(primary_result->empty());
  auto reopened_stats = collection->stats();
  ASSERT_TRUE(reopened_stats.has_value()) << reopened_stats.error().message();
  EXPECT_LT(reopened_stats->doc_count, 33u);
  collection.reset();

  auto read_only_result = Collection::Open(kCollectionPath, Options(true));
  ASSERT_TRUE(read_only_result.has_value())
      << read_only_result.error().message();
  collection = std::move(read_only_result.value());
  auto read_only_query = collection->query(MakeFp32Query(8, kFp32Field));
  ASSERT_TRUE(read_only_query.has_value()) << read_only_query.error().message();
  ASSERT_FALSE(read_only_query->empty());
  auto rejected_docs = MakeDocs(40, 41, false, true);
  EXPECT_FALSE(collection->insert(rejected_docs).has_value());
  EXPECT_FALSE(collection->optimize().ok());
  collection.reset();

  reopen_result = Collection::Open(kCollectionPath, options);
  ASSERT_TRUE(reopen_result.has_value()) << reopen_result.error().message();
  collection = std::move(reopen_result.value());
  ASSERT_TRUE(collection->destroy().ok());
  EXPECT_FALSE(collection->stats().has_value());
  EXPECT_FALSE(Collection::Open(kCollectionPath, options).has_value());
}

TEST_F(DiskAnnMobileCollectionTest, CompleteQuerySurfaceAndMetricMatrix) {
  for (MetricType metric :
       {MetricType::L2, MetricType::IP, MetricType::COSINE}) {
    SCOPED_TRACE(static_cast<uint32_t>(metric));
    ailego::FileHelper::RemoveDirectory(kCollectionPath);
    auto schema = MakeSchema(metric, true, false, true);
    auto create_result =
        Collection::CreateAndOpen(kCollectionPath, *schema, Options());
    ASSERT_TRUE(create_result.has_value()) << create_result.error().message();
    auto collection = std::move(create_result.value());

    auto docs = MakeDocs(0, kDocCount, true, false, true);
    ASSERT_TRUE(WriteSucceeded(collection->insert(docs), docs.size()));
    ASSERT_TRUE(collection->flush().ok());
    ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());

    auto fp32_query = MakeFp32Query(12, kFp32Field, 8);
    fp32_query.filter_ = "category = 0";
    fp32_query.include_vector_ = true;
    fp32_query.include_doc_id_ = true;
    fp32_query.output_fields_ = std::vector<std::string>{"category", "name"};
    auto fp32_result = collection->query(fp32_query);
    ASSERT_TRUE(fp32_result.has_value()) << fp32_result.error().message();
    ASSERT_FALSE(fp32_result->empty());
    for (const auto &doc : *fp32_result) {
      ASSERT_NE(doc, nullptr);
      ASSERT_TRUE(std::isfinite(doc->score()));
      auto category = doc->get<int32_t>("category");
      ASSERT_TRUE(category.has_value());
      EXPECT_EQ(category.value(), 0);
      EXPECT_TRUE(doc->has("name"));
      EXPECT_TRUE(doc->has(kFp32Field));
    }
    EXPECT_TRUE(std::any_of(fp32_result->begin(), fp32_result->end(),
                            [](const Doc::Ptr &doc) {
                              return doc != nullptr && doc->doc_id() != 0;
                            }));
    auto default_params_query = MakeFp32Query(12, kFp32Field);
    default_params_query.target_.query_params_.reset();
    auto default_params_result = collection->query(default_params_query);
    ASSERT_TRUE(default_params_result.has_value())
        << default_params_result.error().message();
    ASSERT_FALSE(default_params_result->empty());

    SearchQuery scalar_query;
    scalar_query.topk_ = 5;
    scalar_query.filter_ = "category = 1";
    scalar_query.output_fields_ = std::vector<std::string>{"category", "name"};
    auto scalar_result = collection->query(scalar_query);
    ASSERT_TRUE(scalar_result.has_value()) << scalar_result.error().message();
    ASSERT_EQ(scalar_result->size(), 5u);
    for (const auto &doc : *scalar_result) {
      ASSERT_NE(doc, nullptr);
      auto category = doc->get<int32_t>("category");
      ASSERT_TRUE(category.has_value());
      EXPECT_EQ(category.value(), 1);
    }

    auto fp16_result = collection->query(MakeFp16Query(12, 8));
    ASSERT_TRUE(fp16_result.has_value()) << fp16_result.error().message();
    ASSERT_FALSE(fp16_result->empty());

    // Use the exhaustive DiskANN path for both sides of radius validation.
    // ANN searches may produce different candidate sets, and result order is
    // not a suitable substitute for computing the actual score extrema.
    auto radius_baseline_query = MakeFp32Query(12, kFp32Field, 8);
    radius_baseline_query.filter_ = "category = 0";
    radius_baseline_query.target_.query_params_->set_is_linear(true);
    auto radius_baseline_result = collection->query(radius_baseline_query);
    ASSERT_TRUE(radius_baseline_result.has_value())
        << radius_baseline_result.error().message();
    ASSERT_GE(radius_baseline_result->size(), 2u);
    std::vector<float> radius_baseline_scores;
    radius_baseline_scores.reserve(radius_baseline_result->size());
    for (const auto &doc : *radius_baseline_result) {
      ASSERT_NE(doc, nullptr);
      ASSERT_TRUE(std::isfinite(doc->score()));
      radius_baseline_scores.emplace_back(doc->score());
    }
    const auto [min_score, max_score] = std::minmax_element(
        radius_baseline_scores.begin(), radius_baseline_scores.end());
    float radius = (*min_score + *max_score) / 2.0F;
    if (radius <= 0.0F) {
      radius = 0.001F;
    }
    const auto is_within_radius = [metric, radius](float score) {
      return metric == MetricType::IP ? score >= radius : score <= radius;
    };
    ASSERT_TRUE(std::any_of(radius_baseline_scores.begin(),
                            radius_baseline_scores.end(), is_within_radius));
    auto radius_query = MakeFp32Query(12, kFp32Field, 8);
    radius_query.filter_ = "category = 0";
    radius_query.target_.query_params_->set_radius(radius);
    radius_query.target_.query_params_->set_is_linear(true);
    auto radius_result = collection->query(radius_query);
    ASSERT_TRUE(radius_result.has_value()) << radius_result.error().message();
    ASSERT_FALSE(radius_result->empty());
    for (const auto &doc : *radius_result) {
      ASSERT_NE(doc, nullptr);
      ASSERT_TRUE(std::isfinite(doc->score()));
      EXPECT_TRUE(is_within_radius(doc->score()));
    }

    MultiQuery multi_query;
    multi_query.topk = 8;
    multi_query.filter = "category = 0";
    multi_query.include_vector = true;
    multi_query.include_doc_id_ = true;
    multi_query.output_fields = std::vector<std::string>{"category", "name"};
    multi_query.rerank = reranker::RrfParams{60};
    for (uint64_t doc_id : {12u, 20u}) {
      auto search_query = MakeFp32Query(doc_id, kFp32Field, 16);
      SubQuery sub_query;
      sub_query.target_ = std::move(search_query.target_);
      sub_query.num_candidates_ = 16;
      multi_query.queries.emplace_back(std::move(sub_query));
    }
    auto multi_result = collection->query(multi_query);
    ASSERT_TRUE(multi_result.has_value()) << multi_result.error().message();
    ASSERT_FALSE(multi_result->empty());
    EXPECT_LE(multi_result->size(), 8u);
    for (const auto &doc : *multi_result) {
      ASSERT_NE(doc, nullptr);
      EXPECT_TRUE(doc->has("category"));
      EXPECT_TRUE(doc->has("name"));
      EXPECT_TRUE(doc->has(kFp32Field));
      auto category = doc->get<int32_t>("category");
      ASSERT_TRUE(category.has_value());
      EXPECT_EQ(category.value(), 0);
    }
    EXPECT_TRUE(std::any_of(multi_result->begin(), multi_result->end(),
                            [](const Doc::Ptr &doc) {
                              return doc != nullptr && doc->doc_id() != 0;
                            }));

    GroupByVectorQuery group_query;
    group_query.target_ = MakeFlatQuery(12, 8).target_;
    group_query.filter_ = "category >= 0";
    group_query.group_by_field_name_ = "category";
    group_query.group_count_ = 4;
    group_query.topk_per_group_ = 2;
    group_query.include_vector_ = true;
    group_query.output_fields_ = std::vector<std::string>{"category", "name"};
    auto group_result = collection->group_by_query(group_query);
    ASSERT_TRUE(group_result.has_value()) << group_result.error().message();
    ASSERT_FALSE(group_result->empty());
    EXPECT_LE(group_result->size(), 4u);
    for (const auto &group : *group_result) {
      EXPECT_FALSE(group.group_by_value_.empty());
      EXPECT_FALSE(group.docs_.empty());
      EXPECT_LE(group.docs_.size(), 2u);
      for (const auto &doc : group.docs_) {
        EXPECT_TRUE(doc.has("category"));
        EXPECT_TRUE(doc.has("name"));
        EXPECT_TRUE(doc.has(kGroupByField));
      }
    }

    auto selected_fetch = collection->fetch(
        {"pk_12"}, std::vector<std::string>{"category"}, false);
    ASSERT_TRUE(selected_fetch.has_value()) << selected_fetch.error().message();
    ASSERT_TRUE(FetchContainsPk(selected_fetch, "pk_12"));
    EXPECT_TRUE(selected_fetch->at("pk_12")->has("category"));
    EXPECT_FALSE(selected_fetch->at("pk_12")->has("name"));
    EXPECT_FALSE(selected_fetch->at("pk_12")->has(kFp32Field));

    collection.reset();
    auto reopen_result = Collection::Open(kCollectionPath, Options());
    ASSERT_TRUE(reopen_result.has_value()) << reopen_result.error().message();
    collection = std::move(reopen_result.value());
    auto reopened_query = collection->query(MakeFp32Query(12, kFp32Field));
    ASSERT_TRUE(reopened_query.has_value()) << reopened_query.error().message();
    ASSERT_FALSE(reopened_query->empty());
  }
}

TEST_F(DiskAnnMobileCollectionTest, ConcurrentQueryAndFetch) {
  constexpr size_t kThreadCount = 4;
  constexpr size_t kIterations = 20;

  auto schema = MakeSchema(MetricType::L2);
  auto create_result =
      Collection::CreateAndOpen(kCollectionPath, *schema, Options());
  ASSERT_TRUE(create_result.has_value()) << create_result.error().message();
  auto collection = std::move(create_result.value());
  auto docs = MakeDocs(0, kDocCount);
  ASSERT_TRUE(WriteSucceeded(collection->insert(docs), docs.size()));
  ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());

  std::array<std::vector<std::string>, kDocCount> query_baselines;
  for (uint64_t doc_id = 0; doc_id < kDocCount; ++doc_id) {
    auto query_result = collection->query(MakeFp32Query(doc_id, kFp32Field));
    ASSERT_TRUE(query_result.has_value()) << query_result.error().message();
    ASSERT_FALSE(query_result->empty());
    query_baselines[doc_id] = SortedPks(*query_result);
    ASSERT_EQ(query_baselines[doc_id].size(), query_result->size());
  }

  std::atomic<size_t> failure_count{0};
  std::mutex failure_mutex;
  std::vector<std::string> failures;
  auto record_failure = [&](const std::string &failure) {
    ++failure_count;
    std::lock_guard<std::mutex> lock(failure_mutex);
    failures.emplace_back(failure);
  };
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
    threads.emplace_back([&, thread_id]() {
      for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        uint64_t doc_id = (thread_id * kIterations + iteration) % kDocCount;
        auto query_result =
            collection->query(MakeFp32Query(doc_id, kFp32Field));
        auto fetch_result = collection->fetch({"pk_" + std::to_string(doc_id)});
        bool query_ok = query_result.has_value() &&
                        SortedPks(*query_result) == query_baselines[doc_id];
        bool fetch_ok =
            FetchContainsPk(fetch_result, "pk_" + std::to_string(doc_id));
        if (!query_ok || !fetch_ok) {
          record_failure(
              "shared collection: thread=" + std::to_string(thread_id) +
              ", iteration=" + std::to_string(iteration) +
              ", query_ok=" + std::to_string(query_ok) +
              ", fetch_ok=" + std::to_string(fetch_ok));
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failure_count.load(), 0u);
  EXPECT_TRUE(failures.empty()) << (failures.empty() ? "" : failures.front());

  ASSERT_TRUE(collection->flush().ok());
  collection.reset();
  failure_count.store(0);
  failures.clear();
  threads.clear();
  for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
    threads.emplace_back([&, thread_id]() {
      auto open_result = Collection::Open(kCollectionPath, Options(true));
      if (!open_result.has_value()) {
        record_failure("read-only open: thread=" + std::to_string(thread_id));
        return;
      }
      auto read_only_collection = std::move(open_result.value());
      for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        uint64_t doc_id = (thread_id * kIterations + iteration) % kDocCount;
        auto query_result =
            read_only_collection->query(MakeFp32Query(doc_id, kFp32Field));
        auto fetch_result =
            read_only_collection->fetch({"pk_" + std::to_string(doc_id)});
        bool query_ok = query_result.has_value() &&
                        SortedPks(*query_result) == query_baselines[doc_id];
        bool fetch_ok =
            FetchContainsPk(fetch_result, "pk_" + std::to_string(doc_id));
        if (!query_ok || !fetch_ok) {
          record_failure(
              "read-only collection: thread=" + std::to_string(thread_id) +
              ", iteration=" + std::to_string(iteration) +
              ", query_ok=" + std::to_string(query_ok) +
              ", fetch_ok=" + std::to_string(fetch_ok));
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failure_count.load(), 0u);
  EXPECT_TRUE(failures.empty()) << (failures.empty() ? "" : failures.front());
}

TEST_F(DiskAnnMobileCollectionTest, OperationFailuresDoNotPoisonCollection) {
  auto schema = MakeSchema(MetricType::L2);
  auto create_result =
      Collection::CreateAndOpen(kCollectionPath, *schema, Options());
  ASSERT_TRUE(create_result.has_value()) << create_result.error().message();
  auto collection = std::move(create_result.value());
  auto docs = MakeDocs(0, kDocCount);
  ASSERT_TRUE(WriteSucceeded(collection->insert(docs), docs.size()));
  ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());

  auto invalid_query = MakeFp32Query(12, kFp32Field);
  invalid_query.target_.set_vector("invalid-size");
  EXPECT_FALSE(collection->query(invalid_query).has_value());

  auto wrong_params_query = MakeFp32Query(12, kFp32Field);
  wrong_params_query.target_.query_params_ =
      std::make_shared<FlatQueryParams>();
  EXPECT_FALSE(collection->query(wrong_params_query).has_value());

  Doc invalid_doc;
  invalid_doc.set_pk("invalid_doc");
  invalid_doc.set<int32_t>("category", 0);
  invalid_doc.set<std::string>("name", "missing required vector");
  std::vector<Doc> invalid_docs{invalid_doc};
  auto invalid_write = collection->insert(invalid_docs);
  EXPECT_TRUE(!invalid_write.has_value() || invalid_write->empty() ||
              !invalid_write->front().ok());

  GroupByVectorQuery unsupported_group_query;
  unsupported_group_query.target_ = MakeFp32Query(12, kFp32Field, 8).target_;
  unsupported_group_query.group_by_field_name_ = "category";
  unsupported_group_query.group_count_ = 4;
  unsupported_group_query.topk_per_group_ = 2;
  EXPECT_FALSE(collection->group_by_query(unsupported_group_query).has_value());

  auto valid_result = collection->query(MakeFp32Query(12, kFp32Field));
  ASSERT_TRUE(valid_result.has_value()) << valid_result.error().message();
  ASSERT_FALSE(valid_result->empty());

  auto recovery_docs = MakeDocs(kDocCount, kDocCount + 1);
  ASSERT_TRUE(
      WriteSucceeded(collection->insert(recovery_docs), recovery_docs.size()));
  ASSERT_TRUE(collection->flush().ok());
  ASSERT_TRUE(collection->optimize(OptimizeOptions{2}).ok());
  collection.reset();

  auto reopen_result = Collection::Open(kCollectionPath, Options());
  ASSERT_TRUE(reopen_result.has_value()) << reopen_result.error().message();
  collection = std::move(reopen_result.value());
  auto recovered_result =
      collection->query(MakeFp32Query(kDocCount, kFp32Field));
  ASSERT_TRUE(recovered_result.has_value())
      << recovered_result.error().message();
  ASSERT_FALSE(recovered_result->empty());
  const std::string recovered_pk = "pk_" + std::to_string(kDocCount);
  auto recovered_fetch = collection->fetch({recovered_pk});
  EXPECT_TRUE(FetchContainsPk(recovered_fetch, recovered_pk));
  auto recovered_stats = collection->stats();
  ASSERT_TRUE(recovered_stats.has_value()) << recovered_stats.error().message();
  EXPECT_EQ(recovered_stats->doc_count, kDocCount + 1);
}

#else

TEST(DiskAnnMobileCollectionTest, PlatformDoesNotClaimMobileSupport) {
  GTEST_SKIP() << "DiskAnn is not enabled on this desktop platform";
}

#endif

}  // namespace
}  // namespace zvec
