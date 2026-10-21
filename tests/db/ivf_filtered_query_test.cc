// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/type.h>

namespace zvec {
namespace {

constexpr char kCollectionPath[] = "ivf_filtered_query_collection";
constexpr uint32_t kDocCount = 2000;
constexpr uint32_t kDimension = 8;
constexpr int kCentroidCount = 8;

class IVFFilteredQueryTest
    : public testing::TestWithParam<std::tuple<bool, QuantizeType>> {
 public:
  static void SetUpTestSuite() {
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = 512ULL * 1024 * 1024;
    config.query_thread_count = 2;
    config.optimize_thread_count = 1;
    ASSERT_TRUE(GlobalConfig::Instance().initialize(config).ok());
  }

 protected:
  void SetUp() override {
    ailego::FileHelper::RemoveDirectory(kCollectionPath);
  }

  void TearDown() override {
    collection_.reset();
    ailego::FileHelper::RemoveDirectory(kCollectionPath);
  }

  static std::vector<float> Vector(uint32_t id) {
    // Exactly representable in both FP32 and FP16, with no distance ties.
    return std::vector<float>(kDimension, static_cast<float>(id + 1) / 16);
  }

  void create() {
    CollectionSchema schema("ivf_filtered_query");
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "category", DataType::INT32, false,
                        std::make_shared<InvertIndexParams>()))
                    .ok());
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "ordinal", DataType::INT32, false))
                    .ok());
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "embedding", DataType::VECTOR_FP32, kDimension, false,
                        std::make_shared<IVFIndexParams>(
                            MetricType::L2, kCentroidCount, 3, false,
                            std::get<1>(GetParam()))))
                    .ok());
    auto result = Collection::CreateAndOpen(
        kCollectionPath, schema,
        CollectionOptions{false, std::get<0>(GetParam())});
    ASSERT_TRUE(result.has_value()) << result.error().message();
    collection_ = std::move(result.value());

    std::vector<Doc> docs;
    docs.reserve(128);
    for (uint32_t id = 0; id < kDocCount; ++id) {
      Doc doc;
      doc.set_pk("doc_" + std::to_string(id));
      ASSERT_TRUE(doc.set<int32_t>("category", id % 20));
      ASSERT_TRUE(doc.set<int32_t>("ordinal", id));
      ASSERT_TRUE(doc.set<std::vector<float>>("embedding", Vector(id)));
      docs.emplace_back(std::move(doc));
      if (docs.size() == 128 || id + 1 == kDocCount) {
        auto inserted = collection_->insert(docs);
        ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
        ASSERT_EQ(docs.size(), inserted->size());
        for (const auto &status : inserted.value()) {
          ASSERT_TRUE(status.ok()) << status.message();
        }
        docs.clear();
      }
    }
  }

  void expect_query(const std::string &filter,
                    const std::vector<uint32_t> &expected,
                    int nprobe = kCentroidCount) {
    SCOPED_TRACE(filter);
    SearchQuery query;
    query.filter_ = filter;
    query.topk_ = 5;
    query.output_fields_ = {"category", "ordinal"};
    query.include_vector_ = true;
    query.target_.field_name_ = "embedding";
    query.target_.query_params_ = std::make_shared<IVFQueryParams>(nprobe);
    const std::vector<float> vector(kDimension, 0.0F);
    query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));

    auto result = collection_->query(query);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(expected.size(), result->size());
    for (size_t i = 0; i < expected.size(); ++i) {
      const auto id = expected[i];
      const auto &doc = result->at(i);
      ASSERT_NE(nullptr, doc);
      EXPECT_EQ("doc_" + std::to_string(id), doc->pk());
      EXPECT_EQ(static_cast<int32_t>(id % 20), doc->get<int32_t>("category"));
      EXPECT_EQ(static_cast<int32_t>(id), doc->get<int32_t>("ordinal"));
      const auto stored = doc->get<std::vector<float>>("embedding");
      ASSERT_TRUE(stored.has_value());
      EXPECT_EQ(Vector(id), stored.value());
      const float value = static_cast<float>(id + 1) / 16;
      EXPECT_FLOAT_EQ(kDimension * value * value, doc->score());
    }
  }

  void check_filters() {
    // Below, exactly at, and above the default 10% brute-force cutoff.
    ASSERT_NO_FATAL_FAILURE(expect_query("category = 3", {3, 23, 43, 63, 83}));
    ASSERT_NO_FATAL_FAILURE(
        expect_query("category IN (3, 4)", {3, 4, 23, 24, 43}));
    ASSERT_NO_FATAL_FAILURE(
        expect_query("category IN (3, 4, 5)", {3, 4, 5, 23, 24}));
    ASSERT_NO_FATAL_FAILURE(expect_query("category = 99", {}));
    // Candidate-key BF must preserve the forward predicate and must not be
    // restricted to the nearest centroid, which contains none of these rows.
    ASSERT_NO_FATAL_FAILURE(expect_query("category = 3 AND ordinal >= 503",
                                         {503, 523, 543, 563, 583}, 1));
    ASSERT_NO_FATAL_FAILURE(expect_query("", {0, 1, 2, 3, 4}));
  }

  Collection::Ptr collection_;
};

TEST_P(IVFFilteredQueryTest, PreservesFiltersBeforeBuildAfterBuildAndReopen) {
  create();
  ASSERT_FALSE(HasFatalFailure());
  // The mutable index is Flat even though the schema declares IVF.
  check_filters();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_TRUE(collection_->optimize(OptimizeOptions{1}).ok());
  auto stats = collection_->stats();
  ASSERT_TRUE(stats.has_value()) << stats.error().message();
  ASSERT_EQ(kDocCount, stats->doc_count);
  ASSERT_FLOAT_EQ(1.0F, stats->index_completeness.at("embedding"));
  check_filters();
  ASSERT_FALSE(HasFatalFailure());

  collection_.reset();
  auto result = Collection::Open(
      kCollectionPath, CollectionOptions{true, std::get<0>(GetParam())});
  ASSERT_TRUE(result.has_value()) << result.error().message();
  collection_ = std::move(result.value());
  check_filters();
}

INSTANTIATE_TEST_SUITE_P(
    StorageAndQuantization, IVFFilteredQueryTest,
    testing::Combine(testing::Bool(), testing::Values(QuantizeType::UNDEFINED,
                                                      QuantizeType::FP16)));

}  // namespace
}  // namespace zvec
