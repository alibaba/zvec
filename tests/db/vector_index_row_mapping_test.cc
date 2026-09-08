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

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>
#include "db/common/file_helper.h"

namespace zvec {
namespace {

struct RowMappingCase {
  const char *name;
  bool create_index;
  QuantizeType quantize;
  bool upsert;
  bool full_compact;
  bool flush_between_batches;
};

using RowMappingParam = std::tuple<bool, RowMappingCase>;

class VectorIndexRowMappingTest
    : public ::testing::TestWithParam<RowMappingParam> {
 protected:
  static constexpr int kDocCount = 28;
  static constexpr int kDimension = 32;

  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    FileHelper::RemoveDirectory(path_);
  }

  void TearDown() override {
    collection_.reset();
    FileHelper::RemoveDirectory(path_);
  }

  static std::string PK(int id) {
    return "pk_" + std::to_string(id);
  }

  static Doc MakeDoc(int id, int vector_position) {
    Doc doc;
    doc.set_pk(PK(id));
    doc.set<int32_t>("marker", vector_position);
    // Distinct unit vectors stay exactly representable after cosine
    // normalization and FP16 conversion; the matching document is unique.
    std::vector<float> vector(kDimension, 0.0f);
    vector[vector_position] = 1.0f;
    doc.set<std::vector<float>>("embedding", vector);
    return doc;
  }

  void CheckDoc(const Doc::Ptr &actual, const Doc &expected) {
    ASSERT_NE(actual, nullptr) << "Missing document " << expected.pk();
    ASSERT_EQ(actual->pk(), expected.pk());
    auto marker = actual->get<int32_t>("marker");
    ASSERT_TRUE(marker.has_value());
    ASSERT_EQ(marker.value(), expected.get<int32_t>("marker").value());
    auto vector = actual->get<std::vector<float>>("embedding");
    ASSERT_TRUE(vector.has_value());
    ASSERT_EQ(vector.value(),
              expected.get<std::vector<float>>("embedding").value())
        << "Vector belongs to a different document: " << expected.pk();
  }

  void CheckFetch(const char *phase) {
    SCOPED_TRACE(phase);
    auto stats = collection_->stats();
    ASSERT_TRUE(stats.has_value()) << stats.error().message();
    ASSERT_EQ(stats->doc_count, expected_.size());
    for (const auto &[pk, expected] : expected_) {
      SCOPED_TRACE("fetch " + pk);
      auto fetched = collection_->fetch({pk});
      ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
      ASSERT_EQ(fetched->count(pk), 1u);
      ASSERT_NO_FATAL_FAILURE(CheckDoc(fetched->at(pk), expected));
    }
    for (const auto &pk : deleted_) {
      SCOPED_TRACE("deleted " + pk);
      auto fetched = collection_->fetch({pk});
      ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
      ASSERT_EQ(fetched->count(pk), 1u);
      ASSERT_EQ(fetched->at(pk), nullptr);
    }
  }

  void CheckQuery() {
    for (const auto &[pk, expected] : expected_) {
      SCOPED_TRACE("query " + pk);
      const auto vector = expected.get<std::vector<float>>("embedding").value();
      SearchQuery query;
      // Ask for every physical row so retained tombstones must be filtered
      // from the results, in addition to preserving each live vector's owner.
      query.topk_ = kDimension;
      query.include_vector_ = true;
      query.target_.field_name_ = "embedding";
      query.target_.set_vector(
          std::string(reinterpret_cast<const char *>(vector.data()),
                      vector.size() * sizeof(float)));
      // Exercise the built HNSW (including its quantized variant), but use
      // exhaustive search so this checks row mapping rather than ANN recall.
      query.target_.query_params_ =
          std::make_shared<HnswQueryParams>(128, 0.0f, true);
      auto result = collection_->query(query);
      ASSERT_TRUE(result.has_value()) << result.error().message();
      ASSERT_EQ(result->size(), expected_.size());
      ASSERT_NO_FATAL_FAILURE(CheckDoc(result->front(), expected));
      std::set<std::string> returned_pks;
      for (const auto &doc : result.value()) {
        ASSERT_NE(doc, nullptr);
        ASSERT_TRUE(returned_pks.insert(doc->pk()).second);
        auto it = expected_.find(doc->pk());
        ASSERT_NE(it, expected_.end()) << "Deleted document " << doc->pk();
        ASSERT_NO_FATAL_FAILURE(CheckDoc(doc, it->second));
      }
    }
  }

  const std::string path_ = "test_vector_index_row_mapping";
  Collection::Ptr collection_;
  std::map<std::string, Doc> expected_;
  std::vector<std::string> deleted_;
};

TEST_P(VectorIndexRowMappingTest, PreservesDocumentVectorAssociation) {
  const bool enable_mmap = std::get<0>(GetParam());
  const auto &test_case = std::get<1>(GetParam());
  auto target_params = std::make_shared<HnswIndexParams>(
      MetricType::COSINE, 16, 200, test_case.quantize);
  IndexParams::Ptr initial_params = target_params;
  if (test_case.create_index) {
    // Changing IP to cosine in the quantized case forces reconstruction of
    // both the raw FLAT block and the quantized search block.
    initial_params = std::make_shared<FlatIndexParams>(
        test_case.quantize == QuantizeType::FP16 ? MetricType::IP
                                                 : MetricType::COSINE);
  }
  CollectionSchema schema("row_mapping");
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "marker", DataType::INT32, false))
                  .ok());
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "embedding", DataType::VECTOR_FP32, uint32_t{kDimension},
                      false, initial_params))
                  .ok());
  CollectionOptions options{false, enable_mmap};
  auto created = Collection::CreateAndOpen(path_, schema, options);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  collection_ = std::move(created.value());

  for (int begin : {0, kDocCount / 2}) {
    std::vector<Doc> docs;
    for (int id = begin; id < begin + kDocCount / 2; ++id) {
      auto doc = MakeDoc(id, id);
      expected_.emplace(doc.pk(), doc);
      docs.push_back(std::move(doc));
    }
    auto inserted = collection_->insert(docs);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    for (const auto &status : inserted.value()) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
    if (begin == 0 && test_case.flush_between_batches) {
      // Flush creates another block within this segment; do not optimize
      // here, since that would bypass the first-build regression.
      auto status = collection_->flush();
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }

  std::vector<int> changed_ids{0, 4, 9};
  if (test_case.full_compact) {
    // 10 / 28 exceeds the rebuild threshold. This control must also pass
    // before the index-only fix, since all data is compacted together.
    changed_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  }
  if (test_case.upsert) {
    std::vector<Doc> replacements;
    for (size_t i = 0; i < changed_ids.size(); ++i) {
      auto doc = MakeDoc(changed_ids[i], kDocCount + static_cast<int>(i));
      expected_.at(doc.pk()) = doc;
      replacements.push_back(std::move(doc));
    }
    auto written = collection_->upsert(replacements);
    ASSERT_TRUE(written.has_value()) << written.error().message();
    for (const auto &status : written.value()) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  } else {
    for (int id : changed_ids) {
      deleted_.push_back(PK(id));
      expected_.erase(PK(id));
    }
    auto removed = collection_->delete_(deleted_);
    ASSERT_TRUE(removed.has_value()) << removed.error().message();
    for (const auto &status : removed.value()) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }

  ASSERT_NO_FATAL_FAILURE(CheckFetch("before index build"));
  auto built = test_case.create_index
                   ? collection_->create_index("embedding", target_params,
                                               CreateIndexOptions{1})
                   : collection_->optimize(OptimizeOptions{1});
  ASSERT_TRUE(built.ok()) << built.message();
  ASSERT_NO_FATAL_FAILURE(CheckFetch("after index build"));
  ASSERT_NO_FATAL_FAILURE(CheckQuery());

  for (bool read_only : {false, true}) {
    SCOPED_TRACE(read_only ? "reopen read-only" : "reopen writable");
    collection_.reset();
    auto reopened =
        Collection::Open(path_, CollectionOptions{read_only, enable_mmap});
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    collection_ = std::move(reopened.value());
    ASSERT_NO_FATAL_FAILURE(CheckFetch("after reopen"));
    ASSERT_NO_FATAL_FAILURE(CheckQuery());
  }
}

INSTANTIATE_TEST_SUITE_P(
    IndexOnlyAndCompaction, VectorIndexRowMappingTest,
    ::testing::Combine(
        ::testing::Bool(),
        ::testing::Values(
            RowMappingCase{"OptimizeDelete", false, QuantizeType::UNDEFINED,
                           false, false, false},
            RowMappingCase{"OptimizeUpsert", false, QuantizeType::UNDEFINED,
                           true, false, false},
            RowMappingCase{"CreateIndex", true, QuantizeType::UNDEFINED, false,
                           false, false},
            RowMappingCase{"QuantizedOptimize", false, QuantizeType::FP16,
                           false, false, true},
            RowMappingCase{"QuantizedCreateIndex", true, QuantizeType::FP16,
                           false, false, false},
            RowMappingCase{"FullCompaction", false, QuantizeType::UNDEFINED,
                           false, true, false})),
    [](const ::testing::TestParamInfo<RowMappingParam> &info) {
      return std::string(std::get<0>(info.param) ? "Mmap" : "Buffered") +
             std::get<1>(info.param).name;
    });

}  // namespace
}  // namespace zvec
