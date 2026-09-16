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

#include <random>
#include <utility>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"

using namespace zvec;

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
  ASSERT_TRUE(reader->close().ok());
  reader.reset();
  FileHelper::RemoveDirectory(path);
}
