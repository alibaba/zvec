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
#include <string>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core {
namespace {

TEST(UniformUint7Metric, UsesCanonicalParamsAndComputesDistance) {
  auto metric = IndexFactory::CreateMetric("UniformUint7");
  ASSERT_TRUE(metric);

  ailego::Params params;
  params.set("proxima.uniform_uint7.metric.origin_metric_name",
             std::string("SquaredEuclidean"));
  constexpr size_t kDimension = 4;
  ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DataType::DT_INT8, kDimension),
                            params));

  std::string metric_name;
  EXPECT_TRUE(metric->params().get(
      "proxima.uniform_uint7.metric.origin_metric_name", &metric_name));
  EXPECT_EQ("SquaredEuclidean", metric_name);

  const int8_t lhs[kDimension] = {0, 1, 2, 127};
  const int8_t rhs[kDimension] = {0, 2, 4, 120};
  float distance = -1.0f;
  metric->distance()(lhs, rhs, kDimension, &distance);
  EXPECT_FLOAT_EQ(54.0f, distance);

  const IndexMeta matching_meta(IndexMeta::DataType::DT_INT8, kDimension);
  const IndexMeta wrong_dimension_meta(IndexMeta::DataType::DT_INT8,
                                       kDimension + 1);
  EXPECT_TRUE(metric->is_matched(matching_meta));
  EXPECT_FALSE(metric->is_matched(wrong_dimension_meta));
  EXPECT_TRUE(metric->is_matched(
      matching_meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, kDimension)));
  EXPECT_FALSE(metric->is_matched(
      matching_meta,
      IndexQueryMeta(IndexMeta::DataType::DT_INT8, kDimension + 1)));
  EXPECT_FALSE(metric->is_matched(
      wrong_dimension_meta,
      IndexQueryMeta(IndexMeta::DataType::DT_INT8, kDimension + 1)));
}

TEST(UniformUint7Metric, RejectsInvalidDimension) {
  ailego::Params params;
  params.set("proxima.uniform_uint7.metric.origin_metric_name",
             std::string("SquaredEuclidean"));
  for (const uint32_t dimension :
       {0U, static_cast<uint32_t>(MAX_DIMENSION + 1)}) {
    auto metric = IndexFactory::CreateMetric("UniformUint7");
    ASSERT_TRUE(metric);
    EXPECT_EQ(IndexError_InvalidArgument,
              metric->init(IndexMeta(IndexMeta::DataType::DT_INT8, dimension),
                           params));
  }
}

}  // namespace
}  // namespace zvec::core
