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

#include <gtest/gtest.h>
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param.h"

namespace zvec::core_interface {

TEST(OmegaQueryParamTest, ClonePreservesOmegaFields) {
  OmegaQueryParam param;
  param.topk = 20;
  param.fetch_vector = true;
  param.radius = 1.25f;
  param.is_linear = true;
  param.ef_search = 432;
  param.training_query_id = 7;
  param.target_recall = 0.91f;

  auto cloned_base = param.Clone();
  auto cloned = std::dynamic_pointer_cast<OmegaQueryParam>(cloned_base);
  ASSERT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->topk, 20U);
  EXPECT_TRUE(cloned->fetch_vector);
  EXPECT_FLOAT_EQ(cloned->radius, 1.25f);
  EXPECT_TRUE(cloned->is_linear);
  EXPECT_EQ(cloned->ef_search, 432U);
  EXPECT_EQ(cloned->training_query_id, 7);
  EXPECT_FLOAT_EQ(cloned->target_recall, 0.91f);
}

TEST(OmegaQueryParamTest, JsonRoundTripPreservesTargetRecall) {
  OmegaQueryParam param;
  param.topk = 15;
  param.fetch_vector = true;
  param.ef_search = 512;
  param.target_recall = 0.92f;

  const std::string json =
      IndexFactory::QueryParamSerializeToJson(param, false);
  auto restored =
      IndexFactory::QueryParamDeserializeFromJson<OmegaQueryParam>(json);

  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->topk, 15U);
  EXPECT_TRUE(restored->fetch_vector);
  EXPECT_EQ(restored->ef_search, 512U);
  EXPECT_FLOAT_EQ(restored->target_recall, 0.92f);
}

TEST(OmegaQueryParamTest, BaseDeserializerReturnsOmegaType) {
  OmegaQueryParam param;
  param.ef_search = 256;
  param.target_recall = 0.9f;

  const std::string json =
      IndexFactory::QueryParamSerializeToJson(param, false);
  auto restored =
      IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(json);

  ASSERT_NE(restored, nullptr);
  auto* omega = dynamic_cast<OmegaQueryParam*>(restored.get());
  ASSERT_NE(omega, nullptr);
  EXPECT_EQ(omega->ef_search, 256U);
  EXPECT_FLOAT_EQ(omega->target_recall, 0.9f);
}

}  // namespace zvec::core_interface
