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
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {
namespace {

TEST(QuantizerParam, SerializesCanonicalUniformNames) {
  QuantizerParam uint7(QuantizerType::kUniformUint7);
  QuantizerParam uint8(QuantizerType::kUniformUint8);
  EXPECT_NE(std::string::npos, uint7.serialize_to_json().find("kUniformUint7"));
  EXPECT_NE(std::string::npos, uint8.serialize_to_json().find("kUniformUint8"));
}

}  // namespace
}  // namespace zvec::core_interface

TEST(RabitqQuantizerParam, CloneAndJsonPreserveRabitqConfiguration) {
  using namespace zvec::core_interface;
  RabitqQuantizerParam param(4);
  param.num_clusters = 8;
  param.sample_count = 200;
  param.niters = 6;
  auto copy = std::dynamic_pointer_cast<RabitqQuantizerParam>(param.clone());
  ASSERT_NE(nullptr, copy);
  EXPECT_EQ(4, copy->total_bits);
  EXPECT_EQ(8, copy->num_clusters);
  EXPECT_EQ(200, copy->sample_count);
  EXPECT_EQ(6, copy->niters);
  auto restored = QuantizerParam::Create(QuantizerType::kRabitq);
  ASSERT_TRUE(restored->deserialize_from_json(param.serialize_to_json()));
  auto rabitq = std::dynamic_pointer_cast<RabitqQuantizerParam>(restored);
  ASSERT_NE(nullptr, rabitq);
  EXPECT_EQ(4, rabitq->total_bits);
  EXPECT_EQ(8, rabitq->num_clusters);
  EXPECT_EQ(200, rabitq->sample_count);
  EXPECT_EQ(6, rabitq->niters);
  EXPECT_TRUE(restored->deserialize_from_json(
      RabitqQuantizerParam(9).serialize_to_json()));
  EXPECT_FALSE(restored->deserialize_from_json(
      RabitqQuantizerParam(10).serialize_to_json()));
  param.num_clusters = 0;
  EXPECT_FALSE(restored->deserialize_from_json(param.serialize_to_json()));
}
