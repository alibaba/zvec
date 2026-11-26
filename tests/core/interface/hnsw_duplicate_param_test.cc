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
#include <zvec/core/interface/index_param_builders.h>

namespace zvec::core_interface {

TEST(HnswDuplicateParam, DefaultsOffAndSurvivesSerialization) {
  HNSWIndexParam original;
  EXPECT_FALSE(original.deduplicate_vectors);
  auto enabled = HNSWIndexParamBuilder().with_deduplicate_vectors(true).build();
  ASSERT_TRUE(enabled->deduplicate_vectors);
  ASSERT_TRUE(original.deserialize_from_json(enabled->serialize_to_json()));
  EXPECT_TRUE(original.deduplicate_vectors);
  HNSWIndexParam copied(original);
  EXPECT_TRUE(copied.deduplicate_vectors);
}

}  // namespace zvec::core_interface
