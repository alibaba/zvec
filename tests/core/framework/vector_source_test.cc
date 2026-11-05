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

#include <array>
#include <gtest/gtest.h>
#include <zvec/core/framework/vector_source.h>
#include <zvec/core/interface/vector_source.h>

namespace zvec::core {
namespace {

// This target links only core_framework and zvec_ailego. Construction, batch
// reads and cursors must not pull the interface or any algorithm library in.
class FrameworkVectorSource final : public ContiguousVectorSource {
 public:
  ContiguousBatch get_contiguous_view() const override {
    return {rows.data(), 0, 2, sizeof(rows[0]), nullptr};
  }

  std::array<std::array<float, 3>, 2> rows{{{1, 2, 3}, {4, 5, 6}}};
};

TEST(VectorSourceTest, FrameworkOwnsSourceAndCursorImplementations) {
  FrameworkVectorSource source;
  const uint32_t ids[]{1, 2, 0};
  const void *vectors[3]{};
  source.get_vectors(ids, 3, vectors);
  EXPECT_EQ(source.rows[1].data(), vectors[0]);
  EXPECT_EQ(nullptr, vectors[1]);
  EXPECT_EQ(source.rows[0].data(), vectors[2]);
  EXPECT_EQ(source.rows[0].data(), source.get_vector(0));
  EXPECT_EQ(nullptr, source.get_vector(2));

  // Exercise the base defaults as well as ContiguousVectorSource overrides.
  source.VectorSource::get_vectors(ids, 3, vectors);
  EXPECT_EQ(source.rows[1].data(), vectors[0]);
  EXPECT_EQ(nullptr, source.VectorSource::create_iterator());

  auto cursor = source.create_iterator();
  ASSERT_NE(nullptr, cursor);
  VectorBatch batch;
  ASSERT_EQ(0, cursor->next_batch(2, &batch));
  const auto &view = std::get<ContiguousBatch>(batch);
  EXPECT_EQ(source.rows.data(), view.data);
  EXPECT_EQ(2U, view.count);
  ASSERT_EQ(0, cursor->next_batch(2, &batch));
  EXPECT_EQ(0U, std::get<ContiguousBatch>(batch).count);
}

}  // namespace
}  // namespace zvec::core
