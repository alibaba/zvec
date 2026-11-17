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
#include "algorithm/flat_sparse/flat_sparse_provider.h"
#include "algorithm/flat_sparse/flat_sparse_searcher_entity.h"

namespace zvec::core {
namespace {

class FailingSparseFlatEntity : public FlatSparseSearcherEntity {
 public:
  uint32_t doc_cnt() const override {
    return 2;
  }
  int read_key(node_id_t id, uint64_t *key) const override {
    *key = kInvalidKey;
    return id == 0 ? 0 : IndexError_ReadData;
  }
};

TEST(FlatSparseProviderReadErrorTest, KeyErrorsSurviveNullSkipping) {
  IndexMeta meta;
  meta.set_meta_type(IndexMeta::MetaType::MT_SPARSE);
  meta.set_data_type(IndexMeta::DataType::DT_FP32);
  FlatSparseIndexProvider<FailingSparseFlatEntity> provider(
      std::make_shared<FailingSparseFlatEntity>(), meta, "test");
  for (bool include_empty : {false, true}) {
    auto iter = include_empty ? provider.create_slot_iterator()
                              : provider.create_iterator();
    ASSERT_NE(iter, nullptr);
    if (include_empty) {
      ASSERT_TRUE(iter->is_valid());
      EXPECT_EQ(UINT64_MAX, iter->key());
      EXPECT_EQ(0, iter->status());
      iter->next();
    }
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(IndexError_ReadData, iter->status());
  }
}

}  // namespace
}  // namespace zvec::core
