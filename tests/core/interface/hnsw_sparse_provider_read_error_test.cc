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
#include "algorithm/hnsw_sparse/hnsw_sparse_builder_entity.h"
#include "algorithm/hnsw_sparse/hnsw_sparse_index_provider.h"
#include "utility/sparse_utility.h"

namespace zvec::core {
namespace {

class FailingSparseHnswEntity : public HnswSparseBuilderEntity {
 public:
  FailingSparseHnswEntity() {
    *mutable_doc_cnt() = 2;
  }
  int read_key(node_id_t id, key_t *key) const override {
    *key = kInvalidKey;
    return id == 0 ? 0 : IndexError_ReadData;
  }
};

TEST(HnswSparseProviderReadErrorTest, KeyErrorsSurviveNullSkipping) {
  IndexMeta meta;
  meta.set_meta_type(IndexMeta::MetaType::MT_SPARSE);
  meta.set_data_type(IndexMeta::DataType::DT_FP32);
  HnswSparseIndexProvider provider(
      meta, std::make_shared<FailingSparseHnswEntity>(), "test");
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
