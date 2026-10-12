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

// Errors while an inverted index seals must reach the caller, and a column
// must never be marked sealed with an incomplete range index.

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/constants.h"
#include "db/common/rocksdb_context.h"
#include "db/index/column/inverted_column/inverted_codec.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "tests/test_util.h"

using namespace zvec;

namespace {

const std::string kDir{"./inverted_indexer_seal_error_dir/"};

std::string Value(int32_t v) {
  return std::string(reinterpret_cast<const char *>(&v), sizeof(v));
}

std::vector<FieldSchema> Fields() {
  return {FieldSchema{"seq", DataType::INT32, false,
                      std::make_shared<InvertIndexParams>(true)}};
}

// Creates the index in kDir with docs 0..n-1 (seq = doc id, every other
// field = doc id % 7), flushes it as a segment flush does, and closes it.
void Build(uint32_t n, const std::vector<FieldSchema> &fields = Fields()) {
  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, true, fields, false);
  ASSERT_TRUE(indexer);
  for (const auto &field : fields) {
    auto column = (*indexer)[field.name()];
    ASSERT_TRUE(column);
    for (uint32_t i = 0; i < n; ++i) {
      const int32_t v = static_cast<int32_t>(field.name() == "seq" ? i : i % 7);
      ASSERT_TRUE(column->insert(i, Value(v)).ok());
    }
  }
  ASSERT_TRUE(indexer->flush().ok());
}

// Opens the index in kDir with the families InvertedIndexer opens.
Status OpenContext(RocksdbContext &ctx) {
  return ctx.open(
      kDir,
      {"seq" + INVERT_SUFFIX_TERMS, "seq" + INVERT_SUFFIX_RANGES, INVERT_CDF},
      false, std::make_shared<InvertedRocksdbValueMerger>());
}

class InvertedIndexerSealErrorTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(kDir);
  }
  void TearDown() override {
    zvec::test_util::RemoveTestPath(kDir);
  }
};

}  // namespace

TEST_F(InvertedIndexerSealErrorTest, SealReturnsColumnSealError) {
  // Columns that can be sealed besides "seq": after a failed seal the caller
  // keeps writing to this index, so none of them may be left read-only.
  auto fields = Fields();
  for (const char *name : {"a", "b", "c"}) {
    fields.emplace_back(name, DataType::INT32, false,
                        std::make_shared<InvertIndexParams>(true));
  }
  Build(5000, fields);
  {
    // A term value whose header byte is above the id-list threshold cannot
    // be decoded, so the seal of "seq" fails.
    RocksdbContext ctx;
    ASSERT_TRUE(OpenContext(ctx).ok());
    auto *terms = ctx.get_cf("seq" + INVERT_SUFFIX_TERMS);
    ASSERT_TRUE(ctx.db_
                    ->Put(ctx.write_opts_, terms,
                          InvertedIndexCodec::Encode(Value(0), DataType::INT32),
                          std::string(1, '\xff'))
                    .ok());
    ASSERT_TRUE(ctx.close().ok());
  }

  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, false, fields, false);
  ASSERT_TRUE(indexer);
  EXPECT_FALSE(indexer->seal().ok());
  for (const auto &field : fields) {
    auto column = (*indexer)[field.name()];
    EXPECT_FALSE(column->is_sealed()) << field.name();
    EXPECT_TRUE(column->insert(5000, Value(1)).ok()) << field.name();
  }
}

// The indexes write without a WAL, so data is durable once its family is
// flushed. When seal() writes the marker, everything the marker describes
// must already be flushed.
TEST_F(InvertedIndexerSealErrorTest, SealFlushesItsDataBeforeTheMarker) {
  Build(5000);
  RocksdbContext ctx;
  ASSERT_TRUE(OpenContext(ctx).ok());
  auto seq = InvertedColumnIndexer::CreateAndOpen("seal", Fields()[0], ctx);
  ASSERT_TRUE(seq);
  ASSERT_TRUE(seq->seal().ok());

  auto unflushed = [&](const std::string &family) {
    uint64_t active = 0, immutable = 0;
    auto *cf = ctx.get_cf(family);
    EXPECT_TRUE(ctx.db_->GetIntProperty(
        cf, "rocksdb.num-entries-active-mem-table", &active));
    EXPECT_TRUE(ctx.db_->GetIntProperty(
        cf, "rocksdb.num-entries-imm-mem-tables", &immutable));
    return active + immutable;
  };
  EXPECT_EQ(unflushed("seq" + INVERT_SUFFIX_TERMS), 0u);
  EXPECT_EQ(unflushed("seq" + INVERT_SUFFIX_RANGES), 0u);
  EXPECT_EQ(unflushed(INVERT_CDF), 0u);
  // Only the marker itself is still in memory.
  EXPECT_EQ(unflushed(rocksdb::kDefaultColumnFamilyName), 1u);

  seq.reset();
  ASSERT_TRUE(ctx.close().ok());
}
