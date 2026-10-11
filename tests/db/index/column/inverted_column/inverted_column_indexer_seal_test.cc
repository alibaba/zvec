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

// Sealing builds the range index of each range-optimized field. These tests
// check that the seal empties the range column family in place (its id stays
// the same, so it was not dropped and recreated), and that range and point
// queries stay exact both when the range index is kept and when it is
// emptied because it would not help.

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include "db/common/constants.h"
#include "db/index/column/inverted_column/inverted_codec.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "tests/test_util.h"

using namespace zvec;

namespace {

const std::string kDir{"./inverted_column_indexer_seal_dir/"};
constexpr uint32_t kDocs = 5000;
// "seq" has a distinct value per document: the range index is kept.
// "cyc" has 7 values: it would not help, so the seal empties it.
constexpr int32_t kCycle = 7;

std::vector<FieldSchema> Fields() {
  auto params = std::make_shared<InvertIndexParams>(true);
  return {FieldSchema{"seq", DataType::INT32, false, params},
          FieldSchema{"cyc", DataType::INT32, false, params}};
}

std::string Value(int32_t v) {
  return std::string(reinterpret_cast<const char *>(&v), sizeof(v));
}

struct RawFamilies {
  rocksdb::DB *db{nullptr};
  std::vector<rocksdb::ColumnFamilyHandle *> handles;

  explicit RawFamilies(const std::string &dir, bool read_only = true) {
    std::vector<std::string> names;
    EXPECT_TRUE(
        rocksdb::DB::ListColumnFamilies(rocksdb::DBOptions(), dir, &names)
            .ok());
    std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
    for (const auto &name : names) {
      descriptors.emplace_back(name, rocksdb::ColumnFamilyOptions());
    }
    auto s = read_only
                 ? rocksdb::DB::OpenForReadOnly(rocksdb::DBOptions(), dir,
                                                descriptors, &handles, &db)
                 : rocksdb::DB::Open(rocksdb::DBOptions(), dir, descriptors,
                                     &handles, &db);
    EXPECT_TRUE(s.ok()) << s.ToString();
  }
  ~RawFamilies() {
    for (auto *h : handles) db->DestroyColumnFamilyHandle(h);
    delete db;
  }
  rocksdb::ColumnFamilyHandle *get(const std::string &name) const {
    for (auto *h : handles) {
      if (h->GetName() == name) return h;
    }
    return nullptr;
  }
  uint32_t id(const std::string &name) const {
    auto *h = get(name);
    EXPECT_NE(h, nullptr) << name;
    return h ? h->GetID() : 0;
  }
  size_t count(const std::string &name) const {
    std::unique_ptr<rocksdb::Iterator> it(
        db->NewIterator(rocksdb::ReadOptions(), get(name)));
    size_t n = 0;
    for (it->SeekToFirst(); it->Valid(); it->Next()) ++n;
    return n;
  }
};

void Insert(const InvertedIndexer::Ptr &indexer) {
  auto seq = (*indexer)["seq"];
  auto cyc = (*indexer)["cyc"];
  ASSERT_TRUE(seq && cyc);
  for (uint32_t i = 0; i < kDocs; ++i) {
    ASSERT_TRUE(seq->insert(i, Value(static_cast<int32_t>(i))).ok());
    ASSERT_TRUE(cyc->insert(i, Value(static_cast<int32_t>(i % kCycle))).ok());
  }
}

void VerifyQueries(const InvertedIndexer::Ptr &indexer) {
  auto seq = (*indexer)["seq"];
  auto cyc = (*indexer)["cyc"];
  ASSERT_TRUE(seq && cyc);
  for (int32_t v : {0, 1, 777, 2500, 4999}) {
    auto lt = seq->search(Value(v), CompareOp::LT);
    ASSERT_TRUE(lt);
    EXPECT_EQ(lt->count(), static_cast<size_t>(v)) << "seq < " << v;
    auto ge = seq->search(Value(v), CompareOp::GE);
    ASSERT_TRUE(ge);
    EXPECT_EQ(ge->count(), kDocs - v) << "seq >= " << v;
    auto eq = seq->search(Value(v), CompareOp::EQ);
    ASSERT_TRUE(eq);
    EXPECT_EQ(eq->count(), 1u);
    EXPECT_TRUE(eq->contains(v));
  }
  for (int32_t v = 0; v < kCycle; ++v) {
    size_t eq_expected = 0, lt_expected = 0;
    for (uint32_t i = 0; i < kDocs; ++i) {
      eq_expected += static_cast<int32_t>(i % kCycle) == v;
      lt_expected += static_cast<int32_t>(i % kCycle) < v;
    }
    auto eq = cyc->search(Value(v), CompareOp::EQ);
    ASSERT_TRUE(eq);
    EXPECT_EQ(eq->count(), eq_expected) << "cyc == " << v;
    auto lt = cyc->search(Value(v), CompareOp::LT);
    ASSERT_TRUE(lt);
    EXPECT_EQ(lt->count(), lt_expected) << "cyc < " << v;
  }
}

class InvertedSealTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(kDir);
  }
  void TearDown() override {
    zvec::test_util::RemoveTestPath(kDir);
  }
};

}  // namespace

TEST_F(InvertedSealTest, SealKeepsRangeFamilyIds) {
  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, true, Fields(), false);
  ASSERT_TRUE(indexer);
  const std::string seq_ranges = "seq" + INVERT_SUFFIX_RANGES;
  const std::string cyc_ranges = "cyc" + INVERT_SUFFIX_RANGES;
  uint32_t seq_id = 0, cyc_id = 0;
  {
    RawFamilies before(kDir);
    seq_id = before.id(seq_ranges);
    cyc_id = before.id(cyc_ranges);
  }
  Insert(indexer);
  ASSERT_TRUE(indexer->seal().ok());
  VerifyQueries(indexer);
  {
    RawFamilies after(kDir);
    EXPECT_EQ(after.id(seq_ranges), seq_id) << "seq range family recreated";
    EXPECT_EQ(after.id(cyc_ranges), cyc_id) << "cyc range family recreated";
  }
  indexer.reset();

  // The sealed index reopens with the same answers. The kept range index has
  // entries; the emptied one has none after the seal's compaction.
  indexer = InvertedIndexer::CreateAndOpen("seal", kDir, false, Fields(), true);
  ASSERT_TRUE(indexer);
  VerifyQueries(indexer);
  indexer.reset();
  RawFamilies sealed(kDir);
  EXPECT_GT(sealed.count(seq_ranges), 0u);
  EXPECT_EQ(sealed.count(cyc_ranges), 0u);
}

TEST_F(InvertedSealTest, SealClearsRangesLeftByAnInterruptedSeal) {
  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, true, Fields(), false);
  ASSERT_TRUE(indexer);
  Insert(indexer);
  ASSERT_TRUE(indexer->flush().ok());  // as a segment flush does
  indexer.reset();

  // A seal that stopped after its range writes reached disk leaves range
  // entries in an unsealed index. Plant one that claims document 999999 for
  // every term, so a range query would return it if the seal kept it.
  {
    RawFamilies raw(kDir, false);
    std::string key("\x00", 1);
    key.append(1, '\0');
    key.append("\x00", 1);
    uint64_t begin_size = 1;
    key.append(reinterpret_cast<const char *>(&begin_size), sizeof(uint64_t));
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    roaring_bitmap_add(bitmap, 999999);
    std::string value;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(bitmap, &value).ok());
    roaring_bitmap_free(bitmap);
    ASSERT_TRUE(raw.db
                    ->Put(rocksdb::WriteOptions(),
                          raw.get("seq" + INVERT_SUFFIX_RANGES), key, value)
                    .ok());
    ASSERT_TRUE(raw.db->Flush(rocksdb::FlushOptions(), raw.handles).ok());
  }

  indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, false, Fields(), false);
  ASSERT_TRUE(indexer);
  ASSERT_TRUE(indexer->seal().ok());
  VerifyQueries(indexer);
  auto lt = (*indexer)["seq"]->search(Value(kDocs), CompareOp::LT);
  ASSERT_TRUE(lt);
  EXPECT_FALSE(lt->contains(999999));
}
