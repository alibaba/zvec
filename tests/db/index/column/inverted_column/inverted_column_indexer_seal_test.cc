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

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <rocksdb/metadata.h>
#include <rocksdb/table_properties.h>
#include "db/common/constants.h"
#include "db/index/column/inverted_column/inverted_codec.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "tests/test_util.h"

using namespace zvec;

namespace {

const std::string kDir{"./inverted_column_indexer_seal_dir/"};
const std::string kCrashDir{"./inverted_column_indexer_seal_crash_dir/"};
const std::string kFirstCrashDir{"./inverted_column_indexer_seal_first_dir/"};
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
    EXPECT_TRUE(it->status().ok()) << name << ": " << it->status().ToString();
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

// Leaves range entries, as an interrupted seal does, in the range family of
// the unsealed index in kDir, flushed into one table file, and flips one byte
// of that file's data blocks: the first data block, or one in the middle.
void LeaveCorruptedRanges(bool first_block) {
  RawFamilies raw(kDir, false);
  auto *cf = raw.get("seq" + INVERT_SUFFIX_RANGES);
  ASSERT_NE(cf, nullptr);
  // Each value is a bitmap of about 4 KB that claims document 999999, so the
  // file has many data blocks.
  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  roaring_bitmap_add(bitmap, 999999);
  for (uint32_t j = 0; j < 2000; ++j) {
    roaring_bitmap_add(bitmap, 1000000 + 7 * j);
  }
  std::string value;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(bitmap, &value).ok());
  roaring_bitmap_free(bitmap);
  for (int i = 0; i < 64; ++i) {
    std::string key(1, static_cast<char>(i));
    key.append(1, '\0');
    key.append(1, static_cast<char>(i));
    uint64_t begin_size = 1;
    key.append(reinterpret_cast<const char *>(&begin_size), sizeof(uint64_t));
    ASSERT_TRUE(raw.db->Put(rocksdb::WriteOptions(), cf, key, value).ok());
  }
  ASSERT_TRUE(raw.db->Flush(rocksdb::FlushOptions(), cf).ok());

  std::vector<rocksdb::LiveFileMetaData> files;
  raw.db->GetLiveFilesMetaData(&files);
  std::string path;
  for (const auto &f : files) {
    if (f.column_family_name == "seq" + INVERT_SUFFIX_RANGES) {
      ASSERT_TRUE(path.empty()) << "more than one range table file";
      path = f.db_path + f.name;
    }
  }
  ASSERT_FALSE(path.empty());
  rocksdb::TablePropertiesCollection props;
  ASSERT_TRUE(raw.db->GetPropertiesOfAllTables(cf, &props).ok());
  ASSERT_EQ(props.size(), 1u);
  const uint64_t data_size = props.begin()->second->data_size;
  ASSERT_GT(data_size, 16u * 4096) << "expected many data blocks";

  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path;
  file.seekg(first_block ? 100 : data_size / 2);
  char byte = 0;
  file.read(&byte, 1);
  byte ^= 0x5a;
  file.seekp(first_block ? 100 : data_size / 2);
  file.write(&byte, 1);
  ASSERT_TRUE(file.good());
}

// An unreadable range family must not stop the seal: the range index is
// rebuilt, and range and point queries stay exact.
void SealRebuildsCorruptedRanges(bool first_block) {
  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, true, Fields(), false);
  ASSERT_TRUE(indexer);
  Insert(indexer);
  ASSERT_TRUE(indexer->flush().ok());
  indexer.reset();
  LeaveCorruptedRanges(first_block);

  indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, false, Fields(), false);
  ASSERT_TRUE(indexer);
  ASSERT_TRUE(indexer->seal().ok());
  EXPECT_TRUE((*indexer)["seq"]->is_sealed());
  EXPECT_TRUE((*indexer)["cyc"]->is_sealed());
  VerifyQueries(indexer);
  auto lt = (*indexer)["seq"]->search(Value(kDocs), CompareOp::LT);
  ASSERT_TRUE(lt);
  EXPECT_FALSE(lt->contains(999999));
  indexer.reset();

  indexer = InvertedIndexer::CreateAndOpen("seal", kDir, false, Fields(), true);
  ASSERT_TRUE(indexer);
  EXPECT_TRUE((*indexer)["seq"]->is_sealed());
  VerifyQueries(indexer);
  indexer.reset();
  RawFamilies sealed(kDir);
  EXPECT_GT(sealed.count("seq" + INVERT_SUFFIX_RANGES), 0u);
}

// Seals the "seq" column of the index in dir, flushes only the column
// families named in flushed, and copies dir to kCrashDir as a crash right then
// would leave it: the indexes write without a write-ahead log, so the flushed
// files are all that survives.
void SealAndCrash(const std::string &dir,
                  const std::vector<std::string> &flushed) {
  RocksdbContext ctx;
  // No background compaction, so the copied files are the same on every run.
  ctx.create_opts_.disable_auto_compactions = true;
  ASSERT_TRUE(
      ctx.open(dir, false, std::make_shared<InvertedRocksdbValueMerger>())
          .ok());
  auto seq = InvertedColumnIndexer::CreateAndOpen("seal", Fields()[0], ctx);
  ASSERT_TRUE(seq);
  ASSERT_TRUE(seq->seal().ok());
  for (const auto &name : flushed) {
    auto *cf = name == rocksdb::kDefaultColumnFamilyName
                   ? ctx.db_->DefaultColumnFamily()
                   : ctx.get_cf(name);
    ASSERT_NE(cf, nullptr) << name;
    ASSERT_TRUE(ctx.db_->Flush(rocksdb::FlushOptions(), cf).ok());
  }
  zvec::test_util::RemoveTestPath(kCrashDir);
  std::error_code copied;
  std::filesystem::copy(dir, kCrashDir,
                        std::filesystem::copy_options::recursive, copied);
  seq.reset();
  ctx.close();
  ASSERT_FALSE(copied) << copied.message();
}

class InvertedSealTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(kDir);
    zvec::test_util::RemoveTestPath(kCrashDir);
    zvec::test_util::RemoveTestPath(kFirstCrashDir);
  }
  void TearDown() override {
    zvec::test_util::RemoveTestPath(kDir);
    zvec::test_util::RemoveTestPath(kCrashDir);
    zvec::test_util::RemoveTestPath(kFirstCrashDir);
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

TEST_F(InvertedSealTest, SealRebuildsRangesWithCorruptedFirstBlock) {
  SealRebuildsCorruptedRanges(true);
}

TEST_F(InvertedSealTest, SealRebuildsRangesWithCorruptedMiddleBlock) {
  SealRebuildsCorruptedRanges(false);
}

TEST_F(InvertedSealTest, ResealAfterACrashDoesNotUseOldRanges) {
  auto indexer =
      InvertedIndexer::CreateAndOpen("seal", kDir, true, Fields(), false);
  ASSERT_TRUE(indexer);
  Insert(indexer);
  ASSERT_TRUE(indexer->flush().ok());
  indexer.reset();

  // A first seal stops after its range entries reach disk, before the seal
  // marker does. The index stays unsealed and gets one more document.
  SealAndCrash(kDir, {"seq" + INVERT_SUFFIX_RANGES});
  std::filesystem::rename(kCrashDir, kFirstCrashDir);
  indexer = InvertedIndexer::CreateAndOpen("seal", kFirstCrashDir, false,
                                           Fields(), false);
  ASSERT_TRUE(indexer);
  ASSERT_FALSE((*indexer)["seq"]->is_sealed());
  ASSERT_TRUE((*indexer)["seq"]->insert(kDocs, Value(0)).ok());
  ASSERT_TRUE(indexer->flush().ok());
  indexer.reset();

  // The next seal stops after its seal marker reaches disk, before its range
  // entries do. The ranges of the first seal must not be used: they miss the
  // new document.
  SealAndCrash(kFirstCrashDir, {rocksdb::kDefaultColumnFamilyName});
  indexer =
      InvertedIndexer::CreateAndOpen("seal", kCrashDir, false, Fields(), true);
  ASSERT_TRUE(indexer);
  ASSERT_TRUE((*indexer)["seq"]->is_sealed());
  auto lt = (*indexer)["seq"]->search(Value(5), CompareOp::LT);
  ASSERT_TRUE(lt);
  EXPECT_EQ(lt->count(), 6u);
  EXPECT_TRUE(lt->contains(kDocs));
}
