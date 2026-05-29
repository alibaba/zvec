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
// limitations under the License

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/sqlengine.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"

namespace zvec::sqlengine {

// Multi-segment FTS recall regression:
//
// The planner's SegmentNode drains per-segment readers in LIFO order, so the
// per-segment BM25 ordering is *not* preserved across the merged stream. The
// planner must therefore add a global order_by on the score column for FTS,
// mirroring what it already does for vector queries.
//
// To make the regression observable we engineer the two segments so that
//   * segments_[0] (read LAST) holds the globally highest-scoring doc, and
//   * segments_[1] (read FIRST) holds many low-scoring docs.
//
// Per-segment BM25 stats (rare term -> high IDF in segments_[0], common term
// -> low IDF in segments_[1]) guarantee s0_0 outranks every doc in
// segments_[1]. Without the global sort the first doc in the merged stream is
// the much lower-scoring s1_*, which breaks both the descending invariant and
// topk truncation.

class FtsMultiSegmentTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    FileHelper::RemoveDirectory(root_path_);
    FileHelper::CreateDirectory(root_path_);

    build_schema();

    // segments_[0]: only one doc contains "apple" but with very high TF and
    // very low df (rare term) -> high BM25.
    auto seg0 = create_segment(root_path_ + "/seg0", "fts_ms_seg0");
    ASSERT_NE(seg0, nullptr);
    insert_docs(seg0, /*pk_prefix=*/"s0_",
                {
                    {"apple apple apple apple apple"},  // doc 0: TF=5, df=1
                    {"banana"},
                    {"cherry"},
                    {"date"},
                    {"elderberry"},
                });

    // segments_[1]: all docs contain "apple" (df=N) -> very low IDF -> low
    // BM25 across the board.
    auto seg1 = create_segment(root_path_ + "/seg1", "fts_ms_seg1");
    ASSERT_NE(seg1, nullptr);
    insert_docs(seg1, /*pk_prefix=*/"s1_",
                {
                    {"apple banana"},
                    {"apple cherry"},
                    {"apple date"},
                    {"apple elderberry"},
                });

    segments_.push_back(seg0);
    segments_.push_back(seg1);

    engine_ = SQLEngine::create(std::make_shared<Profiler>());
  }

  static void TearDownTestSuite() {
    segments_.clear();
    engine_.reset();
    schema_.reset();
    FileHelper::RemoveDirectory(root_path_);
  }

  Result<DocPtrList> fts_search(const std::string &query_string,
                                int topk = 10) {
    SearchQuery vq;
    vq.topk_ = topk;
    vq.target_.field_name_ = "content";
    FtsClause fts;
    fts.query_string_ = query_string;
    vq.target_.clause_ = fts;
    return engine_->execute(schema_, vq, segments_);
  }

 private:
  static void build_schema() {
    auto fts_params = std::make_shared<FtsIndexParams>(
        "whitespace", std::vector<std::string>{"lowercase"}, "");
    schema_ = std::make_shared<CollectionSchema>(
        "fts_multi_segment_test",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("content", DataType::STRING, false,
                                          fts_params),
            // Dummy vector field keeps the schema parity with the single-
            // segment FTS fixture so the analyzer/planner paths behave the
            // same.
            std::make_shared<FieldSchema>(
                "vec", DataType::VECTOR_FP32, 4, false,
                std::make_shared<FlatIndexParams>(MetricType::L2)),
        });
  }

  static Segment::Ptr create_segment(const std::string &seg_path,
                                     const std::string &name) {
    FileHelper::CreateDirectory(seg_path);

    auto segment_meta = std::make_shared<SegmentMeta>();
    segment_meta->set_id(0);

    auto id_map = IDMap::CreateAndOpen(name, seg_path + "/id_map", true, false);
    auto delete_store = std::make_shared<DeleteStore>(name);

    Version v1;
    v1.set_schema(*schema_);
    std::string v_path = seg_path + "/manifest";
    FileHelper::CreateDirectory(v_path);
    auto vm = VersionManager::Create(v_path, v1);
    if (!vm.has_value()) {
      return nullptr;
    }

    BlockMeta mem_block;
    mem_block.id_ = 0;
    mem_block.type_ = BlockType::SCALAR;
    mem_block.min_doc_id_ = 0;
    mem_block.max_doc_id_ = 0;
    mem_block.doc_count_ = 0;
    segment_meta->set_writing_forward_block(mem_block);

    SegmentOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = true;
    options.max_buffer_size_ = 256 * 1024;

    auto result = Segment::CreateAndOpen(seg_path, *schema_, 0, 0, id_map,
                                         delete_store, vm.value(), options);
    if (!result) {
      return nullptr;
    }
    return result.value();
  }

  struct Entry {
    std::string content;
  };

  static void insert_docs(const Segment::Ptr &segment,
                          const std::string &pk_prefix,
                          const std::vector<Entry> &entries) {
    for (size_t i = 0; i < entries.size(); ++i) {
      Doc doc;
      doc.set_pk(pk_prefix + std::to_string(i));
      doc.set_doc_id(i);
      doc.set<std::string>("content", entries[i].content);
      auto status = segment->Insert(doc);
      ASSERT_TRUE(status.ok())
          << pk_prefix << i << " insert failed: " << status.c_str();
    }
  }

 protected:
  static inline std::string root_path_ = "./fts_multi_segment_test_collection";
  static inline CollectionSchema::Ptr schema_;
  static inline std::vector<Segment::Ptr> segments_;
  static inline SQLEngine::Ptr engine_;
};

// The merged stream from all segments must be strictly non-increasing in
// score. Without the global order_by, segments_[1]'s low-scoring docs would
// appear before segments_[0]'s much higher-scoring s0_0, violating BM25 rank.
TEST_F(FtsMultiSegmentTest, ScoreDescendingAcrossSegments) {
  auto result = fts_search("apple");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();

  // s0_0 + s1_0..s1_3 = 5 matches.
  ASSERT_EQ(result->size(), 5u);

  // s0_0 (TF=5, rare term in seg0) dominates the 4 low-IDF s1_* docs.
  EXPECT_EQ((*result)[0]->pk(), "s0_0");
  EXPECT_GT((*result)[0]->score(), (*result)[1]->score());

  for (size_t i = 0; i + 1 < result->size(); ++i) {
    EXPECT_GE((*result)[i]->score(), (*result)[i + 1]->score())
        << "score not descending at rank " << i << ": " << (*result)[i]->pk()
        << "=" << (*result)[i]->score() << " vs " << (*result)[i + 1]->pk()
        << "=" << (*result)[i + 1]->score();
  }
}

// topk must cut against the globally-sorted stream. Without the fix the
// first batch surfaced from SegmentNode comes from segments_[1] (LIFO read),
// so topk=1 would silently drop the highest-scoring s0_0.
TEST_F(FtsMultiSegmentTest, TopkPicksGloballyHighestScore) {
  auto result = fts_search("apple", /*topk=*/1);
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "s0_0");
}

// Sanity: a cross-segment OR query still returns the union of matches and
// stays descending across the segment boundary.
TEST_F(FtsMultiSegmentTest, CrossSegmentUnionDescending) {
  // apple: 5 docs (s0_0, s1_0..s1_3). banana: s0_1 (seg0), s1_0 (seg1).
  // OR-union: {s0_0, s0_1, s1_0, s1_1, s1_2, s1_3} = 6 docs.
  auto result = fts_search("apple banana");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_EQ(result->size(), 6u);
  for (size_t i = 0; i + 1 < result->size(); ++i) {
    EXPECT_GE((*result)[i]->score(), (*result)[i + 1]->score())
        << "score not descending at rank " << i;
  }
}

}  // namespace zvec::sqlengine
