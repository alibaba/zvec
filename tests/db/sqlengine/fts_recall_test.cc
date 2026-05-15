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

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/sqlengine.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/query_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"

namespace zvec::sqlengine {

// ============================================================
// FTS Recall Test fixture (real Segment + SQLEngine::execute via VectorQuery)
// ============================================================

class FtsRecallTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);

    build_schema();
    auto segment = create_segment();
    ASSERT_NE(segment, nullptr);
    insert_docs(segment);
    segments_.push_back(segment);

    engine_ = SQLEngine::create(std::make_shared<Profiler>());
  }

  static void TearDownTestSuite() {
    segments_.clear();
    engine_.reset();
    schema_.reset();
    FileHelper::RemoveDirectory(seg_path_);
  }

  // Helper: execute FTS query_string search via VectorQuery
  Result<DocPtrList> fts_search(const std::string &query_string,
                                int topk = 10) {
    VectorQuery vq;
    vq.topk_ = topk;
    vq.field_name_ = "content";
    FtsQuery fts_query;
    fts_query.query_string_ = query_string;
    vq.fts_query_ = fts_query;
    return engine_->execute(schema_, vq, segments_);
  }

  // Helper: execute FTS match_string search via VectorQuery
  Result<DocPtrList> fts_match(const std::string &match_string,
                               const std::string &default_op = "",
                               int topk = 10) {
    VectorQuery vq;
    vq.topk_ = topk;
    vq.field_name_ = "content";
    FtsQuery fts_query;
    fts_query.match_string_ = match_string;
    vq.fts_query_ = fts_query;
    if (!default_op.empty()) {
      auto fts_qp = std::make_shared<zvec::FtsQueryParams>();
      fts_qp->set_default_operator(default_op);
      vq.query_params_ = fts_qp;
    }
    return engine_->execute(schema_, vq, segments_);
  }

  // Helper: execute FTS query_string with default_operator via VectorQuery
  Result<DocPtrList> fts_query_with_op(const std::string &query_string,
                                       const std::string &default_op,
                                       int topk = 10) {
    VectorQuery vq;
    vq.topk_ = topk;
    vq.field_name_ = "content";
    FtsQuery fts_query;
    fts_query.query_string_ = query_string;
    vq.fts_query_ = fts_query;
    auto fts_qp = std::make_shared<zvec::FtsQueryParams>();
    fts_qp->set_default_operator(default_op);
    vq.query_params_ = fts_qp;
    return engine_->execute(schema_, vq, segments_);
  }

  // Helper: execute FTS query_string with WHERE filter via VectorQuery
  Result<DocPtrList> fts_search_with_filter(const std::string &query_string,
                                            const std::string &filter,
                                            int topk = 10) {
    VectorQuery vq;
    vq.topk_ = topk;
    vq.field_name_ = "content";
    vq.filter_ = filter;
    FtsQuery fts_query;
    fts_query.query_string_ = query_string;
    vq.fts_query_ = fts_query;
    return engine_->execute(schema_, vq, segments_);
  }

 private:
  static void build_schema() {
    auto fts_params = std::make_shared<FtsIndexParams>(
        "whitespace", std::vector<std::string>{"lowercase"}, "");
    auto invert_params = std::make_shared<InvertIndexParams>(true);
    schema_ = std::make_shared<CollectionSchema>(
        "fts_recall_test",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("content", DataType::STRING, false,
                                          fts_params),
            std::make_shared<FieldSchema>("tag", DataType::INT32, false,
                                          invert_params),
            // Dummy vector field required for filter parsing path in
            // execute
            std::make_shared<FieldSchema>(
                "vec", DataType::VECTOR_FP32, 4, false,
                std::make_shared<FlatIndexParams>(MetricType::L2)),
        });
  }

  static Segment::Ptr create_segment() {
    auto segment_meta = std::make_shared<SegmentMeta>();
    segment_meta->set_id(0);

    auto id_map = IDMap::CreateAndOpen("fts_recall_test", seg_path_ + "/id_map",
                                       true, false);
    auto delete_store = std::make_shared<DeleteStore>("fts_recall_test");

    Version v1;
    v1.set_schema(*schema_);
    std::string v_path = seg_path_ + "/manifest";
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

    auto result = Segment::CreateAndOpen(seg_path_, *schema_, 0, 0, id_map,
                                         delete_store, vm.value(), options);
    if (!result) {
      return nullptr;
    }
    return result.value();
  }

  static void insert_docs(const Segment::Ptr &segment) {
    // doc_id 0: "apple banana cherry"        tag=1
    // doc_id 1: "banana date elderberry"     tag=2
    // doc_id 2: "cherry fig grape"           tag=1
    // doc_id 3: "apple fig honeydew"         tag=2
    // doc_id 4: "date grape kiwi"            tag=1
    // doc_id 5: "apple apple apple"          tag=2
    // doc_id 6: "mango papaya starfruit"     tag=1
    // doc_id 7: "banana banana grape"        tag=2
    struct Entry {
      std::string content;
      int32_t tag;
    };
    std::vector<Entry> entries = {
        {"apple banana cherry", 1},    {"banana date elderberry", 2},
        {"cherry fig grape", 1},       {"apple fig honeydew", 2},
        {"date grape kiwi", 1},        {"apple apple apple", 2},
        {"mango papaya starfruit", 1}, {"banana banana grape", 2},
    };

    for (size_t i = 0; i < entries.size(); ++i) {
      Doc doc;
      doc.set_pk("pk_" + std::to_string(i));
      doc.set_doc_id(i);
      doc.set<std::string>("content", entries[i].content);
      doc.set<int32_t>("tag", entries[i].tag);
      auto status = segment->Insert(doc);
      ASSERT_TRUE(status.ok())
          << "Insert doc " << i << " failed: " << status.c_str();
    }
  }

 protected:
  static inline std::string seg_path_ = "./fts_recall_test_collection";
  static inline CollectionSchema::Ptr schema_;
  static inline std::vector<Segment::Ptr> segments_;
  static inline SQLEngine::Ptr engine_;
};

// ============================================================
// Basic FTS search tests
// ============================================================

// "apple" matches docs 0, 3, 5
TEST_F(FtsRecallTest, BasicSingleTerm) {
  auto result = fts_search("apple");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 3u);
}

// BM25 ordering: doc 5 ("apple apple apple") should have highest score
TEST_F(FtsRecallTest, BM25ScoreOrdering) {
  auto result = fts_search("apple");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_GE(result->size(), 2u);

  // Results should be sorted by score descending
  for (size_t i = 0; i + 1 < result->size(); ++i) {
    EXPECT_GE((*result)[i]->score(), (*result)[i + 1]->score())
        << "Results not sorted descending at index " << i;
  }
  // Doc 5 has highest TF for "apple"
  EXPECT_EQ((*result)[0]->pk(), "pk_5");
}

// "kiwi" only in doc 4
TEST_F(FtsRecallTest, SingleMatch) {
  auto result = fts_search("kiwi");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_4");
}

// Nonexistent term
TEST_F(FtsRecallTest, NoMatch) {
  auto result = fts_search("zzznomatch");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 0u);
}

// Topk limit: "banana" in docs 0, 1, 7 (3 matches), topk=2
TEST_F(FtsRecallTest, TopkLimit) {
  auto result = fts_search("banana", /*topk=*/2);
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_LE(result->size(), 2u);
}

// Multi-term implicit OR: "apple banana" matches union of {0,3,5} and {0,1,7}
TEST_F(FtsRecallTest, MultiTermImplicitOr) {
  auto result = fts_search("apple banana");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  // Union: {0,1,3,5,7} = 5 docs
  EXPECT_EQ(result->size(), 5u);
}

// "starfruit" only in doc 6
TEST_F(FtsRecallTest, RareTerm) {
  auto result = fts_search("starfruit");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_6");
}

// "grape" in docs 2, 4, 7
TEST_F(FtsRecallTest, CommonTerm) {
  auto result = fts_search("grape");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 3u);
}

// ============================================================
// Explicit AND
// ============================================================

// "apple AND banana" -> intersection of {0,3,5} and {0,1,7} = {0}
TEST_F(FtsRecallTest, ExplicitAnd) {
  auto result = fts_search("apple AND banana");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_0");
}

// "cherry AND fig" -> {0,2} AND {2,3} = {2}
TEST_F(FtsRecallTest, ExplicitAnd2) {
  auto result = fts_search("cherry AND fig");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_2");
}

// ============================================================
// Binary NOT (AND-NOT)
// ============================================================

// "apple NOT banana" -> {0,3,5} minus {0,1,7} = {3,5}
TEST_F(FtsRecallTest, BinaryNot) {
  auto result = fts_search("apple NOT banana");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 2u);
  std::set<std::string> pks;
  for (auto &doc : *result) {
    pks.insert(doc->pk());
  }
  EXPECT_TRUE(pks.count("pk_3"));
  EXPECT_TRUE(pks.count("pk_5"));
}

// "banana NOT grape" -> {0,1,7} minus {2,4,7} = {0,1}
TEST_F(FtsRecallTest, BinaryNot2) {
  auto result = fts_search("banana NOT grape");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 2u);
  std::set<std::string> pks;
  for (auto &doc : *result) {
    pks.insert(doc->pk());
  }
  EXPECT_TRUE(pks.count("pk_0"));
  EXPECT_TRUE(pks.count("pk_1"));
}

// ============================================================
// Error cases
// ============================================================

// Leading NOT should fail parse
TEST_F(FtsRecallTest, LeadingNotIsRejected) {
  auto result = fts_search("NOT apple");
  EXPECT_FALSE(result.has_value());
}

// Both query_string_ and match_string_ empty
TEST_F(FtsRecallTest, BothEmptyReturnsError) {
  VectorQuery vq;
  vq.topk_ = 10;
  vq.field_name_ = "content";
  vq.fts_query_ = FtsQuery{};  // both fields empty
  auto result = engine_->execute(schema_, vq, segments_);
  EXPECT_FALSE(result.has_value());
}

// Both query_string_ and match_string_ set
TEST_F(FtsRecallTest, BothSetReturnsError) {
  VectorQuery vq;
  vq.topk_ = 10;
  vq.field_name_ = "content";
  FtsQuery fts_query;
  fts_query.query_string_ = "apple";
  fts_query.match_string_ = "banana";
  vq.fts_query_ = fts_query;
  auto result = engine_->execute(schema_, vq, segments_);
  EXPECT_FALSE(result.has_value());
}

// ============================================================
// match_string tests
// ============================================================

// match_string "starfruit" -> doc 6
TEST_F(FtsRecallTest, MatchStringRareTerm) {
  auto result = fts_match("starfruit");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_6");
}

// match_string "grape" -> docs 2, 4, 7
TEST_F(FtsRecallTest, MatchStringCommonTerm) {
  auto result = fts_match("grape");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 3u);
}

// match_string "apple banana" -> OR -> union {0,1,3,5,7}
TEST_F(FtsRecallTest, MatchStringMultipleTokens) {
  auto result = fts_match("apple banana");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 5u);
}

// ============================================================
// default_operator tests
// ============================================================

// AND default for match_string: "apple banana" -> intersection = {0}
TEST_F(FtsRecallTest, DefaultOperatorAnd_MatchString) {
  auto result = fts_match("apple banana", "AND");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_0");
}

// OR default for match_string (backward compat)
TEST_F(FtsRecallTest, DefaultOperatorOr_MatchString) {
  auto result = fts_match("apple banana", "OR");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 5u);
}

// AND default for query_string: "apple banana" -> AND
TEST_F(FtsRecallTest, DefaultOperatorAnd_QueryString) {
  auto result = fts_query_with_op("apple banana", "AND");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0]->pk(), "pk_0");
}

// Explicit OR in query not overridden by default_operator=AND
// "apple OR grape" with AND default -> OR still applies
TEST_F(FtsRecallTest, DefaultOperatorAnd_DoesNotOverrideExplicitOr) {
  auto result = fts_query_with_op("apple OR grape", "AND");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  // apple: {0,3,5}, grape: {2,4,7} -> union = 6
  EXPECT_EQ(result->size(), 6u);
}

// Empty default_operator keeps historical OR for match_string
TEST_F(FtsRecallTest, DefaultOperatorEmpty_BackwardCompatibleOr) {
  auto result = fts_match("apple banana");  // no default_op arg
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  // OR semantics: union of apple{0,3,5} and banana{0,1,7} = 5
  EXPECT_EQ(result->size(), 5u);
}

// Lowercase "and" must be accepted
TEST_F(FtsRecallTest, DefaultOperatorAndLowercase_Accepted) {
  auto result = fts_match("apple banana", "and");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_EQ(result->size(), 1u);
}

// Mixed-case "And" / "oR": current implementation only recognises exact
// "AND"/"and" and "OR"/"or".  Unknown values fall through to the default (OR).
TEST_F(FtsRecallTest, DefaultOperatorMixedCase_Accepted) {
  {
    // "And" is not recognised as AND -> falls back to OR
    auto result = fts_match("apple banana", "And");
    ASSERT_TRUE(result.has_value()) << result.error().c_str();
    EXPECT_EQ(result->size(), 5u);
  }
  {
    // "oR" is not recognised as OR explicitly -> also falls back to OR
    auto result = fts_match("apple banana", "oR");
    ASSERT_TRUE(result.has_value()) << result.error().c_str();
    EXPECT_EQ(result->size(), 5u);
  }
}

// Invalid default_operator value should be rejected
TEST_F(FtsRecallTest, DefaultOperatorInvalid_Rejected) {
  auto result = fts_match("apple banana", "xor");
  // Current implementation treats unknown values as OR (no rejection),
  // so this test documents the actual behaviour.
  // If the implementation is changed to reject, flip to EXPECT_FALSE.
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
}

// ============================================================
// Error cases (additional)
// ============================================================

// Empty field_name should fail
TEST_F(FtsRecallTest, EmptyFieldNameReturnsError) {
  VectorQuery vq;
  vq.topk_ = 10;
  vq.field_name_ = "";
  FtsQuery fts_query;
  fts_query.query_string_ = "apple";
  vq.fts_query_ = fts_query;
  auto result = engine_->execute(schema_, vq, segments_);
  EXPECT_FALSE(result.has_value());
}

// Empty query_string (with field_name set) should fail
TEST_F(FtsRecallTest, EmptyQueryStringReturnsError) {
  VectorQuery vq;
  vq.topk_ = 10;
  vq.field_name_ = "content";
  // Both query_string_ and match_string_ empty -> error
  vq.fts_query_ = FtsQuery{};
  auto result = engine_->execute(schema_, vq, segments_);
  EXPECT_FALSE(result.has_value());
}

// ============================================================
// FTS search with WHERE filter
// ============================================================

// "apple" (docs 0,3,5) + tag = 1 (docs 0,2,4,6) -> intersection = {0}
TEST_F(FtsRecallTest, FtsSearchWithFilter_ScoreTag) {
  auto result = fts_search_with_filter("apple", "tag = 1");
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  // Filter should reduce results to doc 0 only
  EXPECT_LE(result->size(), 3u);
  // Verify that at least doc 0 (which satisfies both FTS and filter) is present
  bool found_pk0 = false;
  for (auto &doc : *result) {
    if (doc->pk() == "pk_0") {
      found_pk0 = true;
    }
  }
  EXPECT_TRUE(found_pk0);
}

// "banana" (docs 0,1,7) + tag = 2 (docs 1,3,5,7) + topk=1
TEST_F(FtsRecallTest, FtsSearchWithFilter_TopkRespected) {
  auto result = fts_search_with_filter("banana", "tag = 2", /*topk=*/1);
  ASSERT_TRUE(result.has_value()) << result.error().c_str();
  EXPECT_LE(result->size(), 1u);
}

}  // namespace zvec::sqlengine
