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


#pragma once

#include <set>
#include <shared_mutex>
#include <rocksdb/cache.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/schema.h>
#include "db/common/concurrent_roaring_bitmap.h"
#include "db/common/rocksdb_context.h"
#include "inverted_codec.h"
#include "inverted_doc_range.h"
#include "inverted_search_result.h"


namespace zvec {


//
// An inverted column indexer manages document-term indexing in two states:
// 1. Streaming:
//      - allows insertion of new terms and indexing of document-term pairs
// 2. Sealed:
//      - read-only, no further writes permitted
//      - statistical index generated when range optimization is enabled
//
//
// InvertedColumnIndexer requires document IDs to be sequential integers
// starting from 0 without any gaps.
// While documents can be inserted in any order, the complete sequence from 0 to
// max id must be present before sealing the index.
//
// Multiple inverted column indexers share the same RocksDB instance but each
// indexer uses its own RocksDB column families within that shared RocksDB
// instance. This design allows for efficient resource utilization while
// maintaining data separation between different columns.
//
class InvertedColumnIndexer {
 public:
  using Ptr = std::shared_ptr<InvertedColumnIndexer>;


  static Ptr CreateAndOpen(const std::string &collection_name,
                           const FieldSchema &field, RocksdbContext &context,
                           bool read_only = false);


  virtual ~InvertedColumnIndexer();


  InvertedColumnIndexer(const InvertedColumnIndexer &) = delete;
  InvertedColumnIndexer(InvertedColumnIndexer &&) = delete;
  InvertedColumnIndexer &operator=(const InvertedColumnIndexer &) = delete;
  InvertedColumnIndexer &operator=(InvertedColumnIndexer &&) = delete;

 protected:
  explicit InvertedColumnIndexer(const std::string &collection_name,
                                 const FieldSchema &field,
                                 RocksdbContext &context, bool read_only)
      : collection_name_(collection_name),
        field_(field),
        path_(context.db_path_),
        ctx_(context),
        read_only_(read_only) {}

  // TODO： for ut, remove this
  InvertedColumnIndexer(RocksdbContext &ctx) : ctx_(ctx) {}


 public:
  /*!
   * \brief Search for documents matching the given value and operation
   * \param value The value to compare against (e.g., "5", "10")
   * \param op The comparison operation (e.g., EQ, GT, LT)
   * \return Pointer to search results containing matching documents
   */
  virtual InvertedSearchResult::Ptr search(const std::string &value,
                                           CompareOp op) const;

  // Candidate membership only: uninserted docIDs may be present. Use search()
  // when enumerating IDs, including brute-force vector candidate generation.
  InvertedSearchResult::Ptr search_filter(const std::string &value,
                                          CompareOp op) const;

  bool supports_bsi(CompareOp op) const {
    return bsi_ && (op == CompareOp::LT || op == CompareOp::LE ||
                    op == CompareOp::GT || op == CompareOp::GE);
  }

  bool has_range_statistics() const {
    return sealed_ && doc_range_stat_ != nullptr;
  }


  /*!
   * \brief Search for documents matching multiple values
   * \param values List of values to compare against (e.g., {"5", "10", "15"})
   * \param comp_op The comparison operation to apply (e.g., CONTAIN_ANY)
   * \return Pointer to search results containing matching documents
   */
  virtual InvertedSearchResult::Ptr multi_search(
      const std::vector<std::string> &values, CompareOp op) const;


  /*!
   * \brief Search for documents matching array length
   * \param len The array length value to compare against
   * \param op The comparison operation to apply (e.g., EQ)
   * \return Pointer to search results containing matching documents
   */
  virtual InvertedSearchResult::Ptr search_array_len(uint32_t len,
                                                     CompareOp op) const;


  /*!
   * \brief Search for documents that have null values
   * \return Pointer to search results containing matching documents
   */
  virtual InvertedSearchResult::Ptr search_null() const;


  /*!
   * \brief Search for documents that have non-null values
   * \return Pointer to search results containing matching documents
   */
  virtual InvertedSearchResult::Ptr search_non_null() const;


  /*!
   * \brief Evaluate the ratio of matching documents compared to total documents
   * \param value The value to compare against (e.g., "5", "10")
   * \param op The comparison operation (e.g., EQ, GT, LT)
   * \param total_size Pointer to store the total number of documents
   * \param range_size Pointer to store the number of matching documents
   * \return Status indicating success or failure of the evaluation
   */
  virtual Status evaluate_ratio(const std::string &value, CompareOp op,
                                uint64_t *total_size,
                                uint64_t *range_size) const;


  /*!
   * \brief Insert a document-term pair into the inverted index
   * \param id The document ID to insert
   * \param value The string-encoded representation of the value to index. This
   *              parameter may contain either a single value or an array of
   *              values depending on the field type. The underlying data type
   *              might differ from std::string - the string serves as a generic
   *              serialization buffer for the actual typed data.
   * \return Status indicating success or failure of the insert operation
   */
  Status insert(uint32_t id, const std::string &value);


  /*!
   * \brief Insert a document with multiple strings into the inverted index
   * \param id The document ID to insert
   * \param values Multiple string values to index
   * \return Status indicating success or failure of the insert operation
   */
  Status insert(uint32_t id, const std::vector<std::string> &values);


  /*!
   * \brief Insert a document with a boolean value into the inverted index
   * \param id The document ID to insert
   * \param value The boolean value to index
   * \return Status indicating success or failure of the insert operation
   */
  Status insert(uint32_t id, bool value);


  /*!
   * \brief Insert a document with multiple booleans into the inverted index
   * \param id The document ID to insert
   * \param value Multiple boolean values to index
   * \return Status indicating success or failure of the insert operation
   */
  Status insert(uint32_t id, const std::vector<bool> &values);


  /*!
   * \brief Insert a document with null value into the inverted index
   * \param id The document ID to insert
   * \return Status indicating success or failure of the insert operation
   */
  Status insert_null(uint32_t id);


  /*!
   * \brief Serialize special values, e.g., null-value bitmap and max doc id
   * \return Status indicating success or failure of the serialization
   */
  Status flush_special_values();


  /*!
   * \brief Seal the index and generate statistical indexes
   * \return Status indicating success or failure of the operation
   */
  Status seal();


  /*!
   * \brief Check if the index is sealed
   * \return True if the index is sealed, false otherwise
   */
  inline bool is_sealed() const {
    return sealed_;
  }

  bool uses_bsi() const {
    return cf_bsi_ != nullptr;
  }

  /*!
   * \brief Drop the index storage
   * \return Status indicating success or failure of the operation
   */
  Status drop_storage();


  /*!
   * \brief Get the name of the corresponding collection
   * \return The name of the corresponding collection
   */
  inline const std::string &collection_name() const {
    return collection_name_;
  }


  inline const std::string id() const {
    return "InvertedColumnIndexer[collection:" + collection_name_ +
           "|field:" + field_.name() + "|path:'" + path_ + "']";
  }


 private:
  using Slice = rocksdb::Slice;
  using PinnableSlice = rocksdb::PinnableSlice;


  Status open();

  void update_bsi(uint32_t id, const std::string &term);
  Status append_bsi_snapshot(rocksdb::WriteBatch *batch);
  Status flush_special_values_unlocked();
  Status restore_bsi_state();
  Status freeze_bsi_state();
  Result<roaring_bitmap_t *> get_bitmap_bsi(const std::string &term,
                                            CompareOp op) const;

  inline std::string encode(const std::string &term) const {
    return InvertedIndexCodec::Encode(term, field_.element_data_type());
  }

  inline std::vector<std::string> encode_array(const std::string &terms) const {
    std::vector<std::string> result{};
    size_t s = field_.element_data_size();
    if (s == 0) {
      return result;
    }
    size_t num_terms = terms.size() / s;
    result.reserve(num_terms);
    for (size_t i = 0; i < num_terms; ++i) {
      std::string_view sv(terms.data() + (i * s), s);
      result.emplace_back(
          InvertedIndexCodec::Encode(sv, field_.element_data_type()));
    }
    return result;
  }

  inline std::vector<std::string> encode(
      const std::vector<std::string> &terms) const {
    std::vector<std::string> result;
    result.reserve(terms.size());
    for (auto &term : terms) {
      result.emplace_back(encode(term));
    }
    return result;
  }

  inline std::string encode(bool value) {
    return InvertedIndexCodec::Encode(value);
  }

  inline std::string encode_reversed(const std::string &term) const {
    return InvertedIndexCodec::encode_reversed(term);
  }

  inline int cmp(const char *s1, size_t s1_len, const char *s2,
                 size_t s2_len) const {
    return InvertedIndexCodec::CMP(s1, s1_len, s2, s2_len);
  }

  inline bool cmp_lt(const char *s1, size_t s1_len, const char *s2,
                     size_t s2_len, bool include_eq) const {
    int ret = InvertedIndexCodec::CMP(s1, s1_len, s2, s2_len);
    return (include_eq && ret <= 0) || (!include_eq && ret < 0);
  }

  inline bool has_prefix(const char *value, size_t value_len,
                         const char *prefix, size_t prefix_len) const {
    return InvertedIndexCodec::has_prefix(value, value_len, prefix, prefix_len);
  }

  inline void update_doc_id_limit(uint32_t id) {
    const uint64_t limit = uint64_t{id} + 1;
    uint64_t expected_limit = doc_id_limit_.load();
    while (expected_limit < limit &&
           !doc_id_limit_.compare_exchange_weak(expected_limit, limit)) {
    }
  }

  inline Status estimate_range_ratio(const std::string &term, CompareOp op,
                                     uint64_t *total_count,
                                     uint64_t *matching_count) const;

  inline bool range_covers_most_values(const std::string &term,
                                       CompareOp op) const;

  inline roaring_bitmap_t *flip_bitmap(roaring_bitmap_t *bitmap) const;

  Result<roaring_bitmap_t *> get_bitmap_eq(const std::string &term) const;

  Result<roaring_bitmap_t *> get_bitmap_contain(
      const std::vector<std::string> &terms, bool is_any) const;

  Result<roaring_bitmap_t *> get_bitmap_ne(const std::string &term) const;

  Result<roaring_bitmap_t *> get_bitmap_not_contain(
      const std::vector<std::string> &terms, bool is_any) const;

  Result<roaring_bitmap_t *> get_bitmap_lt(const std::string &term,
                                           bool include_eq) const;

  Result<roaring_bitmap_t *> get_bitmap_gt(const std::string &term,
                                           bool include_eq) const;

  Result<roaring_bitmap_t *> get_bitmap_array_len_eq(uint32_t len) const;

  Result<roaring_bitmap_t *> get_bitmap_array_len_ne(uint32_t len) const;

  Result<roaring_bitmap_t *> get_bitmap_array_len_lt(uint32_t len,
                                                     bool include_eq) const;

  Result<roaring_bitmap_t *> get_bitmap_array_len_gt(uint32_t len,
                                                     bool include_eq) const;

  Result<roaring_bitmap_t *> get_bitmap_like(std::string term) const;

  Result<roaring_bitmap_t *> get_bitmap_prefix(
      const std::string &term, const std::string &suffix = "") const;

  Result<roaring_bitmap_t *> get_bitmap_suffix(const std::string &term) const;

  Result<roaring_bitmap_t *> get_bitmap_null() const;

  Result<roaring_bitmap_t *> get_bitmap_non_null() const;

  rocksdb::Status index_array_len(uint32_t id, uint32_t len);

  Status generate_statistical_indexes();


 private:
  inline std::string cf_name_terms() const {
    return field_.name() + INVERT_SUFFIX_TERMS;
  };

  inline std::string cf_name_reversed_terms() const {
    return field_.name() + INVERT_SUFFIX_REVERSED_TERMS;
  };

  inline std::string cf_name_array_len() const {
    return field_.name() + INVERT_SUFFIX_ARRAY_LEN;
  };

  inline std::string cf_name_ranges() const {
    return field_.name() + INVERT_SUFFIX_RANGES;
  };

  inline std::string cf_name_cdf() const {
    return INVERT_CDF;
  };

  inline std::string cf_name_bsi() const {
    return field_.name() + INVERT_SUFFIX_BSI;
  }

  inline std::string key_max_id() const {
    return field_.name() + INVERT_KEY_MAX_ID;
  };

  inline std::string key_doc_id_limit() const {
    return field_.name() + INVERT_KEY_DOC_ID_LIMIT;
  }

  inline std::string key_null() const {
    return field_.name() + INVERT_KEY_NULL;
  };

  inline std::string key_sealed() const {
    return field_.name() + INVERT_KEY_SEALED;
  };

  inline bool allow_range_optimization(const FieldSchema &field) const {
    bool not_allowed =
        field.is_array_type() || field.data_type() == DataType::BOOL;
    return !not_allowed;
  }

  inline bool allow_extended_wildcard(const FieldSchema &field) const {
    return field.data_type() == DataType::STRING;
  }


 private:
  const std::string collection_name_{};
  const FieldSchema field_{};
  const std::string path_{};


  // Column families:
  // 1. cf_terms_:              Inverted index for terms
  // 2. cf_reversed_terms_:     Inverted index for reversed terms
  // 3. cf_array_len_:          Inverted index for array length
  // 4. cf_ranges_:             Range index
  // 5. cf_cdf_:                Cumulative distribution function
  //    cf_bsi_:                Optional segment-wide bit slices
  // 6. default cf:             Stores special values
  //                              - null-value bitmap
  //                              - max id
  //                              - is_sealed
  //
  // Some column families are optional and may be nullptr.
  // For example, cf_ranges_ is nullptr when the indexer is not sealed (range
  // index not yet generated) or when range optimization is explicitly disabled.
  RocksdbContext &ctx_;
  rocksdb::ColumnFamilyHandle *cf_terms_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_reversed_terms_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_array_len_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_ranges_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_cdf_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_bsi_{nullptr};

  struct BitmapDeleter {
    void operator()(roaring_bitmap_t *bitmap) const {
      roaring_bitmap_free(bitmap);
    }
  };
  using BitmapPtr = std::unique_ptr<roaring_bitmap_t, BitmapDeleter>;

  struct BsiState {
    // Frozen views must be destroyed before their backing buffers.
    std::vector<std::string> buffers;
    std::vector<BitmapPtr> slices;
    mutable std::shared_mutex mutex;
    std::shared_ptr<rocksdb::Cache> cache;
    uint64_t dirty_slices{0};
  };
  Status load_bsi_state(BsiState *state, bool frozen, BsiCacheMode cache_mode);
  std::unique_ptr<BsiState> bsi_;


  std::atomic<bool> read_only_{false};
  std::atomic<bool> sealed_{false};
  bool enable_range_optimization_{false};
  bool enable_extended_wildcard_{false};
  // Upper bound, not a completed prefix: holes are allowed in BSI filters.
  std::atomic<uint64_t> doc_id_limit_{0};
  ConcurrentRoaringBitmap32 null_bitmap_{};
  SegmentDocRangeStat::Ptr doc_range_stat_{nullptr};
};


};  // namespace zvec
