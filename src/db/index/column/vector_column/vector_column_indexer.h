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
#include <shared_mutex>
#include <string>
#include <utility>
#include <variant>
#include <roaring.hh>
#include <ailego/parallel/lock.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/typedef.h"
#include "db/index/column/common/index_results.h"
#include "db/index/common/meta.h"
#include "zvec/core/framework/index_provider.h"
#include "vector_column_params.h"
#include "vector_index_results.h"

namespace zvec {

class ProximaEngineHelper;

class VectorColumnIndexer {
 public:
  using Ptr = std::shared_ptr<VectorColumnIndexer>;
  PROXIMA_DISALLOW_COPY_AND_ASSIGN(VectorColumnIndexer);

  VectorColumnIndexer(const std::string &index_file_path,
                      const FieldSchema &field_schema,
                      const std::string &engine_name = "proxima")
      : field_schema_(field_schema),
        index_file_path_(index_file_path),
        engine_name_(engine_name) {
    // assert(field_schema.is_dense_vector() ||
    // field_schema.is_sparse_vector());
    is_sparse_ = field_schema.is_sparse_vector();
  }

  virtual ~VectorColumnIndexer() = default;

 public:
  Status open(const vector_column_params::ReadOptions &read_options);

  Status flush();

  // Close will call Flush()
  Status close();

  // Destroy will call Close() and remove index file
  Status destroy();


  // If HNSWIndexer.merge([FlatIndexer1, FlatIndexer2])
  // then the merged indexer is a HNSWIndexer
  Status merge(const std::vector<VectorColumnIndexer::Ptr> &indexers,
               const IndexFilter::Ptr &filter = nullptr,
               const vector_column_params::MergeOptions &merge_options = {});
  // TODO: should we use this function? or a Reducer?
  //  TODO: sstatic reduce, optimize; iterator/scan


  //! Insert vector
  Status insert(const vector_column_params::VectorData &vector_data,
                uint32_t doc_id);
  Status insert_null(uint32_t doc_id);

  // Logical rows include NULLs; core IDs in new nullable indexes are ranks
  // in the validity bitmap, so the core only ever indexes real vectors.
  uint32_t row_count() const {
    return row_count_;
  }
  bool has_validity() const {
    return field_schema_.nullable();
  }
  // TODO: batch insert

  virtual Result<IndexResults::Ptr> search(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params);

  // Dense top-k search without group-by, brute-force keys or vector fetching.
  // Writes IDs and optional scores into topk-sized caller-owned buffers.
  Status search_internal_ids(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params,
      int64_t *output_ids, float *output_scores,
      const VectorColumnIndexer *reference_indexer = nullptr);
  // Result<std::vector<IndexResults::Ptr>> BatchSearch(
  //     const VectorDataset &vector_data,
  //     const  vector_column_params::QueryParams &query_params);

  Result<vector_column_params::VectorDataBuffer> fetch(uint32_t doc_id) const;
  // Result<VectorDataset> BatchFetch(const std::vector<uint32_t> &doc_ids)
  // const;

  core::IndexProvider::Pointer create_index_provider() const {
    return index->create_index_provider();
  }

 public:
  std::string index_file_path() const {
    return index_file_path_;
  }

  const FieldSchema &field_schema() const {
    return field_schema_;
  }

  size_t doc_count() const {
    if (index == nullptr) {
      return -1;
    }
    return index->get_doc_count();
  }

  //! Debug-only accessor for the underlying core_interface Index.
  //! Intended for introspection/testing; not part of the stable API.
  core_interface::Index::Pointer debug_get_index() const {
    return index;
  }

  // for ut
 protected:
  VectorColumnIndexer() = default;

 private:
  Status load_validity(const vector_column_params::ReadOptions &options);
  Status flush_validity();
  Status ensure_empty_index();
  Status recreate_empty_index(bool use_empty_flat);
  bool has_vector(uint32_t row) const;
  uint32_t physical_id(uint32_t row) const;
  bool logical_id(uint64_t physical, uint32_t *row) const;
  // protected:
  //  virtual bool init_proxima_params() = 0;

  // proxima or other engine index param like VSAGE
  // build proxima index
  Status create_proxima_index(
      const vector_column_params::ReadOptions &read_options,
      bool use_empty_flat = true);

 protected:
  friend ProximaEngineHelper;
  core_interface::Index::Pointer index;
  FieldSchema field_schema_{};
  std::string index_file_path_{};


  std::string engine_name_ = "proxima";
  bool is_sparse_{false};  // TODO: eliminate the dynamic flag and make it
                           // static/template/seperate class
  mutable std::shared_mutex validity_mutex_;
  roaring::Roaring valid_rows_;
  uint32_t row_count_{0};
  bool compact_ids_{false};
  bool validity_dirty_{false};
  bool read_only_{false};
  bool use_mmap_{true};
  bool empty_flat_index_{false};
};


}  // namespace zvec
