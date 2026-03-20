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
#include <string>
#include <utility>
#include <variant>
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
  Status Open(const vector_column_params::ReadOptions &read_options);

  Status Flush();

  // Close will call Flush()
  Status Close();

  // Destroy will call Close() and remove index file
  Status Destroy();


  // If HNSWIndexer.merge([FlatIndexer1, FlatIndexer2])
  // then the merged indexer is a HNSWIndexer
  Status Merge(const std::vector<VectorColumnIndexer::Ptr> &indexers,
               const IndexFilter::Ptr &filter = nullptr,
               const vector_column_params::MergeOptions &merge_options = {});
  // TODO: should we use this function? or a Reducer?
  //  TODO: sstatic reduce, optimize; iterator/scan


  //! Insert vector
  Status Insert(const vector_column_params::VectorData &vector_data,
                uint32_t doc_id);
  // TODO: batch insert

  virtual Result<IndexResults::Ptr> Search(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params);
  // Result<std::vector<IndexResults::Ptr>> BatchSearch(
  //     const VectorDataset &vector_data,
  //     const  vector_column_params::QueryParams &query_params);

  Result<vector_column_params::VectorDataBuffer> Fetch(uint32_t doc_id) const;
  // Result<VectorDataset> BatchFetch(const std::vector<uint32_t> &doc_ids)
  // const;

  core::IndexProvider::Pointer create_index_provider() const {
    return index->create_index_provider();
  }

 public:
  std::string index_file_path() const {
    return index_file_path_;
  }

  size_t doc_count() const {
    if (index == nullptr) {
      return -1;
    }
    return index->GetDocCount();
  }

  // for ut
 protected:
  VectorColumnIndexer() = default;

 private:
  // protected:
  //  virtual bool init_proxima_params() = 0;

  // proxima or other engine index param like VSAGE
  // build proxima index
  Status CreateProximaIndex(
      const vector_column_params::ReadOptions &read_options);

 protected:
  friend ProximaEngineHelper;
  core_interface::Index::Pointer index;
  FieldSchema field_schema_{};
  std::string index_file_path_{};


  std::string engine_name_ = "proxima";
  bool is_sparse_{false};  // TODO: eliminate the dynamic flag and make it
                           // static/template/seperate class
};


}  // namespace zvec
