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

#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

/*! Index Reformer
 */
class IndexReformer : public IndexModule {
 public:
  //! Index Reformer Pointer
  typedef std::shared_ptr<IndexReformer> Pointer;

  //! Destructor
  virtual ~IndexReformer(void) {}

  //! Initialize Reformer
  virtual int init(const ailego::Params &params) = 0;

  //! Cleanup Reformer
  virtual int cleanup(void) = 0;

  //! Load index from container
  virtual int load(IndexStorage::Pointer cntr) = 0;

  //! Unload index
  virtual int unload(void) = 0;

  //! Transform a query
  virtual int transform(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/, std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Transform queries
  virtual int transform(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
                        std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Convert a record
  virtual int convert(const void *record, const IndexQueryMeta &rmeta,
                      std::string *out, IndexQueryMeta *ometa) const {
    return this->transform(record, rmeta, out, ometa);
  }

  //! Convert records
  virtual int convert(const void *records, const IndexQueryMeta &rmeta,
                      uint32_t count, std::string *out,
                      IndexQueryMeta *ometa) const {
    return this->transform(records, rmeta, count, out, ometa);
  }

  //! Normalize results
  virtual int normalize(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/,
                        IndexDocumentList & /*result*/) const {
    return IndexError_NotImplemented;
  }

  virtual bool need_revert() const {
    return false;
  }

  virtual int revert(const void * /*in*/, const IndexQueryMeta & /*qmeta*/,
                     std::string * /*out*/) const {
    return IndexError_NotImplemented;
  }

  //! Transform a query
  virtual int transform(uint32_t /*sparse_count*/,
                        const uint32_t * /*sparse_indices*/,
                        const void * /*sparse_query*/,
                        const IndexQueryMeta & /*qmeta*/, std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Transform queries
  virtual int transform(const uint32_t * /*sparse_count*/,
                        const uint32_t * /*sparse_indices*/,
                        const void * /*sparse_query*/,
                        const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
                        std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Convert a record
  virtual int convert(uint32_t sparse_count, const uint32_t *sparse_indices,
                      const void *sparse_query, const IndexQueryMeta &qmeta,
                      std::string *out, IndexQueryMeta *ometa) const {
    return this->transform(sparse_count, sparse_indices, sparse_query, qmeta,
                           out, ometa);
  }

  //! Convert records
  virtual int convert(const uint32_t *sparse_count,
                      const uint32_t *sparse_indices, const void *sparse_query,
                      const IndexQueryMeta &qmeta, uint32_t count,
                      std::string *out, IndexQueryMeta *ometa) const {
    return this->transform(sparse_count, sparse_indices, sparse_query, qmeta,
                           count, out, ometa);
  }

  virtual int revert(const uint32_t /*sparse_count*/,
                     const uint32_t * /*sparse_indices*/,
                     const void * /*sparse_query*/,
                     const IndexQueryMeta & /*qmeta*/,
                     std::string * /*sparse_query_out*/) const {
    return 0;
  }
};

}  // namespace core
}  // namespace zvec
