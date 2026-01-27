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

#include <ailego/parallel/lock.h>
#include <zvec/core/framework/index_streamer.h>
#include "flat_sparse_streamer_entity.h"

namespace zvec {
namespace core {

/*! Flat Sparse Streamer
 */
class FlatSparseStreamer : public IndexStreamer {
 public:
  static const uint32_t VERSION;

 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

  FlatSparseStreamer(void);
  virtual ~FlatSparseStreamer(void);

  FlatSparseStreamer(const FlatSparseStreamer &streamer) = delete;
  FlatSparseStreamer &operator=(const FlatSparseStreamer &streamer) = delete;

 public:
  //! Initialize Streamer
  int init(const IndexMeta &, const ailego::Params &) override;

  //! Cleanup Streamer
  int cleanup(void) override;

  //! Open index from file path
  int open(IndexStorage::Pointer stg) override;

  //! Close file
  int close(void) override;

  //! flush file
  int flush(uint64_t checkpoint) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Create a context
  ContextPointer create_context(void) const override;

  //! Create a new iterator
  IndexStreamer::SparseProvider::Pointer create_sparse_provider(
      void) const override;

  int add_impl(uint64_t pkey, const uint32_t sparse_count,
               const uint32_t *sparse_indices, const void *sparse_query,
               const IndexQueryMeta &qmeta, Context::Pointer &context) override;

  int add_with_id_impl(uint32_t pkey, const uint32_t sparse_count,
                       const uint32_t *sparse_indices, const void *sparse_query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override;

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t *sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta, uint32_t count,
                     Context::Pointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const uint32_t sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               ContextPointer &context) const override;

  //! Linear search by primary keys with sparse inputs
  int search_bf_by_p_keys_impl(const uint32_t *sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               ContextPointer &context) const override;

  //! Fetch sparse vector by key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override;

  int get_sparse_vector_by_id(
      uint32_t id, uint32_t *sparse_count, std::string *sparse_indices_buffer,
      std::string *sparse_values_buffer) const override {
    return get_sparse_vector(id, sparse_count, sparse_indices_buffer,
                             sparse_values_buffer);
  }

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }
  const FlatSparseStreamerEntity &entity(void) const {
    return entity_;
  }

  uint32_t magic(void) const {
    return magic_;
  }

 private:
  inline int check_params(const IndexQueryMeta &qmeta) const {
    if (ailego_unlikely(qmeta.data_type() != meta_.data_type())) {
      LOG_ERROR("Unsupported query meta, type=%d, expected=%d",
                qmeta.data_type(), meta_.data_type());
      return IndexError_Mismatch;
    }
    return 0;
  }

  int do_search(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                const void *sparse_query, bool with_p_keys,
                const std::vector<std::vector<uint64_t>> &p_keys,
                const IndexQueryMeta &qmeta, uint32_t count,
                ContextPointer &context) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_OPENED = 2 };

  IndexMeta meta_{};
  FlatSparseStreamerEntity entity_;

  uint32_t magic_{0U};
  Stats stats_{};
  State state_{STATE_INIT};

  //! avoid add vector while dumping index
  ailego::SharedMutex shared_mutex_{};
};

}  // namespace core
}  // namespace zvec
