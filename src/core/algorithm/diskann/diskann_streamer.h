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

#include <zvec/core/framework/index_framework.h>
#include "diskann_context.h"
#include "diskann_indexer.h"

class LinuxAlignedFileReader;

namespace zvec {
namespace core {

class DiskAnnStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

 public:
  DiskAnnStreamer(void);
  ~DiskAnnStreamer(void);

  DiskAnnStreamer(const DiskAnnStreamer &) = delete;
  DiskAnnStreamer &operator=(const DiskAnnStreamer &) = delete;

 protected:
  //! Initialize Searcher
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup Searcher
  int cleanup(void) override;

  //! Load Index from storage
  int open(IndexStorage::Pointer storage) override;

  //! Unload index from storage
  int unload(void) override;

  //! KNN Search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  ContextPointer &context) const override {
    return search_impl(query, qmeta, 1, context);
  }

  //! KNN Search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, ContextPointer &context) const override;

  //! Linear Search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     ContextPointer &context) const override {
    return search_bf_impl(query, qmeta, 1, context);
  }

  //! Linear Search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, ContextPointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               ContextPointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query, const uint32_t sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(query, &sparse_count, sparse_indices,
                                    sparse_query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(
      const void * /*query*/, const uint32_t * /*sparse_count*/,
      const uint32_t * /*sparse_indices*/, const void * /*sparse_query*/,
      const std::vector<std::vector<uint64_t>> & /*p_keys*/,
      const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
      ContextPointer & /*context*/) const override {
    return IndexError_NotImplemented;
  }

  //! Get vector by key
  int get_vector(uint64_t key, Context::Pointer &context,
                 std::string &vector) const override;

  //! Fetch vector by id
  const void *get_vector_by_id(uint32_t id) const override;

  //! Fetch vector by id into memory block
  int get_vector_by_id(const uint32_t id,
                       IndexStorage::MemoryBlock &block) const override;

  //! Create a searcher context
  ContextPointer create_context() const override;

  //! Create a vector iterator backed by the on-disk vector segment.
  //! Used by the merge code path (``MixedStreamerReducer``) to walk every
  //! vector held by this streamer.
  IndexSearcher::Provider::Pointer create_provider(void) const override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  virtual int flush(uint64_t /*check_point*/) override {
    return 0;
  }

  virtual int close(void) override {
    return this->unload();
  }

  void print_debug_info() override;

 private:
  template <typename T, typename LabelT = uint32_t>
  int search_disk_index(const std::string &query_file,
                        const uint32_t num_nodes_to_cache,
                        const uint32_t recall_at, const uint32_t beamwidth);

  //! To share ctx across streamer/searcher, we need to update the context for
  //! current streamer/searcher
  int update_context(DiskAnnContext *ctx) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  IndexMetric::Pointer measure_{};
  IndexMeta meta_{};
  ailego::Params params_{};

  uint32_t list_size_{200};
  uint32_t cache_nodes_num_{0};

  bool warm_up_{false};
  uint32_t beam_size_{2};

  DiskAnnIndexer::Pointer diskann_indexer_{nullptr};
  DiskAnnSearcherEntity entity_{};

  // Mutable members for get_vector_by_id (caches context and buffer)
  mutable ContextPointer fetch_ctx_{};
  mutable std::string fetch_vector_buffer_;

  uint32_t magic_{0U};

  Stats stats_;
  State state_{STATE_INIT};
};

}  // namespace core
}  // namespace zvec
