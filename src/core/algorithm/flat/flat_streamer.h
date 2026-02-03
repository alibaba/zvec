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
#include "flat_streamer_entity.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Flat Streamer
 */
template <size_t BATCH_SIZE>
class FlatStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::UPointer;

  FlatStreamer(void);
  virtual ~FlatStreamer(void);

  FlatStreamer(const FlatStreamer &streamer) = delete;
  FlatStreamer &operator=(const FlatStreamer &streamer) = delete;

 public:
  //! Initialize Streamer
  int init(const IndexMeta &, const ailego::Params &) override;

  //! Cleanup Streamer
  int cleanup(void) override;

  //! Create a context
  IndexStreamer::Context::UPointer create_context(void) const override;

  //! Create a new iterator
  IndexProvider::Pointer create_provider(void) const override;

  //! Add a vector into index
  int add_impl(uint64_t pkey, const void *query, const IndexQueryMeta &qmeta,
               Context::UPointer &context) override;

  int add_with_id_impl(uint32_t id, const void *query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  Context::UPointer &context) const override {
    return search_bf_impl(query, qmeta, 1, context);
  }

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::UPointer &context) const override {
    return search_bf_impl(query, qmeta, count, context);
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     Context::UPointer &context) const override {
    return search_bf_impl(query, qmeta, 1, context);
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, Context::UPointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               Context::UPointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               Context::UPointer &context) const override;

  int group_by_search_impl(const void *query, const IndexQueryMeta &qmeta,
                           uint32_t count, Context::UPointer &context) const;

  int group_by_search_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, uint32_t count,
      Context::Pointer &context) const;

  //! Open index from file path
  int open(IndexStorage::Pointer stg) override;

  //! Close file
  int close(void) override;

  //! flush file
  int flush(uint64_t checkpoint) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  const FlatStreamerEntity &entity(void) const {
    return entity_;
  }

  virtual const void *get_vector(uint64_t key) const override {
    return this->get_vector_by_key(key);
  }

  virtual int get_vector(const uint64_t key,
                         IndexStorage::MemoryBlock &block) const override {
    return this->get_vector_by_key(key, block);
  }

  const void *get_vector_by_key(uint64_t key) const {
    return entity_.get_vector_by_key(key);
  }

  int get_vector_by_key(const uint64_t key,
                        IndexStorage::MemoryBlock &block) const override {
    return entity_.get_vector_by_key(key, block);
  }
  const void *get_vector_by_id(uint32_t id) const override {
    return get_vector_by_key(id);
  }

  int get_vector_by_id(const uint32_t id,
                       IndexStorage::MemoryBlock &block) const override {
    return get_vector_by_key(id, block);
  }

  uint32_t magic(void) const {
    return magic_;
  }

  //! Retrieve block size of data read
  uint32_t read_block_size(void) const {
    return read_block_size_;
  }

 private:
  //! Constants
  static constexpr uint32_t kDefaultBlockVecCount = 32u;
  static constexpr uint32_t kDefaultSegmentSize = 4 * 1024 * 1024u;
  static constexpr float kDefaultDocsSoftLimitRatio = 0.9f;

  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_OPENED = 2 };

  //! Members
  uint32_t magic_{0};
  uint32_t docs_hard_limit_{std::numeric_limits<uint32_t>::max()};
  uint32_t docs_soft_limit_{0};
  IndexMeta meta_{};
  std::vector<std::vector<std::string>> data_;
  IndexStreamer::Stats stats_{};
  IndexMetric::Pointer metric_{};
  State state_{STATE_INIT};
  mutable std::mutex mapping_mutex_{};
  ailego::SharedMutex dump_mutex_{};
  FlatStreamerEntity entity_;
  bool column_major_order_{false};
  bool use_key_info_map_{true};
  uint32_t read_block_size_{0};
};

}  // namespace core
}  // namespace zvec
