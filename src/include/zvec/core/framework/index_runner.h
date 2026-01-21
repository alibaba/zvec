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

#include <atomic>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_stats.h>
#include <zvec/core/framework/index_threads.h>
#include <zvec/core/framework/index_trainer.h>

namespace zvec {
namespace core {

/*! Index Runner
 */
class IndexRunner : public IndexModule {
 public:
  //! Index Searcher Pointer
  typedef std::shared_ptr<IndexRunner> Pointer;

  /*! Index Searcher Context
   */
  using Context = IndexContext;

  /*! Index Searcher Provider
   */
  using Provider = IndexProvider;

  /*! Index Sparse Searcher Provider
   */
  using SparseProvider = IndexSparseProvider;

  /*! Index Streamer Stats
   */
  class Stats : public IndexStats {
   public:
    Stats() {}
    Stats(const Stats &stats) {
      *this = stats;
    }
    Stats &operator=(const Stats &stats) {
      this->revision_id_ = stats.revision_id_;

      this->trained_count_.store(stats.trained_count_.load());
      this->built_count_.store(stats.built_count_.load());
      this->dumped_count_.store(stats.dumped_count_.load());
      this->loaded_count_.store(stats.loaded_count_.load());
      this->added_count_.store(stats.added_count_.load());
      this->discarded_count_.store(stats.discarded_count_.load());
      this->updated_count_.store(stats.updated_count_.load());
      this->deleted_count_.store(stats.deleted_count_.load());

      this->index_size_.store(stats.index_size_.load());
      this->dumped_size_.store(stats.dumped_size_.load());

      this->check_point_.store(stats.check_point_.load());

      this->create_time_.store(stats.create_time_.load());
      this->update_time_.store(stats.update_time_.load());
      this->loaded_costtime_.store(stats.loaded_costtime_.load());
      this->trained_costtime_.store(stats.trained_costtime_.load());
      this->built_costtime_.store(stats.built_costtime_.load());
      this->dumped_costtime_.store(stats.dumped_costtime_.load());

      return *this;
    }
    //! Set revision id
    void set_revision_id(size_t rev) {
      revision_id_ = rev;
    }

    //! Set count of documents trained
    void set_trained_count(size_t count) {
      trained_count_ = count;
    }

    //! Set count of documents built
    void set_built_count(size_t count) {
      built_count_ = count;
    }

    //! Set count of documents dumped
    void set_dumped_count(size_t count) {
      dumped_count_ = count;
    }

    //! Set count of documents loaded
    void set_loaded_count(size_t count) {
      loaded_count_ = count;
    }

    //! Set count of documents added
    void set_added_count(size_t count) {
      added_count_ = count;
    }

    //! Set count of documents discarded
    void set_discarded_count(size_t count) {
      discarded_count_ = count;
    }

    //! Set count of documents updated
    void set_updated_count(size_t count) {
      updated_count_ = count;
    }

    //! Set count of documents deleted
    void set_deleted_count(size_t count) {
      deleted_count_ = count;
    }

    //! Set size of index
    void set_index_size(size_t count) {
      index_size_ = count;
    }

    //! Set size of index dumped
    void set_dumped_size(size_t count) {
      dumped_size_ = count;
    }

    //! Set size of index dumped
    void set_check_point(uint64_t val) {
      check_point_ = val;
    }

    //! Retrieve create time
    void set_create_time(uint64_t val) {
      create_time_ = val;
    }

    //! Retrieve update time
    void set_update_time(uint64_t val) {
      update_time_ = val;
    }

    //! Retrieve loaded costtime
    void set_loaded_costtime(uint64_t val) {
      loaded_costtime_ = val;
    }

    //! Retrieve train costtime
    void set_trained_costtime(uint64_t val) {
      trained_costtime_ = val;
    }

    //! Retrieve built costtime
    void set_built_costtime(uint64_t val) {
      built_costtime_ = val;
    }

    //! Retrieve update time
    void set_dumped_costtime(uint64_t val) {
      dumped_costtime_ = val;
    }

    //! Retrieve revision id
    size_t revision_id(void) const {
      return revision_id_;
    }

    //! Retrieve count of documents trained
    size_t trained_count(void) const {
      return trained_count_;
    }

    //! Retrieve count of documents built
    size_t built_count(void) const {
      return built_count_;
    }

    //! Retrieve count of documents dumped
    size_t dumped_count(void) const {
      return dumped_count_;
    }

    //! Retrieve count of documents loaded
    size_t loaded_count(void) const {
      return loaded_count_;
    }

    //! Retrieve count of documents added
    size_t added_count(void) const {
      return added_count_;
    }

    //! Retrieve count of documents discarded
    size_t discarded_count(void) const {
      return discarded_count_;
    }

    //! Retrieve count of documents updated
    size_t updated_count(void) const {
      return updated_count_;
    }

    //! Retrieve count of documents deleted
    size_t deleted_count(void) const {
      return deleted_count_;
    }

    //! Retrieve size of index
    size_t index_size(void) const {
      return index_size_;
    }

    //! Retrieve size of index dumped
    size_t dumped_size(void) const {
      return dumped_size_;
    }

    //! Retrieve check point of index
    uint64_t check_point(void) const {
      return check_point_;
    }

    //! Retrieve create time of index
    uint64_t create_time(void) const {
      return create_time_;
    }

    //! Retrieve update time of index
    uint64_t update_time(void) const {
      return update_time_;
    }

    //! Retrieve loaded cost time of index
    uint64_t loaded_costtime(void) const {
      return loaded_costtime_;
    }

    //! Retrieve trained cost time of index
    uint64_t trained_costtime(void) const {
      return trained_costtime_;
    }

    //! Retrieve built cost time of index
    uint64_t built_costtime(void) const {
      return built_costtime_;
    }

    //! Retrieve dumped cost time of index
    uint64_t dumped_costtime(void) const {
      return dumped_costtime_;
    }

    //! Retrieve count of documents loaded (mutable)
    std::atomic<size_t> *mutable_trained_count(void) {
      return &loaded_count_;
    }

    //! Retrieve count of documents built (mutable)
    std::atomic<size_t> *mutable_built_count(void) {
      return &built_count_;
    }

    //! Retrieve count of documents dumped (mutable)
    std::atomic<size_t> *mutable_dumped_count(void) {
      return &dumped_count_;
    }

    //! Retrieve count of documents loaded (mutable)
    std::atomic<size_t> *mutable_loaded_count(void) {
      return &loaded_count_;
    }

    //! Retrieve count of documents added (mutable)
    std::atomic<size_t> *mutable_added_count(void) {
      return &added_count_;
    }

    //! Retrieve count of documents discarded (mutable)
    std::atomic<size_t> *mutable_discarded_count(void) {
      return &discarded_count_;
    }

    //! Retrieve count of documents updated (mutable)
    std::atomic<size_t> *mutable_updated_count(void) {
      return &updated_count_;
    }

    //! Retrieve count of documents deleted (mutable)
    std::atomic<size_t> *mutable_deleted_count(void) {
      return &deleted_count_;
    }

    //! Retrieve size of index (mutable)
    std::atomic<size_t> *mutable_index_size(void) {
      return &index_size_;
    }

    //! Retrieve size of index dumped (mutable)
    std::atomic<size_t> *mutable_dumped_size(void) {
      return &dumped_size_;
    }

    //! Retrieve check point of index (mutable)
    std::atomic<uint64_t> *mutable_check_point(void) {
      return &check_point_;
    }

    //! Retrieve create time of index (mutable)
    std::atomic<uint64_t> *mutable_create_time(void) {
      return &create_time_;
    }

    //! Retrieve update time of index (mutable)
    std::atomic<uint64_t> *mutable_update_time(void) {
      return &update_time_;
    }

    //! Retrieve loaded time of index (mutable)
    std::atomic<uint64_t> *mutable_loaded_costtime(void) {
      return &loaded_costtime_;
    }

    //! Retrieve trained costtime of index (mutable)
    std::atomic<uint64_t> *mutable_trained_costtime(void) {
      return &trained_costtime_;
    }

    //! Retrieve built costtime of index (mutable)
    std::atomic<uint64_t> *mutable_built_costtime(void) {
      return &built_costtime_;
    }

    //! Retrieve dump costtime of index (mutable)
    std::atomic<uint64_t> *mutable_dumped_costtime(void) {
      return &dumped_costtime_;
    }

    void clear() {
      this->clear_attributes();

      revision_id_ = 0u;

      trained_count_ = 0u;
      built_count_ = 0u;
      dumped_count_ = 0u;
      loaded_count_ = 0u;
      added_count_ = 0u;
      discarded_count_ = 0u;
      updated_count_ = 0u;
      deleted_count_ = 0u;

      index_size_ = 0u;
      dumped_size_ = 0u;
      check_point_ = 0u;

      create_time_ = 0u;
      update_time_ = 0u;
      loaded_costtime_ = 0u;
      trained_costtime_ = 0u;
      built_costtime_ = 0u;
      dumped_costtime_ = 0u;
    }

   private:
    //! Members
    size_t revision_id_{0u};

    std::atomic<size_t> trained_count_{0u};
    std::atomic<size_t> built_count_{0u};
    std::atomic<size_t> dumped_count_{0u};
    std::atomic<size_t> loaded_count_{0u};
    std::atomic<size_t> added_count_{0u};
    std::atomic<size_t> discarded_count_{0u};
    std::atomic<size_t> updated_count_{0u};
    std::atomic<size_t> deleted_count_{0u};

    std::atomic<size_t> index_size_{0u};
    std::atomic<size_t> dumped_size_{0u};
    std::atomic<uint64_t> check_point_{0u};

    std::atomic<uint64_t> create_time_{0u};
    std::atomic<uint64_t> update_time_{0u};
    std::atomic<uint64_t> loaded_costtime_{0u};
    std::atomic<uint64_t> trained_costtime_{0u};
    std::atomic<uint64_t> built_costtime_{0u};
    std::atomic<uint64_t> dumped_costtime_{0u};
  };

  //! Constructor
  IndexRunner() = default;

  //! Destructor
  virtual ~IndexRunner() = default;

  //! Retrieve statistics
  virtual const Stats &stats(void) const = 0;

  //! Cleanup Searcher
  virtual int cleanup() = 0;

  //! Unload Searcher
  virtual int unload() {
    return IndexError_NotImplemented;
  }

  //! Print debug info
  virtual void print_debug_info() {};

  //! Create a searcher context
  virtual Context::Pointer create_context(void) const {
    return Context::Pointer();
  }

  //! Create a searcher provider
  virtual Provider::Pointer create_provider(void) const {
    return Provider::Pointer();
  }

  //! Create a searcher sparse provider
  virtual SparseProvider::Pointer create_sparse_provider(void) const {
    return SparseProvider::Pointer();
  }

  //! Get vector by key
  virtual const void *get_vector(uint64_t /*key*/) const {
    return nullptr;
  }

  virtual int get_vector(const uint64_t /*key*/,
                         IndexStorage::MemoryBlock & /*block*/) const {
    return IndexError_NotImplemented;
  }

  //! Fetch vector by id
  virtual const void *get_vector_by_id(uint32_t /*id*/) const {
    return nullptr;
  }

  virtual int get_vector_by_id(const uint32_t /*id*/,
                               IndexStorage::MemoryBlock & /*block*/) const {
    return IndexError_NotImplemented;
  }

  virtual int get_vector_by_key(const uint64_t /*key*/,
                                IndexStorage::MemoryBlock & /*block*/) const {
    return IndexError_NotImplemented;
  }

  //! Get vector by key
  virtual int get_sparse_vector(uint64_t /*key*/, uint32_t * /*sparse_count*/,
                                std::string * /*sparse_indices_buffer*/,
                                std::string * /*sparse_values_buffer*/) const {
    return IndexError_NotImplemented;
  }

  //! Fetch vector by id
  virtual int get_sparse_vector_by_id(
      uint32_t /*id*/, uint32_t * /*sparse_count*/,
      std::string * /*sparse_indices_buffer*/,
      std::string * /*sparse_values_buffer*/) const {
    return IndexError_NotImplemented;
  }

  //! Add a vector into index
  virtual int add_impl(uint64_t /*key*/, const void * /*query*/,
                       const IndexQueryMeta & /*qmeta*/,
                       Context::Pointer & /*context*/) {
    return IndexError_NotImplemented;
  }

  //! Add a vector with id into index
  virtual int add_with_id_impl(uint32_t /*id*/, const void * /*query*/,
                               const IndexQueryMeta & /*qmeta*/,
                               Context::Pointer & /*context*/) {
    return IndexError_NotImplemented;
  }

  //! Similarity search
  virtual int search_impl(const void * /*query*/,
                          const IndexQueryMeta & /*qmeta*/,
                          Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }
  //! Similarity search
  virtual int search_impl(const void * /*query*/,
                          const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
                          Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  virtual int search_impl(const uint32_t * /*sparse_count*/,
                          const uint32_t * /*sparse_indices*/,
                          const void * /*sparse_query*/,
                          const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
                          Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Similarity search with sparse inputs
  virtual int search_impl(const uint32_t /*sparse_count*/,
                          const uint32_t * /*sparse_indices*/,
                          const void * /*sparse_query*/,
                          const IndexQueryMeta & /*qmeta*/,
                          Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Similarity brute force search
  virtual int search_bf_impl(const void * /*query*/,
                             const IndexQueryMeta & /*qmeta*/,
                             Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }
  //! Similarity brute force search
  virtual int search_bf_impl(const void * /*query*/,
                             const IndexQueryMeta & /*qmeta*/,
                             uint32_t /*count*/,
                             Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Add a vector into index with dense and sparse inputs
  virtual int add_impl(uint64_t /* pkey */, const uint32_t /* sparse_count*/,
                       const uint32_t * /* sparse_indices */,
                       const void * /* sparse_query */,
                       const IndexQueryMeta & /* qmeta */,
                       Context::Pointer & /* context */) {
    return IndexError_NotImplemented;
  }

  //! Add a vector with id into index
  virtual int add_with_id_impl(uint32_t /* id */,
                               const uint32_t /* sparse_count*/,
                               const uint32_t * /* sparse_indices */,
                               const void * /* sparse_query */,
                               const IndexQueryMeta & /* qmeta */,
                               Context::Pointer & /* context */) {
    return IndexError_NotImplemented;
  }

  //! Bruteforce search with sparse inputs
  virtual int search_bf_impl(const uint32_t * /*sparse_count*/,
                             const uint32_t * /*sparse_indices*/,
                             const void * /*sparse_query*/,
                             const IndexQueryMeta & /*qmeta*/,
                             uint32_t /*count*/,
                             Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Bruteforce search with sparse inputs
  virtual int search_bf_impl(const uint32_t /*sparse_count*/,
                             const uint32_t * /*sparse_indices*/,
                             const void * /*sparse_query*/,
                             const IndexQueryMeta & /*qmeta*/,
                             Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Similarity brute force search by primary keys
  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, Context::Pointer &context) const {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Similarity brute force search by primary keys
  virtual int search_bf_by_p_keys_impl(
      const void * /*query*/,
      const std::vector<std::vector<uint64_t>> & /*p_keys*/,
      const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
      Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Linear search by primary keys with dense and sparse inputs
  virtual int search_bf_by_p_keys_impl(
      const uint32_t /* sparse_count */, const uint32_t * /* sparse_indices */,
      const void * /* sparse_query */,
      const std::vector<std::vector<uint64_t>> & /* p_keys */,
      const IndexQueryMeta & /* qmeta */,
      Context::Pointer & /* context */) const {
    return IndexError_NotImplemented;
  }

  //! Linear search by primary keys with dense and sparse inputs
  virtual int search_bf_by_p_keys_impl(
      const uint32_t * /* sparse_count */,
      const uint32_t * /* sparse_indices */, const void * /* sparse_query */,
      const std::vector<std::vector<uint64_t>> & /*p_keys */,
      const IndexQueryMeta & /* qmeta */, uint32_t /* count */,
      Context::Pointer & /* context */) const {
    return IndexError_NotImplemented;
  }

  //! Linear search by primary keys with dense and sparse inputs
  virtual int search_bf_by_p_keys_impl(
      const void * /*dense_query*/, const uint32_t /*sparse_count*/,
      const uint32_t * /*sparse_indices*/, const void * /*sparse_query*/,
      const std::vector<std::vector<uint64_t>> & /*p_keys*/,
      const IndexQueryMeta & /*qmeta*/, Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Linear search by primary keys with dense and sparse inputs
  virtual int search_bf_by_p_keys_impl(
      const void * /*dense_query*/, const uint32_t * /*sparse_count*/,
      const uint32_t * /*sparse_indices*/, const void * /*sparse_query*/,
      const std::vector<std::vector<uint64_t>> & /*p_keys*/,
      const IndexQueryMeta & /*qmeta*/, uint32_t /*count*/,
      Context::Pointer & /*context*/) const {
    return IndexError_NotImplemented;
  }

  //! Update the vector in index
  virtual int update_impl(uint64_t /*key*/, const void * /*query*/,
                          const IndexQueryMeta & /*qmeta*/,
                          Context::Pointer & /*context*/) {
    return IndexError_NotImplemented;
  }

  //! Delete the vector in index
  virtual int remove_impl(uint64_t /*key*/, Context::Pointer & /*context*/) {
    return IndexError_NotImplemented;
  }

  //! Optimize the index
  virtual int optimize_impl(IndexThreads::Pointer) {
    return IndexError_NotImplemented;
  }

  //! Delete the vector in index
  int remove(uint64_t key, Context::Pointer &context) {
    return this->remove_impl(key, context);
  }

  //! Optimize the index
  int optimize(IndexThreads::Pointer threads) {
    return this->optimize_impl(threads);
  }

  //! Train the data
  virtual int train(IndexHolder::Pointer holder) {
    return this->train(nullptr, std::move(holder));
  }

  //! Train the data
  virtual int train(IndexThreads::Pointer /*threads*/,
                    IndexHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Train the data
  virtual int train(const IndexTrainer::Pointer & /*trainer*/) {
    return IndexError_NotImplemented;
  }

  //! Train the data
  virtual int train(IndexSparseHolder::Pointer holder) {
    return this->train(nullptr, std::move(holder));
  }

  //! Train the data
  virtual int train(IndexThreads::Pointer /*threads*/,
                    IndexSparseHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Build the index
  virtual int build(IndexHolder::Pointer holder) {
    return this->build(nullptr, std::move(holder));
  }

  //! Build the index
  virtual int build(IndexThreads::Pointer /*threads*/,
                    IndexHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Build the index
  virtual int build(IndexSparseHolder::Pointer holder) {
    return this->build(nullptr, std::move(holder));
  }

  //! Build the index
  virtual int build(IndexThreads::Pointer /*threads*/,
                    IndexSparseHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Build the index with indptr format
  virtual int build(size_t count, const uint64_t *keys,
                    const uint64_t *sparse_indptr,
                    const uint32_t *sparse_indices, const void *sparse_data) {
    return this->build(nullptr, count, keys, sparse_indptr, sparse_indices,
                       sparse_data);
  }

  virtual int build(const IndexQueryMeta &qmeta, size_t count,
                    const uint64_t *keys, const uint64_t *sparse_indptr,
                    const uint32_t *sparse_indices, const void *sparse_data) {
    return this->build(nullptr, qmeta, count, keys, sparse_indptr,
                       sparse_indices, sparse_data);
  }

  //! Build the index with indptr format
  virtual int build(IndexThreads::Pointer /*threads*/, size_t /*count*/,
                    const uint64_t * /*keys*/,
                    const uint64_t * /*sparse_indptr*/,
                    const uint32_t * /*sparse_indices*/,
                    const void * /*sparse_data*/) {
    return IndexError_NotImplemented;
  }

  //! Build the index with indptr format
  virtual int build(IndexThreads::Pointer /*threads*/,
                    const IndexQueryMeta & /*qmeta*/, size_t /*count*/,
                    const uint64_t * /*keys*/,
                    const uint64_t * /*sparse_indptr*/,
                    const uint32_t * /*sparse_indices*/,
                    const void * /*sparse_data*/) {
    return IndexError_NotImplemented;
  }

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer & /*dumper*/) {
    return IndexError_NotImplemented;
  }
};

}  // namespace core
}  // namespace zvec
