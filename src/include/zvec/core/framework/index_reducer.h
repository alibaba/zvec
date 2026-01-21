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

#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include <zvec/core/framework/index_streamer.h>

namespace zvec {
namespace core {

/*! Index Reducer Base
 */
class IndexReducerBase : public IndexModule {
 public:
  //! Index Reducer Pointer
  typedef std::shared_ptr<IndexReducerBase> Pointer;

  /*! Index Reducer Stats
   */
  class Stats : public IndexStats {
   public:
    Stats() {}
    Stats(const Stats &stats) {
      *this = stats;
    }
    Stats &operator=(const Stats &stats) {
      this->loaded_count_.store(stats.loaded_count_.load());
      this->dumped_count_.store(stats.dumped_count_.load());
      this->filtered_count_.store(stats.filtered_count_.load());
      this->duplicated_count_.store(stats.duplicated_count_.load());
      this->reduced_costtime_.store(stats.reduced_costtime_.load());
      this->dumped_costtime_.store(stats.dumped_costtime_.load());
      return *this;
    }
    //! Set count of documents loaded
    void set_loaded_count(size_t count) {
      loaded_count_ = count;
    }

    //! Set count of documents dumped
    void set_dumped_count(size_t count) {
      dumped_count_ = count;
    }

    //! Set count of documents filtered
    void set_filtered_count(size_t count) {
      filtered_count_ = count;
    }

    //! Set count of documents duplicated
    void set_duplicated_count(size_t count) {
      duplicated_count_ = count;
    }

    //! Set time cost of documents reduced
    void set_reduced_costtime(uint64_t cost) {
      reduced_costtime_ = cost;
    }

    //! Set time cost of documents dumped
    void set_dumped_costtime(uint64_t cost) {
      dumped_costtime_ = cost;
    }

    //! Retrieve count of documents loaded
    size_t loaded_count(void) const {
      return loaded_count_;
    }

    //! Retrieve count of documents dumped
    size_t dumped_count(void) const {
      return dumped_count_;
    }

    //! Retrieve count of documents filtered
    size_t filtered_count(void) const {
      return filtered_count_;
    }

    //! Retrieve count of documents duplicated
    size_t duplicated_count(void) const {
      return duplicated_count_;
    }

    //! Retrieve time cost of documents reduced
    uint64_t reduced_costtime(void) const {
      return reduced_costtime_;
    }

    //! Retrieve time cost of documents dumped
    uint64_t dumped_costtime(void) const {
      return dumped_costtime_;
    }

    //! Retrieve count of documents loaded (mutable)
    std::atomic<size_t> *mutable_loaded_count(void) {
      return &loaded_count_;
    }

    //! Retrieve count of documents dumped (mutable)
    std::atomic<size_t> *mutable_dumped_count(void) {
      return &dumped_count_;
    }

    //! Retrieve count of documents filtered (mutable)
    std::atomic<size_t> *mutable_filtered_count(void) {
      return &filtered_count_;
    }

    //! Retrieve count of documents duplicated (mutable)
    std::atomic<size_t> *mutable_duplicated_count(void) {
      return &duplicated_count_;
    }

    //! Retrieve time cost of documents reduced (mutable)
    std::atomic<uint64_t> *mutable_reduced_costtime(void) {
      return &reduced_costtime_;
    }

    //! Retrieve time cost of documents dumped (mutable)
    std::atomic<uint64_t> *mutable_dumped_costtime(void) {
      return &dumped_costtime_;
    }

   private:
    //! Members
    std::atomic<size_t> loaded_count_{0u};
    std::atomic<size_t> dumped_count_{0u};
    std::atomic<size_t> filtered_count_{0u};
    std::atomic<size_t> duplicated_count_{0u};
    std::atomic<uint64_t> reduced_costtime_{0u};
    std::atomic<uint64_t> dumped_costtime_{0u};
  };

  //! Destructor
  virtual ~IndexReducerBase(void) = default;

  //! Initialize Reducer
  virtual int init(const ailego::Params &params) = 0;

  //! Cleanup Reducer
  virtual int cleanup(void) = 0;

  //! Reduce operator (with filter)
  virtual int reduce(const IndexFilter &filter) = 0;

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer &dumper) = 0;

  //! Retrieve statistics
  virtual const Stats &stats(void) const = 0;

  //! Set thread pool
  void set_thread_pool(ailego::ThreadPool *pool) {
    thread_pool_ = pool;
  }

  //! Set stop flag
  void set_stop_flag(std::atomic<bool> *stop_flag) {
    stop_flag_ = stop_flag;
  }

 protected:
  ailego::ThreadPool *thread_pool_{nullptr};
  std::atomic<bool> *stop_flag_{nullptr};
};

/*! Index Reducer
 */
class IndexReducer : public IndexReducerBase {
 public:
  //! Index Reducer Pointer
  typedef std::shared_ptr<IndexReducer> Pointer;

  //! Destructor
  virtual ~IndexReducer(void) = default;
};

/*! Index Sparse Reducer
 */
class IndexSparseReducer : public IndexReducerBase {
 public:
  //! Index Reducer Pointer
  typedef std::shared_ptr<IndexSparseReducer> Pointer;

  //! Destructor
  virtual ~IndexSparseReducer(void) = default;
};

/*! Index Streamer Reducer
 */
class IndexStreamerReducer : public IndexReducerBase {
 public:
  //! Index Reducer Pointer
  typedef std::shared_ptr<IndexStreamerReducer> Pointer;

  virtual int set_target_streamer_wiht_info(
      const IndexBuilder::Pointer builder,
      const IndexStreamer::Pointer streamer,
      const IndexConverter::Pointer converter,
      const IndexReformer::Pointer reformer = nullptr,
      const IndexQueryMeta &original_query_meta = IndexQueryMeta()) = 0;
  virtual int feed_streamer_with_reformer(
      IndexStreamer::Pointer streamer,
      const IndexReformer::Pointer reformer) = 0;

  virtual ~IndexStreamerReducer(void) = default;
};
}  // namespace core
}  // namespace zvec
