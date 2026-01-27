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

#include <vector>
#include <ailego/parallel/lock.h>
#include <ailego/parallel/multi_thread_list.h>
#include <utility/sparse_utility.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_reducer.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>

namespace zvec {
namespace core {


class MixedStreamerReducer : public IndexStreamerReducer {
 public:
  //! Constructor
  MixedStreamerReducer(void) {}

  //! Initialize Reducer
  int init(const ailego::Params &params) override;

  //! Cleanup Reducer
  int cleanup(void) override;

  //! Reduce operator (with filter)
  int reduce(const IndexFilter &filter) override;

  //! Dump index by dumper
  int dump(const IndexDumper::Pointer &dumper) override;

 public:  // StreamerReducer's unique methods
  int set_target_streamer_wiht_info(
      const IndexBuilder::Pointer builder,
      const IndexStreamer::Pointer streamer,
      const IndexConverter::Pointer converter,
      const IndexReformer::Pointer reformer,
      const IndexQueryMeta &original_query_meta) override;
  // feed_streamer
  int feed_streamer_with_reformer(
      IndexStreamer::Pointer streamer,
      const IndexReformer::Pointer reformer) override;

 private:
  int read_vec(size_t source_streamer_index, const IndexFilter &filter,
               const uint32_t id_offset, uint32_t *next_id);
  void add_vec(int *result);
  void add_vec_with_builder(int *result);
  int read_sparse_vec(size_t source_streamer_index, const IndexFilter &filter,
                      const uint32_t id_offset, uint32_t *next_id);
  void add_sparse_vec(int *result);

  void PushToDocCache(const IndexQueryMeta &meta, uint32_t doc_id,
                      std::string &doc);
  int IndexBuild();

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

 private:
  enum State {
    STATE_UNINITED,
    STATE_INITED,
    STATE_STREAMER_SET,
    STATE_FEED,
    STATE_REDUCE
  };

  bool enable_pk_rewrite_{false};
  bool is_sparse_{false};

  Stats stats_{};
  State state_{STATE_UNINITED};

  size_t num_of_add_threads_{0};
  ailego::MultiThreadList<VectorItem> mt_list_;
  ailego::MultiThreadList<SparseVectorItem> sparse_mt_list_;


  ailego::Params params_;
  IndexStreamer::Pointer target_streamer_{nullptr};
  IndexReformer::Pointer target_streamer_reformer_{nullptr};
  bool is_target_and_source_same_reformer_{false};
  IndexQueryMeta original_query_meta_{};

  std::vector<IndexStreamer::Pointer> streamers_;
  std::vector<IndexReformer::Pointer> source_streamers_reformers_;

  IndexBuilder::Pointer target_builder_{nullptr};
  IndexConverter::Pointer target_builder_converter_{nullptr};
  std::mutex mutex_{};
  std::vector<std::pair<uint64_t, std::string>> doc_cache_;
  const uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();
};

}  // namespace core
}  // namespace zvec
