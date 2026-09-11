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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_streamer.h>
#include "utility/ordinal_access_holder.h"

namespace zvec {
namespace core {

/*! A multi-pass holder that presents multiple source providers as one dense
 *  sequence without materializing all source vectors.
 */
class MergedProviderIndexHolder final : public IndexHolder,
                                        public OrdinalAccessHolder {
 public:
  typedef std::shared_ptr<MergedProviderIndexHolder> Pointer;

  struct Source {
    // Keep the streamer alive for as long as a builder may retain this holder.
    // Providers are created lazily so the holder does not retain one provider
    // (and its source-specific buffers) for every source at the same time.
    IndexStreamer::Pointer owner{};
    IndexReformer::Pointer reformer{};
    IndexQueryMeta provider_meta{};
    bool need_revert{false};

    // Filled by init() and consumed by each merged iterator.
    uint64_t logical_id_base{0};
    size_t provider_count{0};
    size_t iterated_count{0};
    std::vector<uint64_t> keep_bits{};
  };

  MergedProviderIndexHolder(IndexQueryMeta output_meta,
                            std::vector<Source> sources);

  //! Scan source keys once, cache the filter decisions and calculate count.
  int init(const IndexFilter &filter, std::atomic<bool> *stop_flag = nullptr);

  size_t count(void) const override;
  size_t dimension(void) const override;
  IndexMeta::DataType data_type(void) const override;
  size_t element_size(void) const override;
  bool multipass(void) const override;
  IndexHolder::Iterator::Pointer create_iterator(void) override;
  int create_ordinal_reader(
      OrdinalAccessHolder::Reader::Pointer *reader) override;

  size_t filtered_count(void) const;
  int status(void) const;

  // The reducer clears this after synchronous train/build finishes because a
  // provider-backed IVF builder may retain the holder until a later dump call.
  void set_stop_flag(std::atomic<bool> *stop_flag);

 private:
  class Iterator;
  class OrdinalReader;

  IndexProvider::Pointer acquire_provider(size_t source_index,
                                          bool validate_planned_count);
  bool keep(size_t source_index, size_t ordinal) const;
  bool canceled(void) const;
  void set_status(int status);

  IndexQueryMeta output_meta_{};
  std::vector<Source> sources_{};
  size_t count_{0};
  size_t filtered_count_{0};
  bool has_filter_{false};
  bool initialized_{false};
  std::atomic<int> status_{0};
  std::atomic<bool> *stop_flag_{nullptr};
};

}  // namespace core
}  // namespace zvec
