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

#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_helper.h>
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Flat Builder
 */
template <size_t BATCH_SIZE>
class FlatBuilder : public IndexBuilder {
 public:
  //! Destructor
  virtual ~FlatBuilder(void) {}

  //! Initialize the builder
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup the builder
  int cleanup(void) override {
    holder_ = nullptr;
    return 0;
  }

  //! Train the data
  int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    stats_.set_trained_count(0u);
    stats_.set_trained_costtime(0u);
    return 0;
  }

  //! Train the data
  int train(const IndexTrainer::Pointer &) override {
    stats_.set_trained_count(0u);
    stats_.set_trained_costtime(0u);
    return 0;
  }

  //! Build the index
  int build(IndexThreads::Pointer, IndexHolder::Pointer holder) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const IndexBuilder::Stats &stats(void) const override {
    return stats_;
  }

 protected:
  //! Dump index keys
  int write_keys(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  //! Dump index keys mapping
  int write_mapping(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  //! Dump index using column-major-order format
  template <typename T>
  int write_column_index(IndexDumper *dumper, std::vector<uint64_t> *keys);

  //! Dump index using column-major-order format
  int write_column_index(IndexDumper *dumper, std::vector<uint64_t> *keys) {
    switch (IndexMeta::AlignSizeof(meta_.data_type())) {
      case 2:
        return this->write_column_index<uint16_t>(dumper, keys);
      case 4:
        return this->write_column_index<uint32_t>(dumper, keys);
      case 8:
        return this->write_column_index<uint64_t>(dumper, keys);
      default:
        ailego_check_with(0, "BAD CASE");
    }
    return IndexError_Runtime;
  }

  //! Dump index using row-major-order format
  int write_row_index(IndexDumper *dumper, std::vector<uint64_t> *keys);

 private:
  IndexMeta meta_{};
  IndexBuilder::Stats stats_{};
  IndexHolder::Pointer holder_{};
};

}  // namespace core
}  // namespace zvec
