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
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_reducer.h>
#include "diskann_holder.h"
#include "diskann_reducer_entity.h"

namespace zvec {
namespace core {

class DiskAnnReducer : public IndexReducer {
 public:
  //! Constructor
  DiskAnnReducer(void) = default;

 protected:
  //! Initialize Reducer
  virtual int init(const ailego::Params &params) override;

  //! Cleanup Reducer
  virtual int cleanup(void) override;

  //! Feed indexes from containers
  // virtual int feed(IndexStorage::Pointer container) override;

  //! Reduce operator (with filter)
  virtual int reduce(const IndexFilter &filter) override;

  //! Dump index by dumper
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override {
    return stats_;
  }

 private:
  enum State {
    STATE_UNINITED = 0,
    STATE_INITED = 1,
    STATE_FEED = 2,
    STATE_REDUCE = 3
  };

  std::string working_path_{""};

  IndexMeta meta_{};
  std::vector<DiskAnnReducerEntity::Pointer> entities_{};

  // bool use_mem_holder_{true};
  bool use_mem_holder_{false};
  RandomAccessIndexHolder::Pointer mem_holder_;
  DiskAnnIndexHolder::Pointer disk_holder_;

  IndexBuilder::Pointer builder_{nullptr};
  std::string reducer_file_path_{""};
  std::string holder_file_path_{""};

  Stats stats_{};
  State state_{STATE_UNINITED};

  const std::string kDiskAnnBuilderName{"DiskAnnBuilder"};
  const std::string kReducerFileName{"diskann.reducer.builder."};
  const std::string kHolderFileName{"diskann.reducer.holder."};
};

}  // namespace core
}  // namespace zvec
