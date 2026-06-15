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

#include "diskann_reducer.h"
#include <vector>
#include <ailego/parallel/lock.h>
#include <ailego/parallel/multi_thread_list.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/framework/index_reducer.h>
#include "diskann_params.h"

namespace zvec {
namespace core {

int DiskAnnReducer::init(const ailego::Params &params) {
  params.get(PARAM_DISKANN_REDUCER_WORKING_PATH, &working_path_);
  if (working_path_.empty()) {
    LOG_ERROR("Missing parameter. %s",
              PARAM_DISKANN_REDUCER_WORKING_PATH.c_str());
    return IndexError_InvalidArgument;
  }

  std::string index_name =
      params.get_as_string(PARAM_DISKANN_REDUCER_INDEX_NAME);
  if (index_name.empty()) {
    index_name = std::to_string(std::clock());
  }

  reducer_file_path_ = ailego::StringHelper::Concat(
      working_path_, "/", kReducerFileName, index_name);

  holder_file_path_ = ailego::StringHelper::Concat(working_path_, "/",
                                                   kHolderFileName, index_name);

  state_ = STATE_INITED;
  return 0;
}

int DiskAnnReducer::cleanup(void) {
  return 0;
}

//! Reduce operator with filter
int DiskAnnReducer::reduce(const IndexFilter &filter) {
  if (entities_.empty() || state_ != STATE_FEED) {
    LOG_ERROR("No container to reduce, feed first");
    return IndexError_NoReady;
  }

  if (use_mem_holder_) {
    mem_holder_ = std::make_shared<RandomAccessIndexHolder>(meta_);
    for (auto entity : entities_) {
      size_t doc_cnt = entity->doc_cnt();
      for (size_t id = 0; id < doc_cnt; ++id) {
        diskann_key_t pkey = entity->get_key(id);

        if (filter.is_valid() && filter(pkey)) {
          continue;
        }

        const void *vec = entity->get_vector(id);
        mem_holder_->emplace(pkey, vec);
      }
    }
  } else {
    disk_holder_ =
        std::make_shared<DiskAnnIndexHolder>(meta_, holder_file_path_);

    int ret = disk_holder_->init();
    if (ret != 0) {
      LOG_ERROR("DiskAnn Index Holder init failed");
      return ret;
    }

    for (auto entity : entities_) {
      size_t doc_cnt = entity->doc_cnt();
      for (size_t id = 0; id < doc_cnt; ++id) {
        diskann_key_t pkey = entity->get_key(id);

        if (filter.is_valid() && filter(pkey)) {
          continue;
        }

        const void *vec = entity->get_vector(id);
        disk_holder_->emplace(pkey, vec);
      }
    }

    disk_holder_->close();
  }

  builder_ = IndexFactory::CreateBuilder(kDiskAnnBuilderName);
  if (!builder_) {
    LOG_ERROR("Create builder failed. name[%s]", kDiskAnnBuilderName.c_str());
    return IndexError_Runtime;
  }

  if (thread_pool_ == nullptr) {
    LOG_ERROR(
        "Only support multi-thread mode. Thread pool is not set for reducer.");
    return IndexError_NoReady;
  }

  LOG_INFO("Start diskann reduce");

  ailego::ElapsedTime timer;

  auto params = meta_.builder_params();

  int ret = builder_->init(meta_, params);
  if (ret != 0) {
    LOG_ERROR("Init proxima streamer failed. ret[%d]", ret);
    return ret;
  }

  if (use_mem_holder_) {
    ret = builder_->train(mem_holder_);
    if (ret != 0) {
      LOG_ERROR("Diskann builder failed to train. ret[%d]", ret);
      return ret;
    }

    ret = builder_->build(mem_holder_);
    if (ret != 0) {
      LOG_ERROR("Diskann builder failed to build. ret[%d]", ret);
      return ret;
    }
  } else {
    ret = builder_->train(disk_holder_);
    if (ret != 0) {
      LOG_ERROR("Diskann builder failed to train. ret[%d]", ret);
      return ret;
    }

    ret = builder_->build(disk_holder_);
    if (ret != 0) {
      LOG_ERROR("Diskann builder failed to build. ret[%d]", ret);
      return ret;
    }
  }

  auto &stats = builder_->stats();

  stats_.set_reduced_costtime(timer.seconds());
  stats_.set_filtered_count(stats.discarded_count());

  state_ = STATE_REDUCE;

  LOG_INFO("End DiskAnn reduce. cost time: [%zu]s", (size_t)timer.seconds());
  return 0;
}

//! Dump index by dumper
int DiskAnnReducer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Begin diskann reducer dump");

  if (state_ != STATE_REDUCE) {
    LOG_WARN("Reduce first before dump.");
    return IndexError_NoReady;
  }

  ailego::ElapsedTime timer;
  int ret = builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("diskann reducer dump failed. ret[%d]", ret);
    return ret;
  }
  stats_.set_dumped_costtime(timer.seconds());

  LOG_INFO("End diskann reducer dump, dump costtime=[%zu]s",
           (size_t)(stats_.dumped_costtime()));

  return 0;
}

INDEX_FACTORY_REGISTER_REDUCER(DiskAnnReducer);

}  // namespace core
}  // namespace zvec
