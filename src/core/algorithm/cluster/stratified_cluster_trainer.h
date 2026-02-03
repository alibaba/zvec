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

#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_trainer.h>

namespace zvec {
namespace core {

/*! Cluster Trainer
 */
class StratifiedClusterTrainer : public IndexTrainer {
 public:
  typedef std::shared_ptr<StratifiedClusterTrainer> Pointer;

  //! Constructor
  StratifiedClusterTrainer(void) {}

  //! Destructor
  ~StratifiedClusterTrainer(void) {}

 protected:
  //! Initialize Trainer
  virtual int init(const IndexMeta &meta, const ailego::Params &params);

  //! Cleanup Trainer
  virtual int cleanup(void);

  //! Train the data
  virtual int train(IndexThreads::Pointer threads, IndexHolder::Pointer holder);

  //! Load index from file path or dir
  virtual int load(IndexStorage::Pointer cntr);

  //! Dump index into file path or dir
  virtual int dump(const IndexDumper::Pointer &dumper);

  //! Retrieve Index Meta
  virtual const IndexMeta &meta(void) const;

  //! Retrieve statistics
  virtual const IndexTrainer::Stats &stats(void) const;

  //! Retrieve the output indexes
  virtual IndexBundle::Pointer indexes(void) const;

 private:
  int init_params(const ailego::Params &params);

 private:
  IndexMeta meta_{};
  uint32_t sample_count_{0u};
  float sample_ratio_{0.0};
  uint32_t thread_count_{0u};
  bool cluster_auto_tuning_{false};
  IndexCluster::Pointer cluster_{};
  IndexCluster::CentroidList centroids_{};

  uint32_t suggest_centriod_cnt_{0u};
  std::string class_name_;
  std::vector<std::string> cluster_class_;
  std::vector<uint64_t> centroid_num_vec_;
  std::vector<ailego::Params> cluster_params_;
  IndexTrainer::Stats stats_{};

 private:
  static const std::string SEP_TOKEN;
  static const std::string DEFAULT_CLUSTER_CLASS;
};

}  // namespace core
}  // namespace zvec
