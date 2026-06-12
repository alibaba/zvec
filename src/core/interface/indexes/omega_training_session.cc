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

#include "omega_training_session.h"
#include "algorithm/omega/omega_streamer.h"

namespace zvec::core_interface {

zvec::Status OmegaTrainingSession::Start(const TrainingSessionConfig &config) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (streamer_ == nullptr) {
    return zvec::Status::InvalidArgument("Omega streamer is not available");
  }

  ResetArtifactsLocked();
  topk_ = config.topk;
  num_queries_ = config.ground_truth.size();
  streamer_->set_training_ground_truth(config.ground_truth, config.k_train);
  streamer_->enable_training_mode(true);
  active_ = true;
  return zvec::Status::OK();
}

void OmegaTrainingSession::BeginQuery(int query_id) {
  if (streamer_ != nullptr) {
    streamer_->set_current_query_id(query_id);
  }
}

void OmegaTrainingSession::CollectQueryArtifacts(
    QueryTrainingArtifacts &&artifacts) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!artifacts.records.empty()) {
    records_.insert(records_.end(),
                    std::make_move_iterator(artifacts.records.begin()),
                    std::make_move_iterator(artifacts.records.end()));
  }
  if (!artifacts.gt_cmps_per_rank.empty() && artifacts.training_query_id >= 0) {
    gt_cmps_map_[artifacts.training_query_id] = {
        std::move(artifacts.gt_cmps_per_rank), artifacts.total_cmps};
  }
}

TrainingArtifacts OmegaTrainingSession::ConsumeArtifacts() {
  std::lock_guard<std::mutex> lock(mutex_);

  TrainingArtifacts artifacts;
  artifacts.records = std::move(records_);

  if (!gt_cmps_map_.empty()) {
    size_t num_queries = num_queries_;
    if (num_queries == 0) {
      num_queries = static_cast<size_t>(gt_cmps_map_.rbegin()->first) + 1;
    }
    size_t topk = topk_;
    if (topk == 0) {
      for (const auto &entry : gt_cmps_map_) {
        if (!entry.second.first.empty()) {
          topk = entry.second.first.size();
          break;
        }
      }
    }

    artifacts.gt_cmps_data.num_queries = num_queries;
    artifacts.gt_cmps_data.topk = topk;
    artifacts.gt_cmps_data.gt_cmps.resize(num_queries);
    artifacts.gt_cmps_data.total_cmps.resize(num_queries, 0);
    for (size_t q = 0; q < num_queries; ++q) {
      artifacts.gt_cmps_data.gt_cmps[q].resize(topk, 0);
    }
    for (const auto &entry : gt_cmps_map_) {
      size_t query_id = static_cast<size_t>(entry.first);
      if (query_id >= num_queries) {
        continue;
      }
      const auto &[gt_cmps_per_rank, total_cmps] = entry.second;
      artifacts.gt_cmps_data.total_cmps[query_id] = total_cmps;
      for (size_t rank = 0; rank < gt_cmps_per_rank.size() && rank < topk;
           ++rank) {
        artifacts.gt_cmps_data.gt_cmps[query_id][rank] = gt_cmps_per_rank[rank];
      }
    }
  }

  gt_cmps_map_.clear();
  return artifacts;
}

void OmegaTrainingSession::Finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_ && streamer_ != nullptr) {
    streamer_->enable_training_mode(false);
  }
  active_ = false;
}

void OmegaTrainingSession::ResetArtifactsLocked() {
  records_.clear();
  gt_cmps_map_.clear();
}

}  // namespace zvec::core_interface
