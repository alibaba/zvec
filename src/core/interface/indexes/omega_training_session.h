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

#include <map>
#include <mutex>
#include <vector>
#include <zvec/core/interface/training_session.h>

namespace zvec {
namespace core {
class OmegaStreamer;
}  // namespace core
namespace core_interface {

class OmegaTrainingSession : public ITrainingSession {
 public:
  explicit OmegaTrainingSession(core::OmegaStreamer *streamer)
      : streamer_(streamer) {}

  zvec::Status Start(const TrainingSessionConfig &config) override;

  void BeginQuery(int query_id) override;

  void CollectQueryArtifacts(QueryTrainingArtifacts &&artifacts) override;

  TrainingArtifacts ConsumeArtifacts() override;

  void Finish() override;

 private:
  void ResetArtifactsLocked();

  core::OmegaStreamer *streamer_{nullptr};
  std::mutex mutex_;
  size_t topk_{0};
  size_t num_queries_{0};
  bool active_{false};
  std::vector<TrainingRecord> records_;
  std::map<int, std::pair<std::vector<int>, int>> gt_cmps_map_;
};

}  // namespace core_interface
}  // namespace zvec
