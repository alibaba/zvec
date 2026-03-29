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

#include <cstdint>
#include <memory>
#include <vector>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>

namespace zvec::core_interface {

struct TrainingSessionConfig {
  std::vector<std::vector<uint64_t>> ground_truth;
  size_t topk = 0;
  int k_train = 1;
};

struct QueryTrainingArtifacts {
  std::vector<TrainingRecord> records;
  std::vector<int> gt_cmps_per_rank;
  int total_cmps = 0;
  int training_query_id = -1;
};

struct TrainingArtifacts {
  std::vector<TrainingRecord> records;
  GtCmpsData gt_cmps_data;
};

class ITrainingSession {
 public:
  using Pointer = std::shared_ptr<ITrainingSession>;

  virtual ~ITrainingSession() = default;

  virtual zvec::Status Start(const TrainingSessionConfig& config) = 0;

  virtual void BeginQuery(int query_id) = 0;

  virtual void CollectQueryArtifacts(QueryTrainingArtifacts&& artifacts) = 0;

  virtual TrainingArtifacts ConsumeArtifacts() = 0;

  virtual void Finish() = 0;
};

}  // namespace zvec::core_interface
