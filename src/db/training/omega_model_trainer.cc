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

#ifdef ZVEC_ENABLE_OMEGA

#include "omega_model_trainer.h"
#include <omega/omega_trainer.h>
#include <chrono>
#include <zvec/ailego/logger/logger.h>

namespace zvec {

namespace {

// Convert zvec TrainingRecord to omega TrainingRecord
omega::TrainingRecord ConvertRecord(const core_interface::TrainingRecord& src) {
  omega::TrainingRecord dst;
  dst.query_id = src.query_id;
  dst.hops_visited = src.hops_visited;
  dst.cmps_visited = src.cmps_visited;
  dst.dist_1st = src.dist_1st;
  dst.dist_start = src.dist_start;
  // Convert std::array<float, 7> to std::vector<float>
  dst.traversal_window_stats.assign(src.traversal_window_stats.begin(),
                                     src.traversal_window_stats.end());
  dst.label = src.label;
  dst.collected_node_ids = src.collected_node_ids;
  return dst;
}

// Convert zvec GtCmpsData to omega GtCmpsData
omega::GtCmpsData ConvertGtCmpsData(const core_interface::GtCmpsData& src) {
  omega::GtCmpsData dst;
  dst.num_queries = src.num_queries;
  dst.topk = src.topk;
  dst.gt_cmps = src.gt_cmps;
  dst.total_cmps = src.total_cmps;
  return dst;
}

}  // namespace

Status OmegaModelTrainer::TrainModel(
    const std::vector<core_interface::TrainingRecord>& training_records,
    const OmegaModelTrainerOptions& options) {
  // Call TrainModelWithGtCmps with empty gt_cmps_data
  core_interface::GtCmpsData empty_gt_cmps;
  return TrainModelWithGtCmps(training_records, empty_gt_cmps, options);
}

Status OmegaModelTrainer::TrainModelWithGtCmps(
    const std::vector<core_interface::TrainingRecord>& training_records,
    const core_interface::GtCmpsData& gt_cmps_data,
    const OmegaModelTrainerOptions& options) {
  if (training_records.empty()) {
    return Status::InvalidArgument("Training records are empty");
  }

  if (options.output_dir.empty()) {
    return Status::InvalidArgument("Output directory is empty");
  }

  auto total_start = std::chrono::high_resolution_clock::now();

  LOG_INFO("Training OMEGA model using C++ LightGBM API (%zu records)",
           training_records.size());

  // Convert training records
  std::vector<omega::TrainingRecord> omega_records;
  omega_records.reserve(training_records.size());
  for (const auto& r : training_records) {
    omega_records.push_back(ConvertRecord(r));
  }

  // Convert gt_cmps data
  omega::GtCmpsData omega_gt_cmps = ConvertGtCmpsData(gt_cmps_data);

  // Setup trainer options
  omega::OmegaTrainerOptions trainer_options;
  trainer_options.output_dir = options.output_dir;
  trainer_options.num_iterations = options.num_iterations;
  trainer_options.num_leaves = options.num_leaves;
  trainer_options.learning_rate = options.learning_rate;
  trainer_options.num_threads = options.num_threads;
  trainer_options.verbose = options.verbose;
  trainer_options.topk = gt_cmps_data.topk > 0 ? gt_cmps_data.topk : 100;

  // Train model
  int ret = omega::OmegaTrainer::TrainModel(omega_records, omega_gt_cmps, trainer_options);

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

  if (ret != 0) {
    LOG_ERROR("OMEGA model training failed (return code: %d)", ret);
    return Status::InternalError("OMEGA model training failed");
  }

  LOG_INFO("[TIMING] TrainModelWithGtCmps (C++ LightGBM) TOTAL: %ld ms", total_ms);
  LOG_INFO("Successfully trained OMEGA model, output: %s", options.output_dir.c_str());

  return Status::OK();
}

}  // namespace zvec

#endif  // ZVEC_ENABLE_OMEGA
