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

#ifdef ZVEC_ENABLE_OMEGA

#include <algorithm>
#include <thread>
#include <string>
#include <vector>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>

namespace zvec {

inline int DefaultOmegaTrainerThreads() {
  const unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0) {
    return 8;
  }
  return static_cast<int>(std::max(1u, hc / 2));
}

/**
 * @brief Configuration options for OMEGA model training
 */
struct OmegaModelTrainerOptions {
  // Output directory for trained model files
  std::string output_dir;

  // LightGBM training parameters
  int num_iterations = 100;
  int num_leaves = 31;
  double learning_rate = 0.1;
  int num_threads = DefaultOmegaTrainerThreads();
  int seed = 42;
  bool deterministic = true;

  // Enable verbose logging during training
  bool verbose = false;
};

/**
 * @brief OMEGA model trainer using LightGBM C API
 *
 * This class trains a LightGBM binary classifier directly in C++,
 * eliminating the need for Python subprocess and CSV serialization.
 */
class OmegaModelTrainer {
 public:
  /**
   * @brief Train OMEGA model with gt_cmps data for table generation
   *
   * This is the extended version that also generates gt_collected_table
   * and gt_cmps_all_table from gt_cmps data.
   *
   * @param training_records Training data collected from searches
   * @param gt_cmps_data Ground truth cmps data for table generation
   * @param options Training configuration
   * @return Status indicating success or failure
   */
  static Status TrainModelWithGtCmps(
      const std::vector<core_interface::TrainingRecord>& training_records,
      const core_interface::GtCmpsData& gt_cmps_data,
      const OmegaModelTrainerOptions& options);
};

}  // namespace zvec

#endif  // ZVEC_ENABLE_OMEGA
