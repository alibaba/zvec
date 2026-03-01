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

#include <string>
#include <vector>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>

namespace zvec {

/**
 * @brief Configuration options for OMEGA model training
 */
struct OmegaModelTrainerOptions {
  // Output directory for trained model files
  std::string output_dir;

  // Path to Python executable (default: python3)
  std::string python_executable = "python3";

  // Enable verbose logging during training
  bool verbose = false;
};

/**
 * @brief OMEGA model trainer (calls Python training script)
 *
 * This class bridges C++ training data collection with Python model training.
 * It exports TrainingRecord data to CSV format and invokes the Python
 * _omega_training.py script to train a LightGBM model.
 */
class OmegaModelTrainer {
 public:
  /**
   * @brief Train OMEGA model from collected training records
   *
   * @param training_records Training data collected from searches
   * @param options Training configuration
   * @return Status indicating success or failure
   */
  static Status TrainModel(
      const std::vector<core_interface::TrainingRecord>& training_records,
      const OmegaModelTrainerOptions& options);

 private:
  /**
   * @brief Export training records to CSV format
   *
   * CSV format:
   * query_id,hops_visited,cmps_visited,dist_1st,dist_start,
   * stat_0,stat_1,stat_2,stat_3,stat_4,stat_5,stat_6,label
   *
   * @param records Training records to export
   * @param csv_path Output CSV file path
   * @return Status indicating success or failure
   */
  static Status ExportToCSV(
      const std::vector<core_interface::TrainingRecord>& records,
      const std::string& csv_path);

  /**
   * @brief Invoke Python training script
   *
   * Calls: python3 -m zvec._omega_training train \
   *          --input <csv_path> --output <output_dir> [--verbose]
   *
   * @param csv_path Input CSV file path
   * @param options Training configuration
   * @return Status indicating success or failure
   */
  static Status InvokePythonTrainer(
      const std::string& csv_path,
      const OmegaModelTrainerOptions& options);
};

}  // namespace zvec
