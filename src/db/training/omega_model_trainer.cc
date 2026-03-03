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

#include "omega_model_trainer.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <zvec/ailego/logger/logger.h>

namespace zvec {

Status OmegaModelTrainer::TrainModel(
    const std::vector<core_interface::TrainingRecord>& training_records,
    const OmegaModelTrainerOptions& options) {
  if (training_records.empty()) {
    return Status::InvalidArgument("Training records are empty");
  }

  if (options.output_dir.empty()) {
    return Status::InvalidArgument("Output directory is empty");
  }

  // Step 1: Export training records to CSV
  std::string csv_path = options.output_dir + "/training_data.csv";
  LOG_INFO("Exporting %zu training records to CSV: %s",
           training_records.size(), csv_path.c_str());

  auto status = ExportToCSV(training_records, csv_path);
  if (!status.ok()) {
    return status;
  }

  // Step 2: Invoke Python training script
  LOG_INFO("Invoking Python training script");
  status = InvokePythonTrainer(csv_path, options);
  if (!status.ok()) {
    return status;
  }

  LOG_INFO("Successfully trained OMEGA model, output: %s",
           options.output_dir.c_str());
  return Status::OK();
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

  // Step 1: Export training records to CSV
  std::string csv_path = options.output_dir + "/training_data.csv";
  LOG_INFO("Exporting %zu training records to CSV: %s",
           training_records.size(), csv_path.c_str());

  auto status = ExportToCSV(training_records, csv_path);
  if (!status.ok()) {
    return status;
  }

  // Step 2: Export gt_cmps data to CSV
  std::string gt_cmps_path = options.output_dir + "/gt_cmps.csv";
  LOG_INFO("Exporting gt_cmps data to CSV: %s", gt_cmps_path.c_str());

  status = ExportGtCmpsToCSV(gt_cmps_data, gt_cmps_path);
  if (!status.ok()) {
    return status;
  }

  // Step 3: Invoke Python training script with gt_cmps
  LOG_INFO("Invoking Python training script with gt_cmps");
  status = InvokePythonTrainer(csv_path, options, gt_cmps_path);
  if (!status.ok()) {
    return status;
  }

  LOG_INFO("Successfully trained OMEGA model with gt_cmps, output: %s",
           options.output_dir.c_str());
  return Status::OK();
}

Status OmegaModelTrainer::ExportToCSV(
    const std::vector<core_interface::TrainingRecord>& records,
    const std::string& csv_path) {
  std::ofstream csv_file(csv_path);
  if (!csv_file.is_open()) {
    return Status::InternalError("Failed to open CSV file for writing: " + csv_path);
  }

  // Write CSV header
  csv_file << "query_id,hops_visited,cmps_visited,dist_1st,dist_start,"
           << "stat_0,stat_1,stat_2,stat_3,stat_4,stat_5,stat_6,label\n";

  // Write training records
  for (const auto& record : records) {
    csv_file << record.query_id << ","
             << record.hops_visited << ","
             << record.cmps_visited << ","
             << record.dist_1st << ","
             << record.dist_start << ",";

    // Write traversal window stats (7 dimensions)
    for (size_t i = 0; i < record.traversal_window_stats.size(); ++i) {
      csv_file << record.traversal_window_stats[i];
      if (i < record.traversal_window_stats.size() - 1) {
        csv_file << ",";
      }
    }

    csv_file << "," << record.label << "\n";
  }

  csv_file.close();

  if (!csv_file.good()) {
    return Status::InternalError("Error writing CSV file: " + csv_path);
  }

  LOG_INFO("Successfully exported %zu records to CSV", records.size());
  return Status::OK();
}

Status OmegaModelTrainer::ExportGtCmpsToCSV(
    const core_interface::GtCmpsData& gt_cmps_data,
    const std::string& csv_path) {
  std::ofstream csv_file(csv_path);
  if (!csv_file.is_open()) {
    return Status::InternalError("Failed to open gt_cmps CSV file for writing: " + csv_path);
  }

  // Write CSV header
  csv_file << "query_id,rank,cmps\n";

  // Write gt_cmps data
  for (size_t query_id = 0; query_id < gt_cmps_data.gt_cmps.size(); ++query_id) {
    const auto& cmps_per_rank = gt_cmps_data.gt_cmps[query_id];
    for (size_t rank = 0; rank < cmps_per_rank.size(); ++rank) {
      csv_file << query_id << "," << rank << "," << cmps_per_rank[rank] << "\n";
    }
  }

  csv_file.close();

  if (!csv_file.good()) {
    return Status::InternalError("Error writing gt_cmps CSV file: " + csv_path);
  }

  LOG_INFO("Successfully exported gt_cmps for %zu queries to CSV",
           gt_cmps_data.num_queries);
  return Status::OK();
}

Status OmegaModelTrainer::InvokePythonTrainer(
    const std::string& csv_path,
    const OmegaModelTrainerOptions& options,
    const std::string& gt_cmps_path) {
  // Build Python command
  std::ostringstream cmd;
  cmd << options.python_executable
      << " -m zvec._omega_training train"
      << " --input " << csv_path
      << " --output " << options.output_dir;

  // Add gt_cmps path if provided
  if (!gt_cmps_path.empty()) {
    cmd << " --gt_cmps " << gt_cmps_path;
  }

  if (options.verbose) {
    cmd << " --verbose";
  }

  std::string command = cmd.str();
  LOG_INFO("Executing: %s", command.c_str());

  // Execute command
  int ret = std::system(command.c_str());

  if (ret != 0) {
    return Status::InternalError(
        "Python training script failed with exit code: " + std::to_string(ret));
  }

  LOG_INFO("Python training script completed successfully");
  return Status::OK();
}

}  // namespace zvec
