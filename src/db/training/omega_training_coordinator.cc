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

#include "db/training/omega_training_coordinator.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include "db/common/file_helper.h"
#include "db/training/omega_model_trainer.h"

namespace zvec {

namespace {

constexpr uint32_t kOmegaQueryCacheMagic = 0x4F514359;  // OQCY
constexpr uint32_t kOmegaQueryCacheVersion = 1;

}  // namespace

static void WriteOmegaTimingStatsJson(
    const std::string& output_path,
    const std::vector<std::pair<std::string, int64_t>>& stats) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    return;
  }
  ofs << "{\n";
  for (size_t i = 0; i < stats.size(); ++i) {
    ofs << "  \"" << stats[i].first << "\": " << stats[i].second;
    if (i + 1 < stats.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "}\n";
}

static std::string OmegaQueryCachePath(const std::string& model_output_dir) {
  return model_output_dir + "/training_queries.bin";
}

static bool SaveOmegaTrainingQueryCache(
    const std::string& model_output_dir,
    const std::vector<std::vector<float>>& queries,
    const std::vector<uint64_t>& query_doc_ids) {
  if (queries.empty() || queries.size() != query_doc_ids.size()) {
    return false;
  }
  const uint32_t dim = static_cast<uint32_t>(queries[0].size());
  for (const auto& query : queries) {
    if (query.size() != dim) {
      return false;
    }
  }

  std::ofstream ofs(OmegaQueryCachePath(model_output_dir), std::ios::binary);
  if (!ofs.is_open()) {
    return false;
  }

  const uint64_t num_queries = queries.size();
  ofs.write(reinterpret_cast<const char*>(&kOmegaQueryCacheMagic),
            sizeof(kOmegaQueryCacheMagic));
  ofs.write(reinterpret_cast<const char*>(&kOmegaQueryCacheVersion),
            sizeof(kOmegaQueryCacheVersion));
  ofs.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));
  ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  for (size_t i = 0; i < queries.size(); ++i) {
    ofs.write(reinterpret_cast<const char*>(&query_doc_ids[i]),
              sizeof(query_doc_ids[i]));
    ofs.write(reinterpret_cast<const char*>(queries[i].data()),
              static_cast<std::streamsize>(dim * sizeof(float)));
  }
  return ofs.good();
}

static bool LoadOmegaTrainingQueryCache(
    const std::string& model_output_dir,
    std::vector<std::vector<float>>* queries,
    std::vector<uint64_t>* query_doc_ids) {
  std::ifstream ifs(OmegaQueryCachePath(model_output_dir), std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t num_queries = 0;
  uint32_t dim = 0;
  ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
  ifs.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
  ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!ifs.good() || magic != kOmegaQueryCacheMagic ||
      version != kOmegaQueryCacheVersion || num_queries == 0 || dim == 0) {
    return false;
  }

  queries->assign(num_queries, std::vector<float>(dim));
  query_doc_ids->assign(num_queries, 0);
  for (size_t i = 0; i < num_queries; ++i) {
    ifs.read(reinterpret_cast<char*>(&(*query_doc_ids)[i]), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>((*queries)[i].data()),
             static_cast<std::streamsize>(dim * sizeof(float)));
    if (!ifs.good()) {
      queries->clear();
      query_doc_ids->clear();
      return false;
    }
  }
  return true;
}

OmegaTrainingParams ResolveOmegaTrainingParams(
    const IndexParams::Ptr& index_params) {
  OmegaTrainingParams params;
  auto omega_params = std::dynamic_pointer_cast<OmegaIndexParams>(index_params);
  if (!omega_params) {
    return params;
  }

  params.num_training_queries = omega_params->num_training_queries();
  params.ef_training = omega_params->ef_training();
  params.ef_groundtruth = omega_params->ef_groundtruth();
  params.min_vector_threshold = omega_params->min_vector_threshold();
  params.k_train = omega_params->k_train();
  return params;
}

Result<TrainingDataCollectorResult> CollectOmegaTrainingDataBeforeFlush(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const VectorColumnIndexer::Ptr& vector_indexer,
    const OmegaTrainingParams& params,
    const std::string& model_output_dir) {
  TrainingDataCollectorOptions collector_opts;
  const size_t doc_count = vector_indexer->doc_count();
  collector_opts.num_training_queries =
      std::min(doc_count, params.num_training_queries);
  collector_opts.ef_training = params.ef_training;
  collector_opts.ef_groundtruth = params.ef_groundtruth;
  collector_opts.topk = 100;
  collector_opts.k_train = params.k_train;

  std::vector<VectorColumnIndexer::Ptr> training_indexers = {vector_indexer};
  auto training_result = TrainingDataCollector::CollectTrainingDataWithGtCmps(
      segment, field_name, collector_opts, training_indexers);
  if (!training_result.has_value()) {
    return training_result;
  }

  if (!FileHelper::DirectoryExists(model_output_dir)) {
    FileHelper::CreateDirectory(model_output_dir);
  }
  if (!SaveOmegaTrainingQueryCache(model_output_dir,
                                   training_result->training_queries,
                                   training_result->query_doc_ids)) {
    LOG_WARN("Failed to persist OMEGA training query cache: %s",
             OmegaQueryCachePath(model_output_dir).c_str());
  }
  WriteOmegaTimingStatsJson(
      model_output_dir + "/training_collection_timing.json",
      TrainingDataCollector::ConsumeTimingStats());
  return training_result;
}

Result<TrainingDataCollectorResult> CollectOmegaRetrainingData(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const std::vector<VectorColumnIndexer::Ptr>& indexers,
    const OmegaTrainingParams& params,
    const std::string& model_output_dir) {
  TrainingDataCollectorOptions collector_options;
  collector_options.num_training_queries = params.num_training_queries;
  collector_options.ef_training = params.ef_training;
  collector_options.ef_groundtruth = params.ef_groundtruth;
  collector_options.topk = 100;
  collector_options.k_train = params.k_train;

  std::vector<std::vector<float>> cached_queries;
  std::vector<uint64_t> cached_query_doc_ids;
  if (LoadOmegaTrainingQueryCache(model_output_dir, &cached_queries,
                                  &cached_query_doc_ids)) {
    LOG_WARN("Loaded %zu cached held-out queries for OMEGA retraining from %s",
             cached_queries.size(), OmegaQueryCachePath(model_output_dir).c_str());
    return TrainingDataCollector::CollectTrainingDataWithGtCmpsFromQueries(
        segment, field_name, cached_queries, cached_query_doc_ids,
        collector_options, indexers);
  }

  LOG_WARN("OMEGA retrain query cache not found, falling back to sampling held-out queries from persisted segment");
  return TrainingDataCollector::CollectTrainingDataWithGtCmps(
      segment, field_name, collector_options, indexers);
}

Status TrainOmegaModelAfterBuild(
    const TrainingDataCollectorResult& training_result,
    const std::string& model_output_dir) {
  if (training_result.records.size() < 100) {
    LOG_INFO("Skipping model training: only %zu records collected (need >= 100)",
             training_result.records.size());
    return Status::OK();
  }

  OmegaModelTrainerOptions trainer_opts;
  trainer_opts.output_dir = model_output_dir;
  trainer_opts.verbose = true;

  if (!FileHelper::DirectoryExists(model_output_dir) &&
      !FileHelper::CreateDirectory(model_output_dir)) {
    LOG_WARN("Failed to create model output directory: %s",
             model_output_dir.c_str());
  }

  auto train_status = OmegaModelTrainer::TrainModelWithGtCmps(
      training_result.records, training_result.gt_cmps_data, trainer_opts);
  if (train_status.ok()) {
    LOG_INFO("OMEGA model training completed successfully: %s",
             trainer_opts.output_dir.c_str());
  } else {
    LOG_WARN("OMEGA model training failed: %s",
             train_status.message().c_str());
  }

  return Status::OK();
}

Status TrainOmegaModelAfterRetrainCollect(
    const TrainingDataCollectorResult& training_result,
    const std::string& model_output_dir,
    SegmentID segment_id,
    const std::string& field_name) {
  const auto& training_records = training_result.records;
  if (training_records.empty()) {
    LOG_WARN("No training records collected, skipping model training");
    return Status::OK();
  }

  size_t positive_count = 0;
  size_t negative_count = 0;
  for (const auto& record : training_records) {
    if (record.label == 1) {
      positive_count++;
    } else {
      negative_count++;
    }
  }

  if (positive_count == 0 || negative_count == 0) {
    LOG_WARN("Insufficient training samples: %zu positive, %zu negative. Need both > 0. Skipping training.",
             positive_count, negative_count);
    return Status::OK();
  }

  if (positive_count < 50 || negative_count < 50) {
    LOG_WARN("Too few training samples: %zu positive, %zu negative. Need at least 50 of each. Skipping training.",
             positive_count, negative_count);
    return Status::OK();
  }

  LOG_INFO("Training data stats: %zu positive, %zu negative samples",
           positive_count, negative_count);

  LOG_WARN("OMEGA retrain step 2/2: start model training for field '%s' in segment %d",
           field_name.c_str(), segment_id);
  OmegaModelTrainerOptions trainer_options;
  trainer_options.output_dir = model_output_dir;
  trainer_options.verbose = true;

  if (!FileHelper::DirectoryExists(trainer_options.output_dir) &&
      !FileHelper::CreateDirectory(trainer_options.output_dir)) {
    return Status::InternalError(
        "Failed to create model output directory: " +
        trainer_options.output_dir);
  }

  WriteOmegaTimingStatsJson(
      trainer_options.output_dir + "/training_collection_timing.json",
      TrainingDataCollector::ConsumeTimingStats());

  auto train_status = OmegaModelTrainer::TrainModelWithGtCmps(
      training_records, training_result.gt_cmps_data, trainer_options);
  if (!train_status.ok()) {
    return Status::InternalError(
        "Failed to train OMEGA model: " + train_status.message());
  }

  LOG_WARN("OMEGA retrain step 2/2: finished model training for segment %d, output: %s",
           segment_id, trainer_options.output_dir.c_str());

  return Status::OK();
}

}  // namespace zvec
