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
#include <string>
#include <vector>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/index_params.h>
#include <zvec/db/status.h>
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/segment/segment.h"
#include "db/training/training_data_collector.h"

namespace zvec {

struct OmegaTrainingParams {
  size_t num_training_queries = 1000;
  int ef_training = 1000;
  int ef_groundtruth = 0;
  uint32_t min_vector_threshold = 100000;
  int k_train = 1;
};

OmegaTrainingParams ResolveOmegaTrainingParams(
    const IndexParams::Ptr &index_params);

#if ZVEC_ENABLE_OMEGA
Result<TrainingDataCollectorResult> CollectOmegaTrainingDataBeforeFlush(
    const Segment::Ptr &segment, const std::string &field_name,
    const VectorColumnIndexer::Ptr &vector_indexer,
    const OmegaTrainingParams &params, const std::string &model_output_dir);

Result<TrainingDataCollectorResult> CollectOmegaRetrainingData(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const OmegaTrainingParams &params, const std::string &model_output_dir);

Status TrainOmegaModelAfterBuild(
    const TrainingDataCollectorResult &training_result,
    const std::string &model_output_dir);

Status TrainOmegaModelAfterRetrainCollect(
    const TrainingDataCollectorResult &training_result,
    const std::string &model_output_dir, SegmentID segment_id,
    const std::string &field_name);
#else
inline OmegaTrainingParams ResolveOmegaTrainingParams(
    const IndexParams::Ptr & /*index_params*/) {
  return {};
}

inline Result<TrainingDataCollectorResult> CollectOmegaTrainingDataBeforeFlush(
    const Segment::Ptr & /*segment*/, const std::string & /*field_name*/,
    const VectorColumnIndexer::Ptr & /*vector_indexer*/,
    const OmegaTrainingParams & /*params*/,
    const std::string & /*model_output_dir*/) {
  return tl::make_unexpected(
      Status::NotSupported("OMEGA is disabled on Android"));
}

inline Result<TrainingDataCollectorResult> CollectOmegaRetrainingData(
    const Segment::Ptr & /*segment*/, const std::string & /*field_name*/,
    const std::vector<VectorColumnIndexer::Ptr> & /*indexers*/,
    const OmegaTrainingParams & /*params*/,
    const std::string & /*model_output_dir*/) {
  return tl::make_unexpected(
      Status::NotSupported("OMEGA is disabled on Android"));
}

inline Status TrainOmegaModelAfterBuild(
    const TrainingDataCollectorResult & /*training_result*/,
    const std::string & /*model_output_dir*/) {
  return Status::NotSupported("OMEGA is disabled on Android");
}

inline Status TrainOmegaModelAfterRetrainCollect(
    const TrainingDataCollectorResult & /*training_result*/,
    const std::string & /*model_output_dir*/, SegmentID /*segment_id*/,
    const std::string & /*field_name*/) {
  return Status::NotSupported("OMEGA is disabled on Android");
}
#endif

}  // namespace zvec
