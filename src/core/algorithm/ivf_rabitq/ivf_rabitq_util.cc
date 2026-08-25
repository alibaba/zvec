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
#include "ivf_rabitq_util.h"
#include <zvec/ailego/logger/logger.h>
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "zvec/core/framework/index_error.h"

namespace zvec {
namespace core {

int PrepareAndCheckIvfRabitqInternalMeta(const IndexMeta &meta,
                                         const ailego::Params &params,
                                         IndexMeta *rabitq_meta,
                                         std::string *metric_name) {
  if (!rabitq_meta || !metric_name) {
    LOG_ERROR("Invalid IVF RaBitQ meta arguments");
    return IndexError_InvalidArgument;
  }

  *rabitq_meta = meta;
  *metric_name = meta.metric_name();
  if (metric_name->empty()) {
    LOG_ERROR("Meta metric is empty");
    return IndexError_InvalidArgument;
  }

  // Keep public meta unchanged. The internal RaBitQ meta describes only the
  // dimensions consumed by centroids, rotator and quantized data.
  uint32_t general_dimension = 0;
  params.get(PARAM_RABITQ_GENERAL_DIMENSION, &general_dimension);
  if (*metric_name == "Cosine" && general_dimension == 0) {
    LOG_ERROR("%s not set for Cosine IVF RaBitQ",
              PARAM_RABITQ_GENERAL_DIMENSION.c_str());
    return IndexError_InvalidArgument;
  }
  if (general_dimension > meta.dimension()) {
    LOG_ERROR("Invalid general dimension=%zu, meta dimension=%zu",
              static_cast<size_t>(general_dimension),
              static_cast<size_t>(meta.dimension()));
    return IndexError_InvalidArgument;
  }
  if (general_dimension > 0) {
    rabitq_meta->set_dimension(general_dimension);
  }

  if (*metric_name == "Cosine") {
    rabitq_meta->set_metric("InnerProduct", 0, ailego::Params());
    *metric_name = "InnerProduct";
  }

  if (rabitq_meta->data_type() != IndexMeta::DataType::DT_FP32) {
    LOG_ERROR("IVF RaBitQ only supports FP32 data type");
    return IndexError_Unsupported;
  }

  uint32_t dim = rabitq_meta->dimension();
  if (dim < kMinRabitqDimSize || dim > kMaxRabitqDimSize) {
    LOG_ERROR("Invalid dimension=%zu, must be in [%d, %d]", (size_t)dim,
              kMinRabitqDimSize, kMaxRabitqDimSize);
    return IndexError_InvalidArgument;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
