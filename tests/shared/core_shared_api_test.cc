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

#include <cmath>
#include <utility>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/turbo/turbo.h>

int main() {
  // The core shared SDK must expose each component's public API even when
  // its implementation has no callers inside the library itself.
  zvec::core_interface::FlatIndexParam flat_params;
  flat_params.dimension = 2;
  flat_params.data_type = zvec::core_interface::DataType::DT_FP32;
  auto index =
      zvec::core_interface::IndexFactory::CreateAndInitIndex(flat_params);
  if (!index) {
    return 1;
  }

  const auto distance = zvec::turbo::get_distance_func(
      zvec::turbo::MetricType::kSquaredEuclidean, zvec::turbo::DataType::kFp32,
      zvec::turbo::QuantizeType::kFp32, zvec::turbo::CpuArchType::kScalar);
  if (!distance) {
    return 2;
  }
  const float left[] = {1.0F, 2.0F};
  const float right[] = {4.0F, 6.0F};
  float result = 0.0F;
  distance(left, right, 2, &result);
  if (std::fabs(result - 25.0F) > 0.001F) {
    return 3;
  }

  auto logger = zvec::ailego::LoggerBroker::Register(nullptr);
  if (!logger || logger->init({}) != 0) {
    return 4;
  }
  zvec::ailego::LoggerBroker::Register(std::move(logger));
  if (!zvec::ailego::LoggerBroker::IsLevelEnabled(
          zvec::ailego::Logger::LEVEL_WARN)) {
    return 5;
  }

  return 0;
}
