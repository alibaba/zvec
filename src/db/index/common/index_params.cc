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

#include <sstream>
#include <zvec/db/index_params.h>
#include "type_helper.h"

namespace zvec {

std::string InvertIndexParams::to_string() const {
  std::ostringstream oss;
  oss << "InvertIndexParams{"
      << "enable_range_optimization:"
      << (enable_range_optimization_ ? "true" : "false")
      << ", enable_extended_wildcard:"
      << (enable_extended_wildcard_ ? "true" : "false") << "}";
  return oss.str();
}

std::string VectorIndexParams::vector_index_params_to_string(
    const std::string &class_name, MetricType metric_type,
    QuantizeType quantize_type) const {
  std::ostringstream oss;
  oss << class_name << "{"
      << "metric:" << MetricTypeCodeBook::AsString(metric_type)
      << ",quantize:" << QuantizeTypeCodeBook::AsString(quantize_type);
  return oss.str();
}

}  // namespace zvec