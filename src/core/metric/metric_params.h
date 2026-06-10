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

namespace zvec {
namespace core {

//! MipsEuclideanMetric
static const std::string MIPS_EUCLIDEAN_METRIC_M_VALUE =
    "mips_euclidean.metric.m_value";
static const std::string MIPS_EUCLIDEAN_METRIC_U_VALUE =
    "mips_euclidean.metric.u_value";
static const std::string MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM =
    "mips_euclidean.metric.max_l2_norm";
static const std::string MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE =
    "mips_euclidean.metric.injection_type";

//! QuantizedInteger Metric
static const std::string QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME =
    "proxima.quantized_integer.metric.origin_metric_name";
static const std::string QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS =
    "proxima.quantized_integer.metric.origin_metric_params";

//! UniformInt8 Metric
static const std::string UNIFORM_INT8_METRIC_ORIGIN_METRIC_NAME =
    "proxima.uniform_int8.metric.origin_metric_name";

}  // namespace core
}  // namespace zvec