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

static const std::string SEPARATOR("/");
static const std::string CENTROID_SEPERATOR = "*";

// builder params
static const std::string PARAM_IVF_BUILDER_CENTROID_COUNT(
    "proxima.ivf.builder.centroid_count");
static const std::string PARAM_IVF_BUILDER_CLUSTER_CLASS(
    "proxima.ivf.builder.cluster_class");
static const std::string PARAM_IVF_BUILDER_THREAD_COUNT(
    "proxima.ivf.builder.thread_count");
static const std::string PARAM_IVF_BUILDER_CLUSTER_AUTO_TUNING(
    "proxima.ivf.builder.cluster_auto_tuning");
static const std::string PARAM_IVF_BUILDER_TRAIN_SAMPLE_COUNT(
    "proxima.ivf.builder.train_sample_count");
static const std::string PARAM_IVF_BUILDER_TRAIN_SAMPLE_RATIO(
    "proxima.ivf.builder.train_sample_ratio");
static const std::string PARAM_IVF_BUILDER_CONVERTER_PARAMS(
    "proxima.ivf.builder.converter_params");
static const std::string PARAM_IVF_BUILDER_CONVERTER_CLASS(
    "proxima.ivf.builder.converter_class");
static const std::string PARAM_IVF_BUILDER_STORE_ORIGINAL_FEATURES(
    "proxima.ivf.builder.store_original_features");
static const std::string PARAM_IVF_BUILDER_QUANTIZER_CLASS(
    "proxima.ivf.builder.quantizer_class");
static const std::string PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID(
    "proxima.ivf.builder.quantize_by_centroid");
static const std::string PARAM_IVF_BUILDER_QUANTIZER_PARAMS(
    "proxima.ivf.builder.quantizer_params");
static const std::string PARAM_IVF_BUILDER_CLUSTER_PARAMS_IN_LEVEL_PREFIX(
    "proxima.ivf.builder.cluster_params_in_level_");
static const std::string PARAM_IVF_BUILDER_OPTIMIZER_CLASS(
    "proxima.ivf.builder.optimizer_class");
static const std::string PARAM_IVF_BUILDER_OPTIMIZER_PARAMS(
    "proxima.ivf.builder.optimizer_params");
static const std::string PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS(
    "proxima.ivf.builder.optimizer_quantizer_class");
static const std::string PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_PARAMS(
    "proxima.ivf.builder.optimizer_quantizer_params");
static const std::string PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT(
    "proxima.ivf.builder.block_vector_count");

// searcher params
static const std::string PARAM_IVF_SEARCHER_SCAN_RATIO(
    "proxima.ivf.searcher.scan_ratio");
static const std::string PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD(
    "proxima.ivf.searcher.brute_force_threshold");
static const std::string PARAM_IVF_SEARCHER_OPTIMIZER(
    "proxima.ivf.searcher.optimizer");
static const std::string PARAM_IVF_SEARCHER_OPTIMIZER_PARAMS(
    "proxima.ivf.searcher.optimizer_params");
static const std::string PARAM_IVF_SEARCHER_CONVERTER_REFORMER(
    "proxima.ivf.searcher.converter_reformer");
static const std::string PARAM_IVF_SEARCHER_NPROBE(
    "proxima.ivf.searcher.nprobe");

// Constants
static constexpr char const *kIPMetricName = "InnerProduct";
static constexpr char const *kMipsMetricName = "MipsSquaredEuclidean";
static constexpr char const *kL2MetricName = "SquaredEuclidean";
static constexpr char const *kMipsConverterName = "MipsConverter";
static constexpr char const *kMipsRevConverterName = "MipsReverseConverter";
static constexpr char const *kMipsReformerName = "MipsReformer";
static constexpr char const *kInt8QuantizerName = "Int8QuantizerConverter";
static constexpr char const *kInt4QuantizerName = "Int4QuantizerConverter";
static constexpr char const *kInt8ReformerName = "Int8QuantizerReformer";
static constexpr char const *kInt4ReformerName = "Int4QuantizerReformer";
static constexpr float kNormalizeScaleFactor = 16.0f;

}  // namespace core
}  // namespace zvec
