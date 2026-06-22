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

//! MipsConverter
static const std::string MIPS_CONVERTER_M_VALUE = "mips.converter.m_value";
static const std::string MIPS_CONVERTER_U_VALUE = "mips.converter.u_value";
static const std::string MIPS_CONVERTER_L2_NORM = "mips.converter.l2_norm";
static const std::string MIPS_CONVERTER_FORCED_HALF_FLOAT =
    "mips.converter.forced_half_float";
static const std::string MIPS_CONVERTER_SPHERICAL_INJECTION =
    "mips.converter.spherical_injection";

//! MipsReverseConverter
static const std::string MIPS_REVERSE_CONVERTER_M_VALUE =
    "mips_reverse.converter.m_value";
static const std::string MIPS_REVERSE_CONVERTER_U_VALUE =
    "mips_reverse.converter.u_value";
static const std::string MIPS_REVERSE_CONVERTER_L2_NORM =
    "mips_reverse.converter.l2_norm";
static const std::string MIPS_REVERSE_CONVERTER_FORCED_SINGLE_FLOAT =
    "mips_reverse.converter.forced_single_float";
static const std::string MIPS_REVERSE_CONVERTER_SPHERICAL_INJECTION =
    "mips_reverse.converter.spherical_injection";

//! MipsReformer
static const std::string MIPS_REFORMER_M_VALUE = "mips.reformer.m_value";
static const std::string MIPS_REFORMER_U_VALUE = "mips.reformer.u_value";
static const std::string MIPS_REFORMER_L2_NORM = "mips.reformer.l2_norm";
static const std::string MIPS_REFORMER_NORMALIZE = "mips.reformer.normalize";
static const std::string MIPS_REFORMER_FORCED_HALF_FLOAT =
    "mips.reformer.forced_half_float";
static const std::string MIPS_REFORMER_SPHERICAL_INJECTION =
    "mips.reformer.spherical_injection";

//! NormalizeConverter
static const std::string NORMALIZE_CONVERTER_FORCED_HALF_FLOAT =
    "normalize.converter.forced_half_float";
static const std::string NORMALIZE_CONVERTER_P_VALUE =
    "normalize.converter.p_value";

//! NormalizeReformer
static const std::string NORMALIZE_REFORMER_FORCED_HALF_FLOAT =
    "normalize.reformer.forced_half_float";
static const std::string NORMALIZE_REFORMER_P_VALUE =
    "normalize.reformer.p_value";

//! Int8Converter
static const std::string INT8_QUANTIZER_CONVERTER_HISTOGRAM_BINS_COUNT =
    "int8_quantizer.converter.histogram_bins_count";
static const std::string INT8_QUANTIZER_CONVERTER_DISABLE_BIAS =
    "int8_quantizer.converter.disable_bias";
static const std::string INT8_QUANTIZER_CONVERTER_BIAS =
    "int8_quantizer.converter.bias";
static const std::string INT8_QUANTIZER_CONVERTER_SCALE =
    "int8_quantizer.converter.scale";

//! Int4Converter
static const std::string INT4_QUANTIZER_CONVERTER_HISTOGRAM_BINS_COUNT =
    "int4_quantizer.converter.histogram_bins_count";
static const std::string INT4_QUANTIZER_CONVERTER_DISABLE_BIAS =
    "int4_quantizer.converter.disable_bias";
static const std::string INT4_QUANTIZER_CONVERTER_BIAS =
    "int4_quantizer.converter.bias";
static const std::string INT4_QUANTIZER_CONVERTER_SCALE =
    "int4_quantizer.converter.scale";

//! Int8Reformer
static const std::string INT8_QUANTIZER_REFORMER_BIAS =
    "int8_quantizer.reformer.bias";
static const std::string INT8_QUANTIZER_REFORMER_SCALE =
    "int8_quantizer.reformer.scale";
static const std::string INT8_QUANTIZER_REFORMER_METRIC =
    "int8_quantizer.reformer.metric";

//! Int4Reformer
static const std::string INT4_QUANTIZER_REFORMER_BIAS =
    "int4_quantizer.reformer.bias";
static const std::string INT4_QUANTIZER_REFORMER_SCALE =
    "int4_quantizer.reformer.scale";
static const std::string INT4_QUANTIZER_REFORMER_METRIC =
    "int4_quantizer.reformer.metric";

//! CosineConverter
static const std::string COSINE_CONVERTER_FORCED_HALF_FLOAT =
    "cosine.converter.forced_half_float";

//! CosineReformer
static const std::string COSINE_REFORMER_FORCED_HALF_FLOAT =
    "cosine.reformer.forced_half_float";

//! IntegerStreamingConverter
static const std::string INTEGER_STREAMING_CONVERTER_ENABLE_NORMALIZE =
    "integer_streaming.converter.enable_normalize";

//! IntegerStreamingConverter
static const std::string INTEGER_STREAMING_REFORMER_ENABLE_NORMALIZE =
    "integer_streaming.reformer.enable_normalize";
static const std::string INTEGER_STREAMING_REFORMER_IS_EUCLIDEAN =
    "integer_streaming.reformer.is_euclidean";

//! UniformInt8StreamingConverter / Reformer
static const std::string UNIFORM_INT8_REFORMER_SCALE =
    "uniform_int8.reformer.scale";
static const std::string UNIFORM_INT8_REFORMER_BIAS =
    "uniform_int8.reformer.bias";

//! DoubleBitConverter
static const std::string DOUBLE_BIT_CONVERTER_TRAIN_SAMPLE_COUNT =
    "double_bit.converter.train_sample_count";
static const std::string DOUBLE_BIT_CONVERTER_A_VALUE =
    "double_bit.converter.a_value";
static const std::string DOUBLE_BIT_CONVERTER_B_VALUE =
    "double_bit.converter.b_value";

//! DoubleBitReformer
static const std::string DOUBLE_BIT_REFORMER_A_VALUE =
    "double_bit.reformer.a_value";
static const std::string DOUBLE_BIT_REFORMER_B_VALUE =
    "double_bit.reformer.b_value";

}  // namespace core
}  // namespace zvec