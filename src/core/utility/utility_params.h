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

static const std::string INDEX_META_SEGMENT_NAME = "IndexMeta";
static const std::string INDEX_VERSION_SEGMENT_NAME = "IndexVersion";

//! FileLogger
static const std::string FILE_LOGGER_PATH = "proxima.file.logger.path";

//! FileContainer
static const std::string FILE_READ_STORAGE_CHECKSUM_VALIDATION =
    "proxima.file.read_storage.checksum_validation";
static const std::string FILE_READ_STORAGE_ENABLE_DIRECT_IO =
    "proxima.file.read_storage.enable_direct_io";
static const std::string FILE_READ_STORAGE_ALONE_FILE_HANDLE =
    "proxima.file.read_storage.alone_file_handle";
static const std::string FILE_READ_STORAGE_MEMORY_LOCKED =
    "proxima.file.read_storage.memory_locked";
static const std::string FILE_READ_STORAGE_MEMORY_WARMUP =
    "proxima.file.read_storage.memory_warmup";
static const std::string FILE_READ_STORAGE_MEMORY_SHARED =
    "proxima.file.read_storage.memory_shared";
static const std::string FILE_READ_STORAGE_HEADER_OFFSET =
    "proxima.file.read_storage.header_offset";
static const std::string FILE_READ_STORAGE_FOOTER_OFFSET =
    "proxima.file.read_storage.footer_offset";

//! MemoryContainer
static const std::string MEMORY_CONTAINER_CHECKSUM_VALIDATION =
    "proxima.memory.container.checksum_validation";

//! MMapFileContainer
static const std::string MMAPFILE_READ_STORAGE_MEMORY_LOCKED =
    "proxima.mmap_file.container.memory_locked";
static const std::string MMAPFILE_READ_STORAGE_MEMORY_WARMUP =
    "proxima.mmap_file.container.memory_warmup";
static const std::string MMAPFILE_READ_STORAGE_MEMORY_SHARED =
    "proxima.mmap_file.container.memory_shared";
static const std::string MMAPFILE_READ_STORAGE_CHECKSUM_VALIDATION =
    "proxima.mmap_file.container.checksum_validation";
static const std::string MMAPFILE_READ_STORAGE_HEADER_OFFSET =
    "proxima.mmap_file.container.header_offset";
static const std::string MMAPFILE_READ_STORAGE_FOOTER_OFFSET =
    "proxima.mmap_file.container.footer_offset";

//! MMapFileStorage
static const std::string MMAPFILE_STORAGE_MEMORY_LOCKED =
    "proxima.mmap_file.storage.memory_locked";
static const std::string MMAPFILE_STORAGE_MEMORY_WARMUP =
    "proxima.mmap_file.storage.memory_warmup";
static const std::string MMAPFILE_STORAGE_COPY_ON_WRITE =
    "proxima.mmap_file.storage.copy_on_write";
static const std::string MMAPFILE_STORAGE_FORCE_FLUSH =
    "proxima.mmap_file.storage.force_flush";
static const std::string MMAPFILE_STORAGE_SEGMENT_META_CAPACITY =
    "proxima.mmap_file.storage.segment_meta_capacity";
//! MipsConverter
static const std::string MIPS_CONVERTER_M_VALUE =
    "proxima.mips.converter.m_value";
static const std::string MIPS_CONVERTER_U_VALUE =
    "proxima.mips.converter.u_value";
static const std::string MIPS_CONVERTER_L2_NORM =
    "proxima.mips.converter.l2_norm";
static const std::string MIPS_CONVERTER_FORCED_HALF_FLOAT =
    "proxima.mips.converter.forced_half_float";
static const std::string MIPS_CONVERTER_SPHERICAL_INJECTION =
    "proxima.mips.converter.spherical_injection";

//! MipsReverseConverter
static const std::string MIPS_REVERSE_CONVERTER_M_VALUE =
    "proxima.mips_reverse.converter.m_value";
static const std::string MIPS_REVERSE_CONVERTER_U_VALUE =
    "proxima.mips_reverse.converter.u_value";
static const std::string MIPS_REVERSE_CONVERTER_L2_NORM =
    "proxima.mips_reverse.converter.l2_norm";
static const std::string MIPS_REVERSE_CONVERTER_FORCED_SINGLE_FLOAT =
    "proxima.mips_reverse.converter.forced_single_float";
static const std::string MIPS_REVERSE_CONVERTER_SPHERICAL_INJECTION =
    "proxima.mips_reverse.converter.spherical_injection";

//! MipsReformer
static const std::string MIPS_REFORMER_M_VALUE =
    "proxima.mips.reformer.m_value";
static const std::string MIPS_REFORMER_U_VALUE =
    "proxima.mips.reformer.u_value";
static const std::string MIPS_REFORMER_L2_NORM =
    "proxima.mips.reformer.l2_norm";
static const std::string MIPS_REFORMER_NORMALIZE =
    "proxima.mips.reformer.normalize";
static const std::string MIPS_REFORMER_FORCED_HALF_FLOAT =
    "proxima.mips.reformer.forced_half_float";
static const std::string MIPS_REFORMER_SPHERICAL_INJECTION =
    "proxima.mips.reformer.spherical_injection";

//! MipsEuclideanMeasure
static const std::string MIPS_EUCLIDEAN_METRIC_M_VALUE =
    "proxima.mips_euclidean.metric.m_value";
static const std::string MIPS_EUCLIDEAN_METRIC_U_VALUE =
    "proxima.mips_euclidean.metric.u_value";
static const std::string MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM =
    "proxima.mips_euclidean.metric.max_l2_norm";
static const std::string MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE =
    "proxima.mips_euclidean.metric.injection_type";

//! NormalizeConverter
static const std::string NORMALIZE_CONVERTER_FORCED_HALF_FLOAT =
    "proxima.normalize.converter.forced_half_float";
static const std::string NORMALIZE_CONVERTER_P_VALUE =
    "proxima.normalize.converter.p_value";

//! NormalizeReformer
static const std::string NORMALIZE_REFORMER_FORCED_HALF_FLOAT =
    "proxima.normalize.reformer.forced_half_float";
static const std::string NORMALIZE_REFORMER_P_VALUE =
    "proxima.normalize.reformer.p_value";

//! Int8Converter
static const std::string INT8_QUANTIZER_CONVERTER_HISTOGRAM_BINS_COUNT =
    "proxima.int8_quantizer.converter.histogram_bins_count";
static const std::string INT8_QUANTIZER_CONVERTER_DISABLE_BIAS =
    "proxima.int8_quantizer.converter.disable_bias";
static const std::string INT8_QUANTIZER_CONVERTER_BIAS =
    "proxima.int8_quantizer.converter.bias";
static const std::string INT8_QUANTIZER_CONVERTER_SCALE =
    "proxima.int8_quantizer.converter.scale";

//! Int4Converter
static const std::string INT4_QUANTIZER_CONVERTER_HISTOGRAM_BINS_COUNT =
    "proxima.int4_quantizer.converter.histogram_bins_count";
static const std::string INT4_QUANTIZER_CONVERTER_DISABLE_BIAS =
    "proxima.int4_quantizer.converter.disable_bias";
static const std::string INT4_QUANTIZER_CONVERTER_BIAS =
    "proxima.int4_quantizer.converter.bias";
static const std::string INT4_QUANTIZER_CONVERTER_SCALE =
    "proxima.int4_quantizer.converter.scale";

//! Int8Reformer
static const std::string INT8_QUANTIZER_REFORMER_BIAS =
    "proxima.int8_quantizer.reformer.bias";
static const std::string INT8_QUANTIZER_REFORMER_SCALE =
    "proxima.int8_quantizer.reformer.scale";
static const std::string INT8_QUANTIZER_REFORMER_METRIC =
    "proxima.int8_quantizer.reformer.metric";

//! Int4Reformer
static const std::string INT4_QUANTIZER_REFORMER_BIAS =
    "proxima.int4_quantizer.reformer.bias";
static const std::string INT4_QUANTIZER_REFORMER_SCALE =
    "proxima.int4_quantizer.reformer.scale";
static const std::string INT4_QUANTIZER_REFORMER_METRIC =
    "proxima.int4_quantizer.reformer.metric";

//! CosineConverter
static const std::string COSINE_CONVERTER_FORCED_HALF_FLOAT =
    "proxima.cosine.converter.forced_half_float";

//! CosineReformer
static const std::string COSINE_REFORMER_FORCED_HALF_FLOAT =
    "proxima.cosine.reformer.forced_half_float";

//! QuantizedInteger Metric
static const std::string QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME =
    "proxima.quantized_integer.metric.origin_metric_name";
static const std::string QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS =
    "proxima.quantized_integer.metric.origin_metric_params";

//! IntegerStreamingConverter
static const std::string INTEGER_STREAMING_CONVERTER_ENABLE_NORMALIZE =
    "proxima.integer_streaming.converter.enable_normalize";

//! IntegerStreamingConverter
static const std::string INTEGER_STREAMING_REFORMER_ENABLE_NORMALIZE =
    "proxima.integer_streaming.reformer.enable_normalize";

//! DoubleBitConverter
static const std::string DOUBLE_BIT_CONVERTER_TRAIN_SAMPLE_COUNT =
    "proxima.double_bit.converter.train_sample_count";
static const std::string DOUBLE_BIT_CONVERTER_A_VALUE =
    "proxima.double_bit.converter.a_value";
static const std::string DOUBLE_BIT_CONVERTER_B_VALUE =
    "proxima.double_bit.converter.b_value";

//! DoubleBitReformer
static const std::string DOUBLE_BIT_REFORMER_A_VALUE =
    "proxima.double_bit.reformer.a_value";
static const std::string DOUBLE_BIT_REFORMER_B_VALUE =
    "proxima.double_bit.reformer.b_value";

//! SimpleForward
static const std::string SIMPLE_FORWARD_DATA_BLOCK_SIZE =
    "proxima.simple.forward.data_block_size";
static const std::string SIMPLE_FORWARD_INDEX_BLOCK_SIZE =
    "proxima.simple.forward.index_block_size";

//! SimpleForward
static const std::string SIMPLE_CLOSET_DATA_BLOCK_SIZE =
    "proxima.simple.closet.data_block_size";
static const std::string SIMPLE_CLOSET_INDEX_BLOCK_SIZE =
    "proxima.simple.closet.index_block_size";

//! ChainCloset
static const std::string CHAIN_CLOSET_SLOT_SIZE =
    "proxima.chain.closet.slot_size";
static const std::string CHAIN_CLOSET_INDEX_BLOCK_SIZE =
    "proxima.chain.closet.index_block_size";
static const std::string CHAIN_CLOSET_DATA_BLOCK_SIZE =
    "proxima.chain.closet.data_block_size";

//! IndexForward
static const std::string PARAM_FORWARD_MULTI_VALUE =
    "proxima.param.forward.multi_value";
static const std::string PARAM_FORWARD_MULTI_COUNT =
    "proxima.param.forward.multi_count";

}  // namespace core
}  // namespace zvec