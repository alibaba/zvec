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

#include "db/index/common/manifest_codec.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "db/index/common/manifest/pb_wire.h"
#include "db/index/common/manifest_enum.h"
#include "db/index/common/type_helper.h"

namespace zvec {
namespace {

using pbwire::Reader;
using pbwire::Writer;

// Field numbers of the manifest wire format. Changing any of these
// breaks compatibility with existing manifest files.
namespace f_quantizer {
constexpr uint32_t kEnableRotate = 1;
}
namespace f_base {
constexpr uint32_t kMetricType = 1;
constexpr uint32_t kQuantizeType = 2;
constexpr uint32_t kQuantizerParam = 4;
}  // namespace f_base
namespace f_invert {
constexpr uint32_t kEnableRangeOptimization = 1;
// Added after the protobuf dependency was dropped; old manifests simply lack
// this field and decode to the default (false).
constexpr uint32_t kEnableExtendedWildcard = 2;
}  // namespace f_invert
namespace f_hnsw {
constexpr uint32_t kBase = 1;
constexpr uint32_t kM = 2;
constexpr uint32_t kEfConstruction = 3;
constexpr uint32_t kUseContiguousMemory = 4;
}  // namespace f_hnsw
namespace f_hnsw_rabitq {
constexpr uint32_t kBase = 1;
constexpr uint32_t kM = 2;
constexpr uint32_t kEfConstruction = 3;
constexpr uint32_t kTotalBits = 4;
constexpr uint32_t kNumClusters = 5;
constexpr uint32_t kSampleCount = 6;
}  // namespace f_hnsw_rabitq
namespace f_ivf_rabitq {
constexpr uint32_t kBase = 1;
constexpr uint32_t kNList = 2;
constexpr uint32_t kTotalBits = 3;
constexpr uint32_t kSampleCount = 4;
}  // namespace f_ivf_rabitq
namespace f_flat {
constexpr uint32_t kBase = 1;
}
namespace f_ivf {
constexpr uint32_t kBase = 1;
constexpr uint32_t kNList = 2;
constexpr uint32_t kNIters = 3;
constexpr uint32_t kUseSoar = 4;
}  // namespace f_ivf
namespace f_diskann {
constexpr uint32_t kBase = 1;
constexpr uint32_t kMaxDegree = 2;
constexpr uint32_t kListSize = 3;
constexpr uint32_t kPqChunkNum = 4;
}  // namespace f_diskann
namespace f_vamana {
constexpr uint32_t kBase = 1;
constexpr uint32_t kMaxDegree = 2;
constexpr uint32_t kSearchListSize = 3;
constexpr uint32_t kAlpha = 4;
constexpr uint32_t kSaturateGraph = 5;
constexpr uint32_t kUseContiguousMemory = 6;
constexpr uint32_t kUseIdMap = 7;
constexpr uint32_t kTwoPassBuild = 8;
}  // namespace f_vamana
namespace f_fts {
constexpr uint32_t kTokenizerName = 1;
constexpr uint32_t kFilters = 2;
constexpr uint32_t kExtraParams = 3;
}  // namespace f_fts
namespace f_index_params {
constexpr uint32_t kInvert = 1;
constexpr uint32_t kHnsw = 2;
constexpr uint32_t kFlat = 3;
constexpr uint32_t kIvf = 4;
constexpr uint32_t kHnswRabitq = 5;
constexpr uint32_t kVamana = 6;
constexpr uint32_t kFts = 7;
constexpr uint32_t kDiskann = 8;
constexpr uint32_t kIvfRabitq = 9;
}  // namespace f_index_params
namespace f_field {
constexpr uint32_t kName = 1;
constexpr uint32_t kDataType = 2;
constexpr uint32_t kDimension = 3;
constexpr uint32_t kNullable = 4;
constexpr uint32_t kIndexParams = 5;
}  // namespace f_field
namespace f_collection {
constexpr uint32_t kName = 1;
constexpr uint32_t kFields = 2;
constexpr uint32_t kMaxDocCountPerSegment = 3;
}  // namespace f_collection
namespace f_block {
constexpr uint32_t kBlockId = 1;
constexpr uint32_t kBlockType = 2;
constexpr uint32_t kMinDocId = 3;
constexpr uint32_t kMaxDocId = 4;
constexpr uint32_t kDocCount = 5;
constexpr uint32_t kColumns = 6;
}  // namespace f_block
namespace f_segment {
constexpr uint32_t kSegmentId = 1;
constexpr uint32_t kPersistedBlocks = 2;
constexpr uint32_t kWritingForwardBlock = 3;
constexpr uint32_t kIndexedVectorFields = 4;
}  // namespace f_segment
namespace f_manifest {
constexpr uint32_t kVersion = 1;
constexpr uint32_t kSchema = 2;
constexpr uint32_t kEnableMmap = 3;
constexpr uint32_t kPersistedSegmentMetas = 4;
constexpr uint32_t kWritingSegmentMeta = 5;
constexpr uint32_t kIdMapPathSuffix = 6;
constexpr uint32_t kDeleteSnapshotPathSuffix = 7;
constexpr uint32_t kNextSegmentId = 8;
}  // namespace f_manifest

//! Mirror of proto BaseIndexParams, shared by all vector index params.
struct BaseParams {
  MetricType metric_type{MetricType::UNDEFINED};
  QuantizeType quantize_type{QuantizeType::UNDEFINED};
  // Whether the quantizer_param sub-message is present. The protobuf-based
  // converter called mutable_quantizer_param() for every index type except
  // HNSW_RABITQ, so presence must be reproduced exactly to keep the encoded
  // bytes identical.
  bool has_quantizer_param{false};
  bool enable_rotate{false};
};

void EncodeBase(const BaseParams &base, std::string *out) {
  Writer w(out);
  w.PutVarint(f_base::kMetricType,
              static_cast<uint64_t>(
                  wire::ToNumber(MetricTypeCodeBook::Get(base.metric_type))));
  w.PutVarint(f_base::kQuantizeType,
              static_cast<uint64_t>(wire::ToNumber(
                  QuantizeTypeCodeBook::Get(base.quantize_type))));
  if (base.has_quantizer_param) {
    std::string quantizer;
    Writer qw(&quantizer);
    qw.PutBool(f_quantizer::kEnableRotate, base.enable_rotate);
    w.PutMessage(f_base::kQuantizerParam, quantizer);
  }
}

BaseParams DecodeBase(std::string_view buf) {
  BaseParams base;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_base::kMetricType:
        base.metric_type = MetricTypeCodeBook::Get(
            wire::FromNumber<wire::MetricType>(r.int32_value()));
        break;
      case f_base::kQuantizeType:
        base.quantize_type = QuantizeTypeCodeBook::Get(
            wire::FromNumber<wire::QuantizeType>(r.int32_value()));
        break;
      case f_base::kQuantizerParam: {
        base.has_quantizer_param = true;
        Reader qr(r.bytes());
        while (qr.Next()) {
          if (qr.field() == f_quantizer::kEnableRotate) {
            base.enable_rotate = qr.bool_value();
          }
        }
        break;
      }
      default:
        break;  // unknown field: already consumed by Next()
    }
  }
  return base;
}

//! Builds the base params of an index that carries a quantizer_param.
template <typename Params>
BaseParams MakeBase(const Params *params) {
  BaseParams base;
  base.metric_type = params->metric_type();
  base.quantize_type = params->quantize_type();
  base.has_quantizer_param = true;
  base.enable_rotate = params->quantizer_param().enable_rotate();
  return base;
}

void EncodeHnsw(const HnswIndexParams *params, std::string *out) {
  std::string base;
  EncodeBase(MakeBase(params), &base);
  Writer w(out);
  w.PutMessage(f_hnsw::kBase, base);
  w.PutVarint(f_hnsw::kM, static_cast<uint64_t>(params->m()));
  w.PutVarint(f_hnsw::kEfConstruction,
              static_cast<uint64_t>(params->ef_construction()));
  w.PutBool(f_hnsw::kUseContiguousMemory, params->use_contiguous_memory());
}

HnswIndexParams::OPtr DecodeHnsw(std::string_view buf) {
  BaseParams base;
  int32_t m = 0;
  int32_t ef_construction = 0;
  bool use_contiguous_memory = false;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_hnsw::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_hnsw::kM:
        m = r.int32_value();
        break;
      case f_hnsw::kEfConstruction:
        ef_construction = r.int32_value();
        break;
      case f_hnsw::kUseContiguousMemory:
        use_contiguous_memory = r.bool_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<HnswIndexParams>(
      base.metric_type, m, ef_construction, base.quantize_type,
      use_contiguous_memory, QuantizerParam(base.enable_rotate));
}

void EncodeHnswRabitq(const HnswRabitqIndexParams *params, std::string *out) {
  // NOTE: unlike the other vector indexes, the protobuf converter never
  // touched quantizer_param here, so the sub-message must stay absent.
  BaseParams base;
  base.metric_type = params->metric_type();
  base.quantize_type = params->quantize_type();
  base.has_quantizer_param = false;

  std::string base_buf;
  EncodeBase(base, &base_buf);
  Writer w(out);
  w.PutMessage(f_hnsw_rabitq::kBase, base_buf);
  w.PutVarint(f_hnsw_rabitq::kM, static_cast<uint64_t>(params->m()));
  w.PutVarint(f_hnsw_rabitq::kEfConstruction,
              static_cast<uint64_t>(params->ef_construction()));
  w.PutVarint(f_hnsw_rabitq::kTotalBits,
              static_cast<uint64_t>(params->total_bits()));
  w.PutVarint(f_hnsw_rabitq::kNumClusters,
              static_cast<uint64_t>(params->num_clusters()));
  w.PutVarint(f_hnsw_rabitq::kSampleCount,
              static_cast<uint64_t>(params->sample_count()));
}

HnswRabitqIndexParams::OPtr DecodeHnswRabitq(std::string_view buf) {
  BaseParams base;
  int32_t m = 0;
  int32_t ef_construction = 0;
  int32_t total_bits = 0;
  int32_t num_clusters = 0;
  int32_t sample_count = 0;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_hnsw_rabitq::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_hnsw_rabitq::kM:
        m = r.int32_value();
        break;
      case f_hnsw_rabitq::kEfConstruction:
        ef_construction = r.int32_value();
        break;
      case f_hnsw_rabitq::kTotalBits:
        total_bits = r.int32_value();
        break;
      case f_hnsw_rabitq::kNumClusters:
        num_clusters = r.int32_value();
        break;
      case f_hnsw_rabitq::kSampleCount:
        sample_count = r.int32_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<HnswRabitqIndexParams>(base.metric_type, total_bits,
                                                 num_clusters, m,
                                                 ef_construction, sample_count);
}

//! Like HNSW_RABITQ, the protobuf converter never touched quantizer_param
//! here, so the sub-message must stay absent to keep the bytes identical.
void EncodeIvfRabitq(const IvfRabitqIndexParams *params, std::string *out) {
  BaseParams base;
  base.metric_type = params->metric_type();
  base.quantize_type = params->quantize_type();
  base.has_quantizer_param = false;

  std::string base_buf;
  EncodeBase(base, &base_buf);
  Writer w(out);
  w.PutMessage(f_ivf_rabitq::kBase, base_buf);
  w.PutVarint(f_ivf_rabitq::kNList, static_cast<uint64_t>(params->nlist()));
  w.PutVarint(f_ivf_rabitq::kTotalBits,
              static_cast<uint64_t>(params->total_bits()));
  w.PutVarint(f_ivf_rabitq::kSampleCount,
              static_cast<uint64_t>(params->sample_count()));
}

IvfRabitqIndexParams::OPtr DecodeIvfRabitq(std::string_view buf) {
  BaseParams base;
  int32_t nlist = 0;
  int32_t total_bits = 0;
  int32_t sample_count = 0;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_ivf_rabitq::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_ivf_rabitq::kNList:
        nlist = r.int32_value();
        break;
      case f_ivf_rabitq::kTotalBits:
        total_bits = r.int32_value();
        break;
      case f_ivf_rabitq::kSampleCount:
        sample_count = r.int32_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<IvfRabitqIndexParams>(base.metric_type, nlist,
                                                total_bits, sample_count);
}

void EncodeFlat(const FlatIndexParams *params, std::string *out) {
  std::string base;
  EncodeBase(MakeBase(params), &base);
  Writer w(out);
  w.PutMessage(f_flat::kBase, base);
}

FlatIndexParams::OPtr DecodeFlat(std::string_view buf) {
  BaseParams base;
  Reader r(buf);
  while (r.Next()) {
    if (r.field() == f_flat::kBase) {
      base = DecodeBase(r.bytes());
    }
  }
  return std::make_shared<FlatIndexParams>(base.metric_type, base.quantize_type,
                                           QuantizerParam(base.enable_rotate));
}

void EncodeIvf(const IVFIndexParams *params, std::string *out) {
  std::string base;
  EncodeBase(MakeBase(params), &base);
  Writer w(out);
  w.PutMessage(f_ivf::kBase, base);
  w.PutVarint(f_ivf::kNList, static_cast<uint64_t>(params->n_list()));
  w.PutVarint(f_ivf::kNIters, static_cast<uint64_t>(params->n_iters()));
  w.PutBool(f_ivf::kUseSoar, params->use_soar());
}

IVFIndexParams::OPtr DecodeIvf(std::string_view buf) {
  BaseParams base;
  int32_t n_list = 0;
  int32_t n_iters = 0;
  bool use_soar = false;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_ivf::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_ivf::kNList:
        n_list = r.int32_value();
        break;
      case f_ivf::kNIters:
        n_iters = r.int32_value();
        break;
      case f_ivf::kUseSoar:
        use_soar = r.bool_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<IVFIndexParams>(base.metric_type, n_list, n_iters,
                                          use_soar, base.quantize_type,
                                          QuantizerParam(base.enable_rotate));
}

void EncodeDiskAnn(const DiskAnnIndexParams *params, std::string *out) {
  std::string base;
  EncodeBase(MakeBase(params), &base);
  Writer w(out);
  w.PutMessage(f_diskann::kBase, base);
  w.PutVarint(f_diskann::kMaxDegree,
              static_cast<uint64_t>(params->max_degree()));
  w.PutVarint(f_diskann::kListSize, static_cast<uint64_t>(params->list_size()));
  w.PutVarint(f_diskann::kPqChunkNum,
              static_cast<uint64_t>(params->pq_chunk_num()));
}

DiskAnnIndexParams::OPtr DecodeDiskAnn(std::string_view buf) {
  BaseParams base;
  int32_t max_degree = 0;
  int32_t list_size = 0;
  int32_t pq_chunk_num = 0;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_diskann::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_diskann::kMaxDegree:
        max_degree = r.int32_value();
        break;
      case f_diskann::kListSize:
        list_size = r.int32_value();
        break;
      case f_diskann::kPqChunkNum:
        pq_chunk_num = r.int32_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<DiskAnnIndexParams>(
      base.metric_type, max_degree, list_size, pq_chunk_num, base.quantize_type,
      QuantizerParam(base.enable_rotate));
}

void EncodeVamana(const VamanaIndexParams *params, std::string *out) {
  std::string base;
  EncodeBase(MakeBase(params), &base);
  Writer w(out);
  w.PutMessage(f_vamana::kBase, base);
  w.PutVarint(f_vamana::kMaxDegree,
              static_cast<uint64_t>(params->max_degree()));
  w.PutVarint(f_vamana::kSearchListSize,
              static_cast<uint64_t>(params->search_list_size()));
  w.PutFloat(f_vamana::kAlpha, params->alpha());
  w.PutBool(f_vamana::kSaturateGraph, params->saturate_graph());
  w.PutBool(f_vamana::kUseContiguousMemory, params->use_contiguous_memory());
  w.PutBool(f_vamana::kUseIdMap, params->use_id_map());
  w.PutBool(f_vamana::kTwoPassBuild, params->two_pass_build());
}

VamanaIndexParams::OPtr DecodeVamana(std::string_view buf) {
  BaseParams base;
  int32_t max_degree = 0;
  int32_t search_list_size = 0;
  float alpha = 0.0f;
  bool saturate_graph = false;
  bool use_contiguous_memory = false;
  bool use_id_map = false;
  bool two_pass_build = false;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_vamana::kBase:
        base = DecodeBase(r.bytes());
        break;
      case f_vamana::kMaxDegree:
        max_degree = r.int32_value();
        break;
      case f_vamana::kSearchListSize:
        search_list_size = r.int32_value();
        break;
      case f_vamana::kAlpha:
        alpha = r.float_value();
        break;
      case f_vamana::kSaturateGraph:
        saturate_graph = r.bool_value();
        break;
      case f_vamana::kUseContiguousMemory:
        use_contiguous_memory = r.bool_value();
        break;
      case f_vamana::kUseIdMap:
        use_id_map = r.bool_value();
        break;
      case f_vamana::kTwoPassBuild:
        two_pass_build = r.bool_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<VamanaIndexParams>(
      base.metric_type, max_degree, search_list_size, alpha, saturate_graph,
      use_contiguous_memory, use_id_map, base.quantize_type,
      QuantizerParam(base.enable_rotate), two_pass_build);
}

void EncodeInvert(const InvertIndexParams *params, std::string *out) {
  Writer w(out);
  w.PutBool(f_invert::kEnableRangeOptimization,
            params->enable_range_optimization());
  w.PutBool(f_invert::kEnableExtendedWildcard,
            params->enable_extended_wildcard());
}

InvertIndexParams::OPtr DecodeInvert(std::string_view buf) {
  bool enable_range_optimization = false;
  bool enable_extended_wildcard = false;
  Reader r(buf);
  while (r.Next()) {
    if (r.field() == f_invert::kEnableRangeOptimization) {
      enable_range_optimization = r.bool_value();
    } else if (r.field() == f_invert::kEnableExtendedWildcard) {
      enable_extended_wildcard = r.bool_value();
    }
  }
  return std::make_shared<InvertIndexParams>(enable_range_optimization,
                                             enable_extended_wildcard);
}

void EncodeFts(const FtsIndexParams *params, std::string *out) {
  Writer w(out);
  w.PutString(f_fts::kTokenizerName, params->tokenizer_name());
  for (const auto &filter : params->filters()) {
    w.AddString(f_fts::kFilters, filter);
  }
  w.PutString(f_fts::kExtraParams, params->extra_params());
}

FtsIndexParams::Ptr DecodeFts(std::string_view buf) {
  std::string tokenizer_name;
  std::vector<std::string> filters;
  std::string extra_params;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_fts::kTokenizerName:
        tokenizer_name = r.string_value();
        break;
      case f_fts::kFilters:
        filters.push_back(r.string_value());
        break;
      case f_fts::kExtraParams:
        extra_params = r.string_value();
        break;
      default:
        break;
    }
  }
  return std::make_shared<FtsIndexParams>(tokenizer_name, std::move(filters),
                                          extra_params);
}

}  // namespace

void ManifestCodec::EncodeIndexParams(const IndexParams *params,
                                      std::string *out) {
  if (params == nullptr) {
    return;
  }
  Writer w(out);
  std::string payload;
  switch (params->type()) {
    case IndexType::INVERT:
      if (auto *p = dynamic_cast<const InvertIndexParams *>(params)) {
        EncodeInvert(p, &payload);
        w.PutMessage(f_index_params::kInvert, payload);
      }
      break;
    case IndexType::HNSW:
      if (auto *p = dynamic_cast<const HnswIndexParams *>(params)) {
        EncodeHnsw(p, &payload);
        w.PutMessage(f_index_params::kHnsw, payload);
      }
      break;
    case IndexType::FLAT:
      if (auto *p = dynamic_cast<const FlatIndexParams *>(params)) {
        EncodeFlat(p, &payload);
        w.PutMessage(f_index_params::kFlat, payload);
      }
      break;
    case IndexType::IVF:
      if (auto *p = dynamic_cast<const IVFIndexParams *>(params)) {
        EncodeIvf(p, &payload);
        w.PutMessage(f_index_params::kIvf, payload);
      }
      break;
    case IndexType::HNSW_RABITQ:
      if (auto *p = dynamic_cast<const HnswRabitqIndexParams *>(params)) {
        EncodeHnswRabitq(p, &payload);
        w.PutMessage(f_index_params::kHnswRabitq, payload);
      }
      break;
    case IndexType::IVF_RABITQ:
      if (auto *p = dynamic_cast<const IvfRabitqIndexParams *>(params)) {
        EncodeIvfRabitq(p, &payload);
        w.PutMessage(f_index_params::kIvfRabitq, payload);
      }
      break;
    case IndexType::VAMANA:
      if (auto *p = dynamic_cast<const VamanaIndexParams *>(params)) {
        EncodeVamana(p, &payload);
        w.PutMessage(f_index_params::kVamana, payload);
      }
      break;
    case IndexType::FTS:
      if (auto *p = dynamic_cast<const FtsIndexParams *>(params)) {
        EncodeFts(p, &payload);
        w.PutMessage(f_index_params::kFts, payload);
      }
      break;
    case IndexType::DISKANN:
      if (auto *p = dynamic_cast<const DiskAnnIndexParams *>(params)) {
        EncodeDiskAnn(p, &payload);
        w.PutMessage(f_index_params::kDiskann, payload);
      }
      break;
    default:
      break;
  }
}

IndexParams::Ptr ManifestCodec::DecodeIndexParams(std::string_view buf) {
  // oneof semantics: the last branch present on the wire wins.
  IndexParams::Ptr params;
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_index_params::kInvert:
        params = DecodeInvert(r.bytes());
        break;
      case f_index_params::kHnsw:
        params = DecodeHnsw(r.bytes());
        break;
      case f_index_params::kFlat:
        params = DecodeFlat(r.bytes());
        break;
      case f_index_params::kIvf:
        params = DecodeIvf(r.bytes());
        break;
      case f_index_params::kHnswRabitq:
        params = DecodeHnswRabitq(r.bytes());
        break;
      case f_index_params::kIvfRabitq:
        params = DecodeIvfRabitq(r.bytes());
        break;
      case f_index_params::kVamana:
        params = DecodeVamana(r.bytes());
        break;
      case f_index_params::kFts:
        params = DecodeFts(r.bytes());
        break;
      case f_index_params::kDiskann:
        params = DecodeDiskAnn(r.bytes());
        break;
      default:
        break;
    }
  }
  return params;
}

void ManifestCodec::EncodeFieldSchema(const FieldSchema &field,
                                      std::string *out) {
  Writer w(out);
  w.PutString(f_field::kName, field.name());
  w.PutVarint(f_field::kDataType,
              static_cast<uint64_t>(
                  wire::ToNumber(DataTypeCodeBook::Get(field.data_type()))));
  w.PutVarint(f_field::kDimension, field.dimension());
  w.PutBool(f_field::kNullable, field.nullable());
  auto index_params = field.index_params();
  if (index_params) {
    std::string payload;
    EncodeIndexParams(index_params.get(), &payload);
    w.PutMessage(f_field::kIndexParams, payload);
  }
}

FieldSchema::Ptr ManifestCodec::DecodeFieldSchema(std::string_view buf) {
  auto field = std::make_shared<FieldSchema>();
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_field::kName:
        field->set_name(r.string_value());
        break;
      case f_field::kDataType:
        field->set_data_type(DataTypeCodeBook::Get(
            wire::FromNumber<wire::DataType>(r.int32_value())));
        break;
      case f_field::kDimension:
        field->set_dimension(r.uint32_value());
        break;
      case f_field::kNullable:
        field->set_nullable(r.bool_value());
        break;
      case f_field::kIndexParams:
        field->set_index_params(DecodeIndexParams(r.bytes()));
        break;
      default:
        break;
    }
  }
  return field;
}

void ManifestCodec::EncodeCollectionSchema(const CollectionSchema &schema,
                                           std::string *out) {
  Writer w(out);
  w.PutString(f_collection::kName, schema.name());
  for (const auto &field : schema.fields()) {
    std::string payload;
    EncodeFieldSchema(*field, &payload);
    w.PutMessage(f_collection::kFields, payload);
  }
  w.PutVarint(f_collection::kMaxDocCountPerSegment,
              schema.max_doc_count_per_segment());
}

CollectionSchema::Ptr ManifestCodec::DecodeCollectionSchema(
    std::string_view buf) {
  auto schema = std::make_shared<CollectionSchema>();
  // The protobuf-based converter read max_doc_count_per_segment straight from
  // the wire, so a message without the field decoded to zero rather than to
  // the C++ default. Mirror that to stay compatible with such manifests.
  schema->set_max_doc_count_per_segment(0);
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_collection::kName:
        schema->set_name(r.string_value());
        break;
      case f_collection::kFields:
        schema->add_field(DecodeFieldSchema(r.bytes()));
        break;
      case f_collection::kMaxDocCountPerSegment:
        schema->set_max_doc_count_per_segment(r.varint());
        break;
      default:
        break;
    }
  }
  return schema;
}

void ManifestCodec::EncodeBlockMeta(const BlockMeta &meta, std::string *out) {
  Writer w(out);
  w.PutVarint(f_block::kBlockId, meta.id());
  w.PutVarint(f_block::kBlockType, static_cast<uint64_t>(wire::ToNumber(
                                       BlockTypeCodeBook::Get(meta.type()))));
  w.PutVarint(f_block::kMinDocId, meta.min_doc_id());
  w.PutVarint(f_block::kMaxDocId, meta.max_doc_id());
  w.PutVarint(f_block::kDocCount, meta.doc_count());
  for (const auto &column : meta.columns()) {
    w.AddString(f_block::kColumns, column);
  }
}

BlockMeta::Ptr ManifestCodec::DecodeBlockMeta(std::string_view buf) {
  auto meta = std::make_shared<BlockMeta>();
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_block::kBlockId:
        meta->set_id(r.uint32_value());
        break;
      case f_block::kBlockType:
        meta->set_type(BlockTypeCodeBook::Get(
            wire::FromNumber<wire::BlockType>(r.int32_value())));
        break;
      case f_block::kMinDocId:
        meta->set_min_doc_id(r.varint());
        break;
      case f_block::kMaxDocId:
        meta->set_max_doc_id(r.varint());
        break;
      case f_block::kDocCount:
        meta->set_doc_count(r.uint32_value());
        break;
      case f_block::kColumns:
        meta->add_column(r.string_value());
        break;
      default:
        break;
    }
  }
  return meta;
}

void ManifestCodec::EncodeSegmentMeta(const SegmentMeta &meta,
                                      std::string *out) {
  Writer w(out);
  w.PutVarint(f_segment::kSegmentId, meta.id());
  for (const auto &block : meta.persisted_blocks()) {
    std::string payload;
    EncodeBlockMeta(block, &payload);
    w.PutMessage(f_segment::kPersistedBlocks, payload);
  }
  if (meta.has_writing_forward_block()) {
    std::string payload;
    EncodeBlockMeta(meta.writing_forward_block().value(), &payload);
    w.PutMessage(f_segment::kWritingForwardBlock, payload);
  }
  for (const auto &field : meta.indexed_vector_fields()) {
    w.AddString(f_segment::kIndexedVectorFields, field);
  }
}

SegmentMeta::Ptr ManifestCodec::DecodeSegmentMeta(std::string_view buf) {
  auto meta = std::make_shared<SegmentMeta>(0);
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_segment::kSegmentId:
        meta->set_id(r.uint32_value());
        break;
      case f_segment::kPersistedBlocks:
        meta->add_persisted_block(*DecodeBlockMeta(r.bytes()));
        break;
      case f_segment::kWritingForwardBlock:
        meta->set_writing_forward_block(*DecodeBlockMeta(r.bytes()));
        break;
      case f_segment::kIndexedVectorFields:
        meta->add_indexed_vector_field(r.string_value());
        break;
      default:
        break;
    }
  }
  return meta;
}

Status ManifestCodec::Encode(const ManifestData &data, std::string *out) {
  Writer w(out);
  w.PutVarint(f_manifest::kVersion, data.version);
  if (data.schema) {
    std::string payload;
    EncodeCollectionSchema(*data.schema, &payload);
    w.PutMessage(f_manifest::kSchema, payload);
  }
  w.PutBool(f_manifest::kEnableMmap, data.enable_mmap);
  for (const auto &meta : data.persisted_segment_metas) {
    if (!meta) {
      continue;
    }
    std::string payload;
    EncodeSegmentMeta(*meta, &payload);
    w.PutMessage(f_manifest::kPersistedSegmentMetas, payload);
  }
  if (data.writing_segment_meta) {
    std::string payload;
    EncodeSegmentMeta(*data.writing_segment_meta, &payload);
    w.PutMessage(f_manifest::kWritingSegmentMeta, payload);
  }
  w.PutVarint(f_manifest::kIdMapPathSuffix, data.id_map_path_suffix);
  w.PutVarint(f_manifest::kDeleteSnapshotPathSuffix,
              data.delete_snapshot_path_suffix);
  w.PutVarint(f_manifest::kNextSegmentId, data.next_segment_id);
  return Status::OK();
}

Status ManifestCodec::Decode(std::string_view buf, ManifestData *data) {
  Reader r(buf);
  while (r.Next()) {
    switch (r.field()) {
      case f_manifest::kVersion:
        data->version = r.uint32_value();
        break;
      case f_manifest::kSchema:
        data->schema = DecodeCollectionSchema(r.bytes());
        break;
      case f_manifest::kEnableMmap:
        data->enable_mmap = r.bool_value();
        break;
      case f_manifest::kPersistedSegmentMetas:
        data->persisted_segment_metas.push_back(DecodeSegmentMeta(r.bytes()));
        break;
      case f_manifest::kWritingSegmentMeta:
        data->writing_segment_meta = DecodeSegmentMeta(r.bytes());
        break;
      case f_manifest::kIdMapPathSuffix:
        data->id_map_path_suffix = r.uint32_value();
        break;
      case f_manifest::kDeleteSnapshotPathSuffix:
        data->delete_snapshot_path_suffix = r.uint32_value();
        break;
      case f_manifest::kNextSegmentId:
        data->next_segment_id = r.uint32_value();
        break;
      default:
        break;
    }
  }
  if (!r.ok()) {
    return Status::InternalError("Malformed manifest data");
  }
  if (!data->schema) {
    // An absent schema means the file is not a valid manifest; the protobuf
    // parser accepted it but downstream code always expects a schema.
    // The old reader decoded the default (empty) protobuf message in that
    // case, yielding a schema whose max_doc_count_per_segment is zero.
    data->schema = std::make_shared<CollectionSchema>();
    data->schema->set_max_doc_count_per_segment(0);
  }
  return Status::OK();
}

}  // namespace zvec
