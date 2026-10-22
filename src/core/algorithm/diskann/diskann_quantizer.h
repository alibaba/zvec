// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_params.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

// Creation and storage boundary for DiskANN candidate quantizers. Algorithm
// code uses the returned Turbo quantizer directly: no adapter dispatch is
// added to encoding or distance calculation. DiskAnnPqMeta's chunk_num remains
// the persisted byte stride, preserving both legacy and current PQ layouts.
class DiskAnnQuantizer {
 public:
  int init(IndexMeta *stored_meta, const ailego::Params &params) {
    *this = DiskAnnQuantizer{};
    name_ = params.get_as_string(PARAM_DISKANN_BUILDER_QUANTIZER_CLASS);
    if (name_.empty()) {
      name_ = stored_meta->quantizer_name() == "RabitqQuantizer"
                  ? "RabitqQuantizer"
                  : "PqInt8Quantizer";
    }
    if (name_ != "PqInt8Quantizer" && name_ != "RabitqQuantizer")
      return IndexError_Unsupported;
    if (name_ == "RabitqQuantizer") {
#if !RABITQ_SUPPORTED
      return IndexError_Unsupported;
#else
      settings_ = stored_meta->quantizer_params();
      params.get(PARAM_DISKANN_BUILDER_QUANTIZER_PARAMS, &settings_);
      stored_meta->set_quantizer(name_, 0, settings_);
      int ret = candidate_quantizer_meta(*stored_meta, &meta_);
      if (ret != 0) return ret;
      // Fixed-width formats can be initialized and validated immediately.
      return create();
#endif
    }
    // An explicit PQ selection replaces an inherited candidate descriptor.
    if (stored_meta->quantizer_name() == "RabitqQuantizer")
      stored_meta->set_quantizer("", 0, ailego::Params{});
    uint32_t chunks = 0;
    if (params.get(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, &chunks)) {
      if (chunks > stored_meta->dimension()) return IndexError_InvalidArgument;
      max_pq_chunks_ = chunks;
    }
    return DiskAnnUtil::quantizer_init_meta(*stored_meta, &meta_);
  }

  // PQ keeps its existing dimension/budget-based chunk selection. Fixed-width
  // encodings must fit the budget without silently changing their precision.
  int prepare(size_t count, bool has_budget, size_t budget_bytes,
              uint32_t thread_count) {
    if (count == 0) return IndexError_InvalidLength;
    if (name_ == "PqInt8Quantizer") {
      uint32_t chunks = max_pq_chunks_;
      if (chunks == 0 || chunks == std::numeric_limits<uint32_t>::max())
        chunks = std::max(1U, static_cast<uint32_t>(meta_.dimension() / 2));
      if (has_budget) {
        const size_t budget_chunks = budget_bytes / count;
        if (budget_chunks == 0) return IndexError_InvalidArgument;
        chunks = std::min(chunks, static_cast<uint32_t>(std::min<size_t>(
                                      budget_chunks, meta_.dimension())));
      }
      if (chunks == 0 || chunks > meta_.dimension())
        return IndexError_InvalidArgument;
      settings_.set("num_chunk", chunks);
      settings_.set("thread_count", std::max(1U, thread_count));
      settings_.set("use_zero_mean", false);
      int ret = create();
      if (ret != 0) return ret;
    }
    if (!quantizer_) return IndexError_NoReady;
    const size_t bytes = quantizer_->quantized_datapoint_vector_length();
    if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max() ||
        count > std::numeric_limits<size_t>::max() / bytes)
      return IndexError_InvalidLength;
    if (has_budget && bytes > budget_bytes / count) return IndexError_NoMemory;
    code_size_ = static_cast<uint32_t>(bytes);
    return 0;
  }

  const turbo::Quantizer::Pointer &get() const {
    return quantizer_;
  }
  uint32_t code_size() const {
    return code_size_;
  }

  //! Resolve the quantizer implementation name from a serialized quantizer
  //! meta buffer (see turbo::QuantizerSerHeader).  Extend the mapping here when
  //! DiskAnn supports a new quantize type.
  static const char *quantizer_name_from_meta_buffer(
      const std::string &meta_buffer) {
    if (meta_buffer.size() < sizeof(turbo::QuantizerSerHeader)) {
      return nullptr;
    }
    turbo::QuantizerSerHeader hdr;
    std::memcpy(&hdr, meta_buffer.data(), sizeof(hdr));
    if (hdr.magic != turbo::kQuantizerMagic) {
      return nullptr;
    }
    switch (static_cast<turbo::QuantizeType>(hdr.quant_type)) {
      case turbo::QuantizeType::kRabit:
        return "RabitqQuantizer";
      case turbo::QuantizeType::kPQ:
        return "PqInt8Quantizer";
      case turbo::QuantizeType::kFp16:
        return "Fp16Quantizer";
      case turbo::QuantizeType::kFp32:
        return "Fp32Quantizer";
      default:
        return nullptr;
    }
  }

  // DiskANN graph construction keeps its existing metric. Candidate codes
  // use IP directly, and L2 on the normalized components for Cosine.
  static int candidate_quantizer_meta(const IndexMeta &meta, IndexMeta *out) {
    int ret = DiskAnnUtil::quantizer_init_meta(meta, out);
    if (ret == 0 && meta.quantizer_name() == "RabitqQuantizer" &&
        meta.metric_name() == "InnerProduct")
      out->set_metric("InnerProduct", 0, ailego::Params{});
    return ret;
  }

  //! Resolve the layout from the DiskAnn header, not from centroid values.
  //! The legacy chunk count occupies bytes reserved as zero by current writers.
  //! Preserve this result when interpreting the payload: valid legacy pivots
  //! can start with the very same bytes as kQuantizerMagic.
  static int normalize_pq_meta(const IndexMeta &meta, DiskAnnPqMeta *pq_meta,
                               bool *legacy_layout) {
    *legacy_layout = false;
    DiskAnnLegacyPqMeta legacy;
    std::memcpy(static_cast<void *>(&legacy), pq_meta, sizeof(legacy));
    IndexMeta init_meta;
    int ret = DiskAnnUtil::quantizer_init_meta(meta, &init_meta);
    if (ret != 0) {
      return ret;
    }
    if (legacy.chunk_num == 0) {
      if (pq_meta->chunk_num == 0 ||
          pq_meta->chunk_num > std::numeric_limits<uint32_t>::max() ||
          (meta.quantizer_name() != "RabitqQuantizer" &&
           pq_meta->chunk_num > init_meta.dimension()) ||
          pq_meta->quantizer_meta_buffer_size <
              sizeof(turbo::QuantizerSerHeader)) {
        return IndexError_InvalidFormat;
      }
      return 0;
    }

    const uint64_t mean_bytes =
        uint64_t{init_meta.dimension()} * init_meta.unit_size();
    if (legacy.chunk_num > init_meta.dimension() || mean_bytes == 0 ||
        legacy.centroid_data_size != mean_bytes ||
        legacy.full_pivot_data_size != 256 * mean_bytes) {
      LOG_ERROR("Invalid legacy DiskAnn PQ metadata");
      return IndexError_InvalidFormat;
    }

    *legacy_layout = true;
    pq_meta->clear();
    pq_meta->chunk_num = legacy.chunk_num;
    // Legacy writers left chunk_offsets_size unset.
    pq_meta->quantizer_meta_buffer_size =
        legacy.full_pivot_data_size + legacy.centroid_data_size +
        (legacy.chunk_num + 1) * sizeof(uint32_t);
    return 0;
  }

  //! Repack a legacy codebook into the layout the PQ quantizer keeps in memory:
  //! on disk 256 pivot rows of `dim` components, a per-dimension mean and
  //! chunk_num + 1 chunk offsets; in memory each chunk holds [256][its width].
  //! Leading chunks carry any remainder dimensions, matching legacy chunking.
  //! Both sides work in squared euclidean space, and the
  //! writers of this layout always persisted a zero mean.
  static int legacy_pq_codebook_to_centroids(const std::string &meta_buffer,
                                             const IndexMeta &quantizer_meta,
                                             uint32_t chunk_num,
                                             std::string *centroids) {
    constexpr size_t kClusterNum = 256;

    const size_t dim = quantizer_meta.dimension();
    const size_t unit = quantizer_meta.unit_size();
    if (dim == 0 || unit == 0 || chunk_num == 0 || chunk_num > dim) {
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook with %u chunks over %zu dimensions is "
          "not loadable, rebuild the index",
          chunk_num, dim);
      return IndexError_Unsupported;
    }
    const size_t sub_dim = dim / chunk_num;
    const size_t remainder = dim % chunk_num;

    const size_t pivot_bytes = kClusterNum * dim * unit;
    const size_t mean_bytes = dim * unit;
    const size_t offset_count = static_cast<size_t>(chunk_num) + 1;
    const size_t offset_bytes = offset_count * sizeof(uint32_t);
    if (meta_buffer.size() != pivot_bytes + mean_bytes + offset_bytes) {
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook size mismatch, expect: %zu, actual: %zu",
          pivot_bytes + mean_bytes + offset_bytes, meta_buffer.size());
      return IndexError_InvalidFormat;
    }

    const char *pivots = meta_buffer.data();
    const char *mean = pivots + pivot_bytes;
    // Copied out instead of cast in place: the offsets are not guaranteed to be
    // aligned inside the buffer.
    std::vector<uint32_t> offsets(offset_count, 0);
    std::memcpy(offsets.data(), mean + mean_bytes, offset_bytes);
    for (size_t m = 0; m <= chunk_num; ++m) {
      if (offsets[m] != m * sub_dim + std::min(m, remainder)) {
        LOG_ERROR(
            "Legacy DiskAnn PQ codebook has invalid chunk offsets at chunk "
            "%zu, rebuild the index",
            m);
        return IndexError_Unsupported;
      }
    }

    // The quantizer is initialized without a mean, so a residual codebook would
    // be decoded against the wrong origin.
    if (std::any_of(mean, mean + mean_bytes,
                    [](char byte) { return byte != 0; })) {
      LOG_ERROR(
          "Legacy DiskAnn PQ codebook carries a non-zero mean, rebuild the "
          "index");
      return IndexError_Unsupported;
    }

    centroids->assign(pivot_bytes, '\0');
    char *dest = &(*centroids)[0];
    for (size_t m = 0; m < chunk_num; ++m) {
      const size_t start = offsets[m];
      const size_t width = offsets[m + 1] - start;
      for (size_t c = 0; c < kClusterNum; ++c) {
        std::memcpy(dest + (kClusterNum * start + c * width) * unit,
                    pivots + (c * dim + start) * unit, width * unit);
      }
    }

    return 0;
  }

  //! Construct the quantizer persisted in `meta_buffer`, picking the
  //! implementation from the meta buffer header instead of a hardcoded type.
  //! Contract: the quantizer is initialized with the meta derived from
  //! `index_meta` before deserialize(), so the metric policy comes from the
  //! meta instead of the default-constructed one. The layout comes from the
  //! DiskAnn header; legacy codebooks are adopted after repacking.
  static turbo::Quantizer::Pointer create_quantizer_from_meta_buffer(
      std::string &meta_buffer, const IndexMeta &index_meta, uint32_t chunk_num,
      bool legacy) {
    const char *name = legacy ? "PqInt8Quantizer"
                              : quantizer_name_from_meta_buffer(meta_buffer);
    if (name == nullptr) {
      LOG_ERROR("Unsupported or corrupted quantizer meta buffer");
      return turbo::Quantizer::Pointer();
    }
    auto quantizer = IndexFactory::CreateQuantizer(name);
    if (!quantizer) {
      LOG_ERROR("Create quantizer %s failed", name);
      return turbo::Quantizer::Pointer();
    }
    IndexMeta init_meta;
    if (candidate_quantizer_meta(index_meta, &init_meta) != 0) {
      return turbo::Quantizer::Pointer();
    }
    const bool rabitq = std::strcmp(name, "RabitqQuantizer") == 0;
    if (rabitq != (index_meta.quantizer_name() == "RabitqQuantizer"))
      return nullptr;
    ailego::Params params =
        rabitq ? index_meta.quantizer_params() : ailego::Params{};
    if (!rabitq) params.set("num_chunk", chunk_num);
    if (quantizer->init(init_meta, params) != 0) {
      LOG_ERROR("Quantizer %s init failed", name);
      return turbo::Quantizer::Pointer();
    }
    if (quantizer->quantized_datapoint_vector_length() != chunk_num) {
      LOG_ERROR("DiskAnn candidate code stride does not match quantizer");
      return nullptr;
    }
    if (legacy) {
      std::string centroids;
      if (legacy_pq_codebook_to_centroids(meta_buffer, init_meta, chunk_num,
                                          &centroids) != 0) {
        return turbo::Quantizer::Pointer();
      }
      if (quantizer->import_codebook(centroids.data(), centroids.size()) != 0) {
        LOG_ERROR("Quantizer %s legacy codebook import failed", name);
        return turbo::Quantizer::Pointer();
      }
      return quantizer;
    }
    if (quantizer->deserialize(meta_buffer) != 0) {
      LOG_ERROR("Quantizer %s deserialize failed", name);
      return turbo::Quantizer::Pointer();
    }
    if (rabitq && quantizer->require_train()) return nullptr;
    return quantizer;
  }

 private:
  int create() {
    quantizer_ = IndexFactory::CreateQuantizer(name_);
    if (!quantizer_) return IndexError_NoExist;
    return quantizer_->init(meta_, settings_);
  }

  std::string name_;
  ailego::Params settings_;
  IndexMeta meta_;
  uint32_t max_pq_chunks_{std::numeric_limits<uint32_t>::max()};
  uint32_t code_size_{0};
  turbo::Quantizer::Pointer quantizer_;
};

}  // namespace core
}  // namespace zvec
