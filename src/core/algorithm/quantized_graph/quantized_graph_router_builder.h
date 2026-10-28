// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
#include "quantized_graph_router_format.h"

namespace zvec::core {

// Offline spherical k-means (cosine) or ordinary k-means (squared L2).
// Leaves are GRAPH INTERNAL IDs, never external keys or training-row offsets.
// The training algorithm matches train_router59.py, but mt19937_64 and the
// native dot-product reduction do not reproduce NumPy PCG64/BLAS bit for bit.
class QuantizedGraphRouterBuilder {
 public:
  static constexpr size_t kDim = 768, kProjected = 384, kWidth = 16;
  static constexpr size_t kMaxAnchors = 65536;
  using Vector = std::vector<float>;
  struct Data {
    uint32_t docs = 0;
    Vector mean = Vector(kDim);
    bool cosine = true;
    std::vector<uint32_t> ids;
    std::vector<Vector> vectors;
  };
  // Read vectors in internal-ID order to preserve deterministic anchors and
  // full-corpus mean accumulation. The callback returns dimension floats,
  // valid until its next call, or nullptr for an unreadable/deleted vector.
  template <class ReadVector>
  static Data collect(uint64_t docs, size_t dimension, bool cosine,
                      size_t anchors, ReadVector read_vector) {
    if (!docs || docs > uint64_t(INT32_MAX) || !dimension ||
        dimension > QuantizedGraphRouterFormat::kMaxDimension || !anchors ||
        anchors > kMaxAnchors)
      throw std::invalid_argument("invalid router sampling shape");
    Data data;
    data.docs = docs;
    data.cosine = cosine;
    data.mean.resize(dimension);
    anchors = std::min(anchors, size_t(docs));
    data.ids.resize(anchors);
    data.vectors.assign(anchors, Vector(dimension));
    // Preserve the historical Cohere sequence; a coprime stride keeps IDs
    // unique for other corpus sizes too.
    uint64_t stride = 2654435761ULL;
    while (std::gcd(stride, docs) != 1) ++stride;
    std::vector<int32_t> selected(docs, -1);
    for (size_t i = 0; i < anchors; ++i) {
      const uint32_t id = (uint64_t(i) * stride + 7193) % docs;
      data.ids[i] = id;
      selected[id] = i;
    }
    std::vector<double> sum(dimension);
    for (uint32_t id = 0; id < docs; ++id) {
      const float *v = read_vector(id);
      if (!v) throw std::domain_error("unreadable router training vector");
      double norm = 0;
      for (size_t d = 0; d < dimension; ++d) {
        if (!std::isfinite(v[d]))
          throw std::domain_error("nonfinite router training vector");
        sum[d] += v[d];
        norm += double(v[d]) * v[d];
      }
      if (cosine && std::abs(std::sqrt(norm) - 1.) > 1e-4)
        throw std::domain_error("router training vectors must be normalized");
      if (selected[id] >= 0)
        std::copy_n(v, dimension, data.vectors[selected[id]].begin());
    }
    for (size_t d = 0; d < dimension; ++d) data.mean[d] = float(sum[d] / docs);
    return data;
  }
  struct Options {
    uint64_t seed = 202610261259ULL;
    size_t iterations = 10;
    size_t projected = 0;  // Zero selects half the original dimension.
  };
  struct Group {
    std::array<Vector, kWidth> centers{};
    std::array<int32_t, kWidth> targets{};
    uint32_t count = 0;
    Group() {
      targets.fill(INT32_MIN);
    }
  };
  struct Model {
    uint32_t docs = 0;
    Vector mean = Vector(kDim);
    bool cosine = true;
    std::vector<Group> groups;
    size_t max_depth = 0, balanced_splits = 0, anchors = 0;
    size_t projected = kProjected;

    // V2 for the legacy 768/384 cosine layout, otherwise V3. The caller
    // owns publication/overwrite policy; stream failures are never ignored.
    void write(std::ostream &out, size_t projection = 0) const {
      const size_t dim = mean.size();
      if (projection == 0) projection = projected;
      turbo::SymphonyQGRotation rotate(dim);
      const size_t rotated_dim = rotate.padded_dim();
      if (projection == 0 || projection > rotated_dim)
        throw std::invalid_argument("invalid router projection");
      const size_t stride = (projection + 63) & ~size_t{63};
      const uint32_t version =
          dim == 768 && projection == 384 && cosine ? 2 : 3;
      const uint32_t endian = 1;
      if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
        throw std::runtime_error("router writer requires little-endian host");
      if (groups.empty() || groups.size() > 100000)
        throw std::runtime_error("invalid router group count");
      auto put = [&](const void *p, size_t n) {
        out.write(static_cast<const char *>(p), n);
        if (!out) throw std::runtime_error("writing router model failed");
      };
      const uint32_t header[]{0x51475259, version, uint32_t(projection),
                              uint32_t(groups.size()), docs};
      const uint32_t tag = 0x53514701;
      std::vector<float> rotated_mean;
      rotate.rotate(mean.data(), rotated_mean);
      put(header, sizeof(header));
      if (version == 3) {
        const uint32_t extra[]{uint32_t(dim), cosine ? 0u : 1u,
                               uint32_t(rotated_dim), uint32_t(stride)};
        put(extra, sizeof(extra));
      }
      put(mean.data(), dim * sizeof(float));
      put(&tag, sizeof(tag));
      put(rotated_mean.data(), rotated_dim * sizeof(float));
      for (const auto &group : groups) {
        std::array<float, kWidth> scales{}, offsets{}, mean_dot{};
        std::vector<uint8_t> codes(kWidth * stride);
        const Vector zero(dim);
        for (size_t j = 0; j < kWidth; ++j) {
          std::vector<float> rotated;
          const auto &center = j < group.count ? group.centers[j] : zero;
          rotate.rotate(center.data(), rotated);
          double dot = 0;
          for (size_t d = 0; d < rotated_dim; ++d)
            rotated[d] -= rotated_mean[d];
          for (size_t d = 0; d < dim; ++d) {
            const double residual = double(center[d]) - mean[d];
            dot += cosine ? double(mean[d]) * center[d] : residual * residual;
          }
          mean_dot[j] = float(dot);
          auto range = std::minmax_element(rotated.begin(),
                                           rotated.begin() + projection);
          offsets[j] = *range.first;
          scales[j] = (*range.second - *range.first) / 255.f;
          const float inverse = scales[j] > 0 ? 1.f / scales[j] : 0.f;
          for (size_t d = 0; d < projection; ++d) {
            const float code = (rotated[d] - offsets[j]) * inverse;
            codes[j * stride + d] =
                uint8_t(std::clamp(std::nearbyint(code), 0.f, 255.f));
          }
        }
        put(&group.count, sizeof(group.count));
        put(group.targets.data(), sizeof(group.targets));
        put(scales.data(), sizeof(scales));
        put(offsets.data(), sizeof(offsets));
        put(mean_dot.data(), sizeof(mean_dot));
        put(codes.data(), codes.size());
      }
    }
  };

  static void validate(const Data &data) {
    const size_t dim = data.mean.size();
    if (dim == 0 || dim > QuantizedGraphRouterFormat::kMaxDimension)
      throw std::invalid_argument("router dimension must be in [1, 4096]");
    if (data.docs == 0 || data.docs > uint32_t(INT32_MAX) || data.ids.empty() ||
        data.ids.size() > kMaxAnchors || data.ids.size() != data.vectors.size())
      throw std::invalid_argument("invalid router training shape");
    for (float v : data.mean)
      if (!std::isfinite(v) || (data.cosine && std::abs(v) > 1.0001f))
        throw std::invalid_argument("invalid normalized corpus mean");
    std::unordered_set<uint32_t> seen;
    for (size_t i = 0; i < data.ids.size(); ++i) {
      if (data.ids[i] >= data.docs || !seen.insert(data.ids[i]).second)
        throw std::invalid_argument("invalid or duplicate anchor internal ID");
      if (data.vectors[i].size() != dim)
        throw std::invalid_argument("anchor dimension mismatch");
      double norm = 0;
      for (float v : data.vectors[i]) {
        if (!std::isfinite(v))
          throw std::invalid_argument("nonfinite anchor vector");
        norm += double(v) * v;
      }
      if (!std::isfinite(norm) ||
          norm > std::numeric_limits<float>::max() / 16.)
        throw std::invalid_argument("anchor norm overflows FP32 distance");
      if (data.cosine && std::abs(std::sqrt(norm) - 1.) > 1e-4)
        throw std::invalid_argument("cosine anchors must be unit FP32 vectors");
    }
  }

  static Model train(const Data &data, const Options &options) {
    validate(data);
    if (options.iterations == 0 || options.iterations > 100)
      throw std::invalid_argument("iterations must be in [1, 100]");
    const size_t projected = options.projected
                                 ? options.projected
                                 : std::max(size_t{1}, data.mean.size() / 2);
    if (projected > turbo::SymphonyQGRotation(data.mean.size()).padded_dim())
      throw std::invalid_argument("projection exceeds rotated dimension");
    QuantizedGraphRouterBuilder builder(data, options);
    builder.model_.projected = projected;
    std::vector<size_t> rows(data.ids.size());
    std::iota(rows.begin(), rows.end(), size_t{0});
    builder.split(rows, 0);
    return std::move(builder.model_);
  }

 private:
  QuantizedGraphRouterBuilder(const Data &data, const Options &options)
      : data_(data), options_(options), rng_(options.seed) {
    model_.docs = data.docs;
    model_.mean = data.mean;
    model_.cosine = data.cosine;
    model_.anchors = data.ids.size();
  }

  // Rejection sampling rather than implementation-defined distributions.
  size_t random_below(size_t n) {
    const uint64_t bound = uint64_t(n), threshold = -bound % bound;
    uint64_t value;
    do {
      value = rng_();
    } while (value < threshold);
    return value % bound;
  }

  Vector center(const std::vector<size_t> &rows) const {
    const size_t dim = data_.mean.size();
    std::vector<double> sum(dim);
    for (size_t row : rows)
      for (size_t d = 0; d < dim; ++d) sum[d] += data_.vectors[row][d];
    Vector result(dim);
    double norm = 0;
    for (size_t d = 0; d < dim; ++d) {
      result[d] = float(sum[d] / rows.size());
      norm += double(result[d]) * result[d];
    }
    const float scale = float(std::max(std::sqrt(norm), 1e-30));
    if (data_.cosine)
      for (float &v : result) v /= scale;
    return result;
  }

  std::vector<std::vector<size_t>> assign(
      const std::vector<size_t> &rows,
      const std::vector<Vector> &centers) const {
    std::vector<std::vector<size_t>> parts(centers.size());
    for (size_t row : rows) {
      size_t best = 0;
      float best_dot = -std::numeric_limits<float>::infinity();
      for (size_t j = 0; j < centers.size(); ++j) {
        float dot;
        if (data_.cosine) {
          ailego::InnerProductMatrix<float, 1, 1>::Compute(
              data_.vectors[row].data(), centers[j].data(), data_.mean.size(),
              &dot);
        } else {
          ailego::SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
              data_.vectors[row].data(), centers[j].data(), data_.mean.size(),
              &dot);
          dot = -dot;
        }
        if (dot > best_dot) {
          best = j;
          best_dot = dot;
        }
      }
      parts[best].push_back(row);
    }
    parts.erase(std::remove_if(parts.begin(), parts.end(),
                               [](const auto &p) { return p.empty(); }),
                parts.end());
    return parts;
  }

  size_t split(const std::vector<size_t> &rows, size_t depth) {
    const size_t group = model_.groups.size();
    model_.groups.emplace_back();
    model_.max_depth = std::max(model_.max_depth, depth);
    if (rows.size() <= kWidth) {
      auto &g = model_.groups[group];
      g.count = rows.size();
      for (size_t j = 0; j < rows.size(); ++j) {
        g.centers[j] = data_.vectors[rows[j]];
        g.targets[j] = -int32_t(data_.ids[rows[j]]) - 1;
      }
      return group;
    }
    std::vector<size_t> shuffled(rows);
    std::vector<Vector> centers;
    for (size_t j = 0; j < kWidth; ++j) {
      std::swap(shuffled[j], shuffled[j + random_below(rows.size() - j)]);
      centers.push_back(data_.vectors[shuffled[j]]);
    }
    for (size_t i = 0; i < options_.iterations; ++i) {
      auto parts = assign(rows, centers);
      centers.clear();
      for (const auto &part : parts) centers.push_back(center(part));
    }
    auto parts = assign(rows, centers);
    // As in Python, degenerate splits are balanced contiguous partitions.
    // Also bound pathological chains: with <=65536 anchors, balancing from
    // depth 11 reaches leaves by depth 14, within the loader's <16 limit.
    if (parts.size() < 2 || depth >= 11) {
      ++model_.balanced_splits;
      parts.assign(kWidth, {});
      size_t start = 0;
      for (size_t j = 0; j < kWidth; ++j) {
        size_t count = rows.size() / kWidth + (j < rows.size() % kWidth);
        parts[j].assign(rows.begin() + start, rows.begin() + start + count);
        start += count;
      }
    }
    model_.groups[group].count = parts.size();
    for (size_t j = 0; j < parts.size(); ++j) {
      const auto &part = parts[j];
      model_.groups[group].centers[j] =
          part.size() == 1 ? data_.vectors[part[0]] : center(part);
      // Recursive push_back may reallocate groups_: never keep a reference.
      const int32_t target = part.size() == 1 ? -int32_t(data_.ids[part[0]]) - 1
                                              : int32_t(split(part, depth + 1));
      model_.groups[group].targets[j] = target;
    }
    return group;
  }

  const Data &data_;
  Options options_;
  std::mt19937_64 rng_;
  Model model_;
};
}  // namespace zvec::core
