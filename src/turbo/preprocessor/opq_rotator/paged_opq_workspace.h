// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cmath>
#include <random>
#include <rabitqlib/third/Eigen/Dense>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include "core/utility/paged_build_buffer.h"
#include "turbo/training_workspace.h"

namespace zvec::turbo {

// A small tile, not an Eigen dimension-squared allocation, is the unit of
// scratch. All matrices share one pageable backing file.
class PagedOpqWorkspace {
 public:
  static constexpr size_t kTile = 16;
  using Tile = Eigen::Matrix<double, kTile, kTile, Eigen::RowMajor>;
  static_assert(sizeof(Tile) == kTile * kTile * sizeof(double),
                "Packed tile required");
  explicit PagedOpqWorkspace(size_t dim)
      : dim_(dim), tiles_((dim + kTile - 1) / kTile) {}
  int init(size_t matrices) {
    const size_t max = std::numeric_limits<size_t>::max();
    if (!dim_ || !matrices || tiles_ > max / tiles_ / sizeof(Tile) / matrices)
      return core::IndexError_InvalidArgument;
    stride_ = tiles_ * tiles_ * sizeof(Tile);
    int ret = data_.init(matrices * stride_, 64 * 1024);
    if (ret) return ret;
    Tile zero = Tile::Zero();
    for (size_t offset = 0; offset < data_.size(); offset += sizeof(Tile)) {
      ret = data_.write(offset, zero.data(), sizeof(Tile));
      if (ret) return ret;
    }
    return 0;
  }
  int tile(size_t matrix, size_t row, size_t column, Tile *value,
           bool write = false) {
    const size_t offset =
        matrix * stride_ + (row * tiles_ + column) * sizeof(Tile);
    return write ? data_.write(offset, value->data(), sizeof(Tile))
                 : data_.read(offset, value->data(), sizeof(Tile));
  }
  int line(size_t matrix, size_t index, bool column, double *values,
           bool write = false) {
    Tile block;
    for (size_t i = 0; i < tiles_; ++i) {
      size_t r = column ? i : index / kTile, c = column ? index / kTile : i;
      int ret = tile(matrix, r, c, &block);
      if (ret) return ret;
      size_t count = std::min(kTile, dim_ - i * kTile);
      for (size_t j = 0; j < count; ++j) {
        double &entry =
            column ? block(j, index % kTile) : block(index % kTile, j);
        if (write)
          entry = values[i * kTile + j];
        else
          values[i * kTile + j] = entry;
      }
      if (write && (ret = tile(matrix, r, c, &block, true))) return ret;
    }
    return 0;
  }
  int identity(size_t matrix) {
    for (size_t i = 0; i < tiles_; ++i) {
      Tile block = Tile::Identity();
      int ret = tile(matrix, i, i, &block, true);
      if (ret) return ret;
    }
    return 0;
  }
  int rotation(size_t matrix, size_t p, size_t q, bool rows,
               const Eigen::JacobiRotation<double> &rotation) {
    // Eigen's left/right operations use opposite signs for the sine.
    const double c = rotation.c(), s = rows ? rotation.s() : -rotation.s();
    Tile a, b;
    for (size_t i = 0; i < tiles_; ++i) {
      size_t ar = rows ? p / kTile : i, ac = rows ? i : p / kTile;
      size_t br = rows ? q / kTile : i, bc = rows ? i : q / kTile;
      int ret = tile(matrix, ar, ac, &a);
      if (ret) return ret;
      bool same = ar == br && ac == bc;
      if (!same && (ret = tile(matrix, br, bc, &b))) return ret;
      Tile &other = same ? a : b;
      for (size_t j = 0; j < std::min(kTile, dim_ - i * kTile); ++j) {
        double &x = rows ? a(p % kTile, j) : a(j, p % kTile);
        double &y = rows ? other(q % kTile, j) : other(j, q % kTile);
        double old = x;
        x = c * old + s * y;
        y = -s * old + c * y;
      }
      if ((ret = tile(matrix, ar, ac, &a, true))) return ret;
      if (!same && (ret = tile(matrix, br, bc, &b, true))) return ret;
    }
    return 0;
  }
  int value(size_t matrix, size_t row, size_t col, double *out) {
    Tile block;
    int ret = tile(matrix, row / kTile, col / kTile, &block);
    if (!ret) *out = block(row % kTile, col % kTile);
    return ret;
  }
  int random_rotation(uint64_t seed, core::PagedBuildBuffer *output) {
    int ret = init(2);
    if (ret) return ret;
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> distribution(0, 1);
    std::vector<double, ailego::BudgetedAllocator<double>> row(dim_), v(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      for (auto &x : row) x = distribution(generator);
      if ((ret = line(0, i, false, row.data(), true))) return ret;
    }
    if ((ret = identity(1))) return ret;
    for (size_t k = 0; k < dim_; ++k) {
      if ((ret = line(0, k, true, v.data()))) return ret;
      double tail_norm = 0;
      for (size_t i = k + 1; i < dim_; ++i)
        tail_norm = std::hypot(tail_norm, v[i]);
      // Match Eigen's identity reflector for a zero tail (in particular the
      // final scalar pivot); reflecting that pivot would flip the last column.
      if (!tail_norm) continue;
      const double norm = std::hypot(v[k], tail_norm);
      v[k] += std::copysign(norm, v[k]);
      double length = 0;
      for (size_t i = k; i < dim_; ++i) length = std::hypot(length, v[i]);
      for (size_t i = k; i < dim_; ++i) v[i] /= length;
      for (size_t c = k; c < dim_; ++c) {
        if ((ret = line(0, c, true, row.data()))) return ret;
        double dot = 0;
        for (size_t i = k; i < dim_; ++i) dot += v[i] * row[i];
        for (size_t i = k; i < dim_; ++i) row[i] -= 2 * dot * v[i];
        if ((ret = line(0, c, true, row.data(), true))) return ret;
      }
      for (size_t r = 0; r < dim_; ++r) {
        if ((ret = line(1, r, false, row.data()))) return ret;
        double dot = 0;
        for (size_t i = k; i < dim_; ++i) dot += v[i] * row[i];
        for (size_t i = k; i < dim_; ++i) row[i] -= 2 * dot * v[i];
        if ((ret = line(1, r, false, row.data(), true))) return ret;
      }
    }
    ret = output->init(dim_ * dim_ * sizeof(float), 64 * 1024);
    if (ret) return ret;
    std::vector<float, ailego::BudgetedAllocator<float>> result(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      if ((ret = line(1, i, false, row.data()))) return ret;
      std::copy(row.begin(), row.end(), result.begin());
      if ((ret = output->write(i * dim_ * sizeof(float), result.data(),
                               dim_ * sizeof(float))))
        return ret;
    }
    return 0;
  }

  template <typename ReadPairs>
  int fit(size_t num, const ReadPairs &read, core::PagedBuildBuffer *output) {
    int ret = init(3);  // covariance/work, U, V
    if (ret) return ret;
    const size_t batch =
        TrainingTileRows(std::min<size_t>(num, 1024), dim_ * sizeof(float), 2);
    std::vector<float, ailego::BudgetedAllocator<float>> x(batch * dim_),
        y(x.size());
    using Rows =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Tile block;
    for (size_t first = 0; first < num; first += batch) {
      size_t count = std::min(batch, num - first);
      if ((ret = read(first, count, x.data(), y.data()))) return ret;
      Eigen::Map<const Rows> a(x.data(), count, dim_), b(y.data(), count, dim_);
      if (!a.allFinite() || !b.allFinite())
        return core::IndexError_InvalidArgument;
      for (size_t i = 0; i < tiles_; ++i)
        for (size_t j = 0; j < tiles_; ++j) {
          if ((ret = tile(0, i, j, &block))) return ret;
          size_t rows = std::min(kTile, dim_ - i * kTile),
                 cols = std::min(kTile, dim_ - j * kTile);
          block.topLeftCorner(rows, cols).noalias() +=
              a.middleCols(i * kTile, rows)
                  .template cast<double>()
                  .transpose() *
              b.middleCols(j * kTile, cols).template cast<double>();
          if ((ret = tile(0, i, j, &block, true))) return ret;
        }
    }
    double scale = 0;
    for (size_t i = 0; i < tiles_; ++i)
      for (size_t j = 0; j < tiles_; ++j) {
        if ((ret = tile(0, i, j, &block))) return ret;
        scale = std::max(scale, block.cwiseAbs().maxCoeff());
      }
    if (!std::isfinite(scale)) return core::IndexError_InvalidArgument;
    if (!scale) scale = 1;
    double max_diag = 0;
    for (size_t i = 0; i < tiles_; ++i)
      for (size_t j = 0; j < tiles_; ++j) {
        if ((ret = tile(0, i, j, &block))) return ret;
        block /= scale;
        if (i == j)
          max_diag = std::max(max_diag, block.diagonal().cwiseAbs().maxCoeff());
        if ((ret = tile(0, i, j, &block, true))) return ret;
      }
    if ((ret = identity(1)) || (ret = identity(2))) return ret;
    bool finished = false;
    for (size_t sweep = 0; sweep < 128 && !finished; ++sweep) {
      finished = true;
      for (size_t p = 1; p < dim_; ++p)
        for (size_t q = 0; q < p; ++q) {
          Eigen::Matrix2d pair;
          if ((ret = value(0, p, p, &pair(0, 0))) ||
              (ret = value(0, p, q, &pair(0, 1))) ||
              (ret = value(0, q, p, &pair(1, 0))) ||
              (ret = value(0, q, q, &pair(1, 1))))
            return ret;
          const double threshold =
              std::max(std::numeric_limits<double>::min(),
                       2 * std::numeric_limits<double>::epsilon() * max_diag);
          if (std::abs(pair(0, 1)) <= threshold &&
              std::abs(pair(1, 0)) <= threshold)
            continue;
          finished = false;
          Eigen::JacobiRotation<double> left, right;
          Eigen::internal::real_2x2_jacobi_svd(pair, 0, 1, &left, &right);
          if ((ret = rotation(0, p, q, true, left)) ||
              (ret = rotation(1, p, q, false, left.transpose())) ||
              (ret = rotation(0, p, q, false, right)) ||
              (ret = rotation(2, p, q, false, right)))
            return ret;
          pair.applyOnTheLeft(0, 1, left);
          pair.applyOnTheRight(0, 1, right);
          max_diag = std::max(max_diag, pair.diagonal().cwiseAbs().maxCoeff());
        }
    }
    if (!finished) return core::IndexError_Runtime;
    // Sign normalization is needed; sorting singular values is not: the same
    // permutation on U and V cancels in V * U^T, including zero singular
    // values.
    std::vector<double, ailego::BudgetedAllocator<double>> signs(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      double diagonal = 0;
      if ((ret = value(0, i, i, &diagonal))) return ret;
      signs[i] = diagonal < 0 ? -1 : 1;
    }
    if ((ret = output->init(dim_ * dim_ * sizeof(float), 64 * 1024)))
      return ret;
    Tile u, v, result;
    for (size_t i = 0; i < tiles_; ++i)
      for (size_t j = 0; j < tiles_; ++j) {
        result.setZero();
        for (size_t k = 0; k < tiles_; ++k) {
          if ((ret = tile(2, i, k, &v)) || (ret = tile(1, j, k, &u)))
            return ret;
          for (size_t c = 0; c < kTile; ++c)
            u.col(c) *= k * kTile + c < dim_ ? signs[k * kTile + c] : 0;
          result.noalias() += v * u.transpose();
        }
        std::array<float, kTile> row{};
        size_t cols = std::min(kTile, dim_ - j * kTile);
        for (size_t r = 0; r < std::min(kTile, dim_ - i * kTile); ++r) {
          for (size_t c = 0; c < cols; ++c)
            row[c] = static_cast<float>(result(r, c));
          ret = output->write(
              ((i * kTile + r) * dim_ + j * kTile) * sizeof(float), row.data(),
              cols * sizeof(float));
          if (ret) return ret;
        }
      }
    return 0;
  }

 private:
  size_t dim_, tiles_, stride_{0};
  core::PagedBuildBuffer data_;
};
}  // namespace zvec::turbo
