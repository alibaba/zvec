// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "paged_build_buffer.h"

namespace zvec::core {

// Row-major matrix with O(D) application/QR scratch. Both QR matrices are
// pageable; no Eigen/ISA-dependent implementation is linked into core.
class PagedRotationMatrix {
 public:
  using Reader = std::function<int(size_t, void *, size_t)>;

  int load(size_t rows, size_t columns, const Reader &read) try {
    if (!valid(rows, columns)) return IndexError_InvalidArgument;
    PagedBuildBuffer data;
    int ret = data.init(rows * columns * sizeof(float));
    if (ret != 0) return ret;
    ailego::BudgetedVector<float> row(columns);
    for (size_t i = 0; i < rows; ++i) {
      const size_t offset = i * columns * sizeof(float);
      ret = read(offset, row.data(), columns * sizeof(float));
      if (ret != 0) return ret;
      ret = data.write(offset, row.data(), columns * sizeof(float));
      if (ret != 0) return ret;
    }
    data_ = std::move(data);
    rows_ = rows;
    columns_ = columns;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  int generate(size_t rows, size_t columns) try {
    if (!valid(rows, columns)) return IndexError_InvalidArgument;
    // R is transposed, Q is row-major: both Householder updates scan rows.
    PagedBuildBuffer r, q;
    const size_t bytes = columns * columns * sizeof(float);
    int ret = r.init(bytes);
    if (ret != 0) return ret;
    ret = q.init(bytes);
    if (ret != 0) return ret;
    const size_t stride = columns * sizeof(float);
    ailego::BudgetedVector<float> row(columns), v(columns);
    std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for (size_t i = 0; i < columns; ++i) {
      for (float &value : row) value = gaussian(generator);
      if ((ret = r.write(i * stride, row.data(), stride)) != 0) return ret;
      std::fill(row.begin(), row.end(), 0.0f);
      row[i] = 1.0f;
      if ((ret = q.write(i * stride, row.data(), stride)) != 0) return ret;
    }
    for (size_t k = 0; k < columns; ++k) {
      if ((ret = r.read(k * stride, v.data(), stride)) != 0) return ret;
      double norm = 0;
      for (size_t i = k; i < columns; ++i) norm += double(v[i]) * v[i];
      if (norm == 0) continue;
      v[k] += std::copysign(std::sqrt(norm), v[k]);
      norm = 0;
      for (size_t i = k; i < columns; ++i) norm += double(v[i]) * v[i];
      const double scale = 1.0 / std::sqrt(norm);
      for (size_t i = k; i < columns; ++i) v[i] *= scale;
      for (auto *matrix : {&r, &q}) {
        for (size_t j = matrix == &r ? k : 0; j < columns; ++j) {
          if ((ret = matrix->read(j * stride, row.data(), stride)) != 0)
            return ret;
          double dot = 0;
          for (size_t i = k; i < columns; ++i) dot += double(row[i]) * v[i];
          for (size_t i = k; i < columns; ++i) row[i] -= 2 * dot * v[i];
          if ((ret = matrix->write(j * stride, row.data(), stride)) != 0)
            return ret;
        }
      }
    }
    r = PagedBuildBuffer();
    // Transpose Q in bounded tiles, not one storage operation per element.
    constexpr size_t tile = 32;
    std::array<float, tile * tile> a{}, b{}, transposed{};
    for (size_t i = 0; i < columns; i += tile) {
      const size_t height = std::min(tile, columns - i);
      for (size_t j = i; j < columns; j += tile) {
        const size_t width = std::min(tile, columns - j);
        for (size_t y = 0; y < height; ++y) {
          ret = q.read((i + y) * stride + j * sizeof(float),
                       a.data() + y * tile, width * sizeof(float));
          if (ret != 0) return ret;
        }
        if (i != j) {
          for (size_t y = 0; y < width; ++y) {
            ret = q.read((j + y) * stride + i * sizeof(float),
                         b.data() + y * tile, height * sizeof(float));
            if (ret != 0) return ret;
          }
        }
        for (size_t y = 0; y < width; ++y) {
          for (size_t x = 0; x < height; ++x)
            transposed[y * tile + x] = a[x * tile + y];
          ret = q.write((j + y) * stride + i * sizeof(float),
                        transposed.data() + y * tile, height * sizeof(float));
          if (ret != 0) return ret;
        }
        if (i != j) {
          for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x)
              transposed[y * tile + x] = b[x * tile + y];
            ret = q.write((i + y) * stride + j * sizeof(float),
                          transposed.data() + y * tile, width * sizeof(float));
            if (ret != 0) return ret;
          }
        }
      }
    }
    ret = q.truncate(rows * stride);
    if (ret != 0) return ret;
    data_ = std::move(q);
    rows_ = rows;
    columns_ = columns;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  void apply(const float *in, float *out, bool inverse = false) const {
    ailego::BudgetedVector<float> row(columns_);
    ailego::BudgetedVector<float> input;
    if (in == out) {
      input.assign(in, in + (inverse ? columns_ : rows_));
      in = input.data();
    }
    std::fill(out, out + (inverse ? rows_ : columns_), 0.0f);
    for (size_t i = 0; i < rows_; ++i) {
      check(data_.read(i * columns_ * sizeof(float), row.data(),
                       columns_ * sizeof(float)));
      for (size_t j = 0; j < columns_; ++j) {
        if (inverse)
          out[i] += in[j] * row[j];
        else
          out[j] += in[i] * row[j];
      }
    }
  }
  int read(size_t offset, void *out, size_t bytes) const {
    return data_.read(offset, out, bytes);
  }
  size_t size() const {
    return rows_ * columns_ * sizeof(float);
  }
  bool paged() const {
    return data_.paged();
  }
  static void check(int status) {
    if (status == IndexError_NoMemory) throw std::bad_alloc();
    if (status != 0)
      throw std::runtime_error("Rotation matrix storage failure");
  }

 private:
  static bool valid(size_t rows, size_t columns) {
    return rows != 0 && rows <= columns &&
           columns <=
               std::numeric_limits<size_t>::max() / sizeof(float) / columns;
  }
  size_t rows_{0}, columns_{0};
  PagedBuildBuffer data_;
};
}  // namespace zvec::core
