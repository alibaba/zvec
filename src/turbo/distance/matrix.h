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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>
#include <zvec/ailego/utility/type_helper.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {

template <typename T>
constexpr DataType MatrixDataType() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, ailego::Float16> ||
                    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                    std::is_same_v<T, uint8_t>,
                "Unsupported matrix element type");
  if constexpr (std::is_same_v<T, double>) return DataType::kFp64;
  if constexpr (std::is_same_v<T, int16_t>) return DataType::kInt16;
  if constexpr (std::is_same_v<T, float>) return DataType::kFp32;
  if constexpr (std::is_same_v<T, ailego::Float16>) return DataType::kFp16;
  if constexpr (std::is_same_v<T, int8_t>) return DataType::kInt8;
  // The matrix API stores signed INT4 as packed bytes.
  return DataType::kInt4;
}

// Adapt the interleaved matrix layout used by clustering and IndexMetric to
// Turbo's contiguous vector kernels. INT8 and packed INT4 matrices interleave
// 32-bit words; other types interleave individual elements. Output is
// query-major.
inline DistanceFunc MakeDistanceMatrix(DistanceFunc distance, size_t rows,
                                       size_t columns, DataType type) {
  if (!distance || rows == 0 || columns == 0 || rows > 32 || columns > rows ||
      (rows & (rows - 1)) || (columns & (columns - 1)))
    return {};
  if (rows == 1 && columns == 1) return distance;
  return [=](const void *m, const void *q, size_t dim, float *out) {
    const size_t bytes = type == DataType::kFp64    ? dim * 8
                         : type == DataType::kInt16 ? dim * 2
                         : type == DataType::kFp32  ? dim * 4
                         : type == DataType::kFp16  ? dim * 2
                         : type == DataType::kInt4  ? (dim + 1) / 2
                                                    : dim;
    const size_t unit = type == DataType::kFp64 ? 8
                        : (type == DataType::kFp16 || type == DataType::kInt16)
                            ? 2
                            : 4;
    ailego_assert(bytes % unit == 0);
    const size_t units = bytes / unit;
    std::vector<uint32_t> matrix((bytes * rows + 3) / 4);
    std::vector<uint32_t> query((bytes + 3) / 4);
    auto *data = reinterpret_cast<char *>(matrix.data());
    for (size_t i = 0; i < rows; ++i) {
      for (size_t d = 0; d < units; ++d) {
        std::memcpy(data + i * bytes + d * unit,
                    static_cast<const char *>(m) + (d * rows + i) * unit, unit);
      }
    }
    for (size_t j = 0; j < columns; ++j) {
      const void *query_data = q;
      if (columns != 1) {
        for (size_t d = 0; d < units; ++d) {
          std::memcpy(reinterpret_cast<char *>(query.data()) + d * unit,
                      static_cast<const char *>(q) + (d * columns + j) * unit,
                      unit);
        }
        query_data = query.data();
      }
      for (size_t i = 0; i < rows; ++i) {
        distance(data + i * bytes, query_data, dim, out + j * rows + i);
      }
    }
  };
}

// Raw matrix kernels consume the clustering layout directly. Keeping the row
// count static lets the compiler vectorize across rows without a transpose,
// heap allocations, or an indirect distance call for every pair.
template <typename T, size_t Rows, MetricType Metric>
struct InterleavedDistanceMatrix {
  static_assert(Metric == MetricType::kSquaredEuclidean ||
                    Metric == MetricType::kInnerProduct,
                "Unsupported raw matrix metric");
  static constexpr bool kOptimizedSingleColumn =
      (std::is_same_v<T, ailego::Float16> &&
       Metric == MetricType::kSquaredEuclidean) ||
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>;
  static auto Value(const T *data, size_t d, size_t row, size_t rows) {
    if constexpr (std::is_same_v<T, uint8_t>) {
      const size_t offset = (d / 8 * rows + row) * 4 + d % 8 / 2;
      const int code = (data[offset] >> (4 * (d % 2))) & 15;
      return float(code < 8 ? code : code - 16);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return float(data[(d / 4 * rows + row) * 4 + d % 4]);
    } else if constexpr (std::is_same_v<T, double> ||
                         (std::is_same_v<T, int16_t> && Rows > 2 &&
                          Metric == MetricType::kInnerProduct)) {
      return data[d * rows + row];
    } else {
      return float(data[d * rows + row]);
    }
  }

  static const auto &PackedDistances() {
    // One entry for each pair of signed nibbles. Squared differences
    // range from 0 to 225; negative products range from -64 to 56.
    using Entry = std::conditional_t<Metric == MetricType::kInnerProduct,
                                     int8_t, uint8_t>;
    static constexpr auto distance_table = [] {
      std::array<Entry, 256> table{};
      for (int a = 0; a < 16; ++a) {
        for (int b = 0; b < 16; ++b) {
          const int lhs = a < 8 ? a : a - 16;
          const int rhs = b < 8 ? b : b - 16;
          if constexpr (Metric == MetricType::kInnerProduct) {
            table[a * 16 + b] = Entry(-lhs * rhs);
          } else {
            table[a * 16 + b] = Entry((lhs - rhs) * (lhs - rhs));
          }
        }
      }
      return table;
    }();
    return distance_table;
  }

  template <size_t Columns>
  static void ComputePackedColumns(const T *m, const T *q, size_t dim,
                                   float *out) {
    const auto &table = PackedDistances();
    const T *q_end = q + dim / 2 * Columns;
    // Keep accumulators disjoint from byte inputs so alias checks do not
    // prevent reuse of each packed word across the query tile.
    float sums[Rows * Columns]{};
    while (q != q_end) {
      for (size_t i = 0; i < Rows; ++i) {
        uint32_t word;
        std::memcpy(&word, m + i * 4, 4);
        for (size_t j = 0; j < Columns; ++j) {
          uint32_t query;
          std::memcpy(&query, q + j * 4, 4);
          int sum = 0;
          for (size_t shift = 0; shift < 32; shift += 4)
            sum += table[((word >> shift) & 15) * 16 + ((query >> shift) & 15)];
          sums[j * Rows + i] += float(sum);
        }
      }
      m += Rows * 4;
      q += Columns * 4;
    }
    std::copy_n(sums, Rows * Columns, out);
  }

  // Reuse a query component (or packed word) across the whole block. A
  // strided query also serves one column of a many-to-many matrix. Keep
  // integer products in word-sized groups before converting to FP32.
  template <bool WithNorms = false>
  static void ComputeSingleColumn(const T *m, const T *q, size_t dim,
                                  float *out, size_t query_stride = 1,
                                  float *row_norms = nullptr,
                                  float *query_norm = nullptr) {
    static_assert(kOptimizedSingleColumn, "Unsupported single-column kernel");
    static_assert(!WithNorms || (std::is_same_v<T, int8_t> &&
                                 Metric == MetricType::kInnerProduct));
    float *dot_result = out, *norm_result = row_norms;
    float *query_norm_result = query_norm;
    float dots[Rows], norms[Rows], qnorm;
    if constexpr (WithNorms) {
      // Disjoint local accumulators let the compiler vectorize dot products
      // and norms together without testing whether output pointers alias.
      out = dots;
      row_norms = norms;
      query_norm = &qnorm;
      std::fill_n(row_norms, Rows, 0.0f);
      *query_norm = 0.0f;
    }
    if constexpr (std::is_same_v<T, ailego::Float16>) {
      if (dim == 0) {
        for (size_t i = 0; i < Rows; ++i) out[i] = 0;
        return;
      }
      const T *q_end = q + dim;
      const T first = *q++;
      for (size_t i = 0; i < Rows; ++i) {
        const auto delta = m[i] - first;
        out[i] = delta * delta;
      }
      m += Rows;
      while (q != q_end) {
        const T query = *q++;
        for (size_t i = 0; i < Rows; ++i) {
          const auto delta = m[i] - query;
          // Match the legacy FP32 product rounding before accumulation.
          const float squared = delta * delta;
          out[i] += squared;
        }
        m += Rows;
      }
    } else {
      for (size_t i = 0; i < Rows; ++i) out[i] = 0;
      size_t d = 0;
      if constexpr (std::is_same_v<T, int8_t>) {
        for (; d + 4 <= dim; d += 4) {
          const auto *query = q + d * query_stride;
          const int q0 = query[0], q1 = query[1], q2 = query[2], q3 = query[3];
          if constexpr (WithNorms)
            *query_norm += float(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
          const auto *block = m + d * Rows;
          for (size_t i = 0; i < Rows; ++i) {
            const auto *row = block + i * 4;
            if constexpr (WithNorms)
              row_norms[i] += float(row[0] * row[0] + row[1] * row[1] +
                                    row[2] * row[2] + row[3] * row[3]);
            if constexpr (Metric == MetricType::kInnerProduct) {
              const int dot =
                  row[0] * q0 + row[1] * q1 + row[2] * q2 + row[3] * q3;
              if constexpr (WithNorms)
                out[i] += float(dot);
              else
                out[i] -= float(dot);
            } else {
              const int d0 = row[0] - q0, d1 = row[1] - q1;
              const int d2 = row[2] - q2, d3 = row[3] - q3;
              out[i] += float(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
            }
          }
        }
      } else {
        const auto &distance_table = PackedDistances();
        for (; d + 8 <= dim; d += 8) {
          uint32_t query;
          std::memcpy(&query, q + d / 2 * query_stride, sizeof(query));
          const auto *block = m + d / 2 * Rows;
          for (size_t i = 0; i < Rows; ++i) {
            uint32_t word;
            std::memcpy(&word, block + i * 4, sizeof(word));
            int sum = 0;
            for (size_t shift = 0; shift < 32; shift += 4) {
              sum += distance_table[((word >> shift) & 15) * 16 +
                                    ((query >> shift) & 15)];
            }
            out[i] += float(sum);
          }
        }
      }
      // Matrices pad a final partial word; the single query need not be
      // padded. Read only its valid bytes/nibbles and ignore matrix padding.
      for (; d < dim; ++d) {
        const float query = Value(q, d, 0, query_stride);
        if constexpr (WithNorms) *query_norm += query * query;
        for (size_t i = 0; i < Rows; ++i) {
          const float value = Value(m, d, i, Rows);
          if constexpr (WithNorms) row_norms[i] += value * value;
          if constexpr (Metric == MetricType::kInnerProduct) {
            if constexpr (WithNorms)
              out[i] += value * query;
            else
              out[i] -= value * query;
          } else {
            const float delta = value - query;
            out[i] += delta * delta;
          }
        }
      }
    }
    if constexpr (WithNorms) {
      for (size_t i = 0; i < Rows; ++i) dot_result[i] = -dots[i];
      std::copy_n(norms, Rows, norm_result);
      *query_norm_result = qnorm;
    }
  }

  // Use the same accumulation order as the matrix inner product, including
  // packed-word groups. This keeps a vector's self MIPS distance at zero.
  static void SquaredNorms(const T *data, size_t dim, float *out) {
    std::fill_n(out, Rows, 0.0f);
    size_t d = 0;
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
      constexpr size_t step = std::is_same_v<T, int8_t> ? 4 : 8;
      for (; d + step <= dim; d += step) {
        for (size_t i = 0; i < Rows; ++i) {
          int sum = 0;
          const auto *word = data + (d / step * Rows + i) * 4;
          if constexpr (std::is_same_v<T, int8_t>) {
            sum = word[0] * word[0] + word[1] * word[1] + word[2] * word[2] +
                  word[3] * word[3];
          } else {
            static constexpr auto squares = [] {
              std::array<uint8_t, 256> table{};
              for (int v = 0; v < 256; ++v) {
                const int lo = ((v & 15) ^ 8) - 8;
                const int hi = ((v >> 4) ^ 8) - 8;
                table[v] = uint8_t(lo * lo + hi * hi);
              }
              return table;
            }();
            sum = squares[word[0]] + squares[word[1]] + squares[word[2]] +
                  squares[word[3]];
          }
          out[i] -= float(sum);
        }
      }
    }
    for (; d < dim; ++d) {
      for (size_t i = 0; i < Rows; ++i) {
        const auto value = Value(data, d, i, Rows);
        out[i] -= value * value;
      }
    }
    for (size_t i = 0; i < Rows; ++i) out[i] = -out[i];
  }

  // Small numeric blocks benefit from reusing each matrix value across
  // queries. A static column count lets the compiler unroll the tiny tile.
  template <size_t Columns>
  static void ComputeSmall(const T *m, const T *q, size_t dim, float *out) {
    float local[Rows * Columns];
    float *result = out;
    if constexpr (std::is_same_v<T, float>) out = local;
    if constexpr (Metric == MetricType::kInnerProduct) {
      // Keep the negative-product reduction across rows. A positive sum
      // followed by negation can induce costly strided dimension reductions.
      const T *q_end = q + dim * Columns;
      if (dim > 0) {
        for (size_t i = 0; i < Rows; ++i) {
          const T value = m[i];
          float *r = out + i;
          for (size_t j = 0; j < Columns; ++j) {
            *r = -float(value * q[j]);
            r += Rows;
          }
        }
        m += Rows;
        q += Columns;
      }
      for (size_t d = 1; Columns == 1 ? q != q_end : d < dim; ++d) {
        for (size_t i = 0; i < Rows; ++i) {
          const T value = m[i];
          float *r = out + i;
          for (size_t j = 0; j < Columns; ++j) {
            *r -= value * q[j];
            r += Rows;
          }
        }
        m += Rows;
        q += Columns;
      }
      if constexpr (std::is_same_v<T, float>)
        std::copy_n(local, Rows * Columns, result);
      return;
    }
    using Arithmetic =
        std::conditional_t<std::is_same_v<T, double>, double, float>;
    for (size_t i = 0; i < Rows; ++i) {
      const auto value = Value(m, 0, i, Rows);
      for (size_t j = 0; j < Columns; ++j) {
        const auto query = Value(q, 0, j, Columns);
        const Arithmetic delta = Arithmetic(value) - Arithmetic(query);
        out[j * Rows + i] = float(delta * delta);
      }
    }
    m += Rows;
    q += Columns;
    for (size_t d = 1; d < dim; ++d) {
      for (size_t i = 0; i < Rows; ++i) {
        const auto value = Value(m, 0, i, Rows);
        for (size_t j = 0; j < Columns; ++j) {
          const auto query = Value(q, 0, j, Columns);
          const Arithmetic delta = Arithmetic(value) - Arithmetic(query);
          const float squared = delta * delta;
          out[j * Rows + i] += squared;
        }
      }
      m += Rows;
      q += Columns;
    }
    if constexpr (std::is_same_v<T, float>)
      std::copy_n(local, Rows * Columns, result);
  }

  template <size_t Columns>
  static void Compute(const T *m, const T *q, size_t dim, float *out) {
    constexpr size_t columns = Columns;
    if (dim == 0) {
      std::fill_n(out, Rows * columns, 0.0f);
      return;
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
    if constexpr (std::is_same_v<T, ailego::Float16>) {
      // Scalar Float16 conversion uses a software lookup on x86. Convert the
      // interleaved buffers in bounded stack tiles with the SIMD converter,
      // then reuse the FP32 kernel without a transpose or heap allocation.
      constexpr size_t tile = 64;
      float matrix[Rows * tile], query[Columns * tile];
      float partial[Rows * Columns];
      for (size_t d = 0; d < dim; d += tile) {
        const size_t count = std::min(tile, dim - d);
        ailego::FloatHelper::ToFP32(
            reinterpret_cast<const uint16_t *>(m + d * Rows), Rows * count,
            matrix);
        ailego::FloatHelper::ToFP32(
            reinterpret_cast<const uint16_t *>(q + d * Columns),
            Columns * count, query);
        InterleavedDistanceMatrix<float, Rows, Metric>::template Compute<
            Columns>(matrix, query, count, d == 0 ? out : partial);
        if (d != 0)
          for (size_t i = 0; i < Rows * Columns; ++i) out[i] += partial[i];
      }
      return;
    }
#endif
    if constexpr (std::is_same_v<T, uint8_t> && Columns > 1) {
      // Reuse each packed matrix word across queries. Partial
      // words retain the bounds-safe single-column implementation below.
      if (dim % 8 == 0) return ComputePackedColumns<Columns>(m, q, dim, out);
    }
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
      for (size_t j = 0; j < columns; ++j)
        ComputeSingleColumn(m, q + j * 4, dim, out + j * Rows, columns);
      return;
    }
    if constexpr (kOptimizedSingleColumn) {
      if (columns == 1) {
        ComputeSingleColumn(m, q, dim, out);
        return;
      }
    }
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      // Retain query-major reductions where they already beat the legacy
      // row/query tile; static tiling must not discard those gains.
      constexpr bool fp16_query_major =
          (Metric == MetricType::kSquaredEuclidean && Rows == 4 &&
           Columns == 4) ||
          (Metric == MetricType::kInnerProduct &&
           ((Rows >= 8 && Columns == Rows) || (Rows == 16 && Columns == 4)));
      constexpr bool small_tile =
          (std::is_same_v<T, ailego::Float16> && columns > 1 && columns <= 16 &&
           !fp16_query_major) ||
          (std::is_same_v<T, double> && Rows <= 16 && columns <= 8) ||
          (std::is_same_v<T, int16_t> && Rows <= 8 &&
           !(Rows == 2 && Columns == 1) &&
           !(Metric == MetricType::kSquaredEuclidean && Rows == 4 &&
             Columns == 4) &&
           !(Metric == MetricType::kInnerProduct && Rows == 8 &&
             Columns == 8)) ||
          (std::is_same_v<T, float> &&
           ((Rows == 2 && columns > 1) ||
            (Rows == 4 && Columns == 2 &&
             Metric == MetricType::kInnerProduct) ||
            (Rows == 16 && columns == 4) ||
            (Rows == 16 && columns == 8 &&
             Metric == MetricType::kInnerProduct) ||
            (Rows == 32 && Columns == 4 &&
             Metric == MetricType::kSquaredEuclidean)));
      if constexpr (small_tile) return ComputeSmall<Columns>(m, q, dim, out);
    }
    for (size_t j = 0; j < columns; ++j) {
      float sums[Rows]{};
      for (size_t d = 0; d < dim; ++d) {
        const auto query = Value(q, d, j, columns);
        for (size_t i = 0; i < Rows; ++i) {
          const auto value = Value(m, d, i, Rows);
          if constexpr (Metric == MetricType::kSquaredEuclidean) {
            using Arithmetic =
                std::conditional_t<std::is_same_v<T, double>, double, float>;
            const Arithmetic delta = Arithmetic(value) - Arithmetic(query);
            // Round each product to FP32 before accumulation, as in the
            // original kernels; avoid a serial FP64 accumulator conversion.
            if constexpr (std::is_same_v<T, double>) {
              const float squared = delta * delta;
              sums[i] += squared;
            } else {
              sums[i] += delta * delta;
            }
          } else {
            sums[i] -= value * query;
          }
        }
      }
      std::memcpy(out + j * Rows, sums, sizeof(sums));
    }
  }
};

template <typename T, size_t Rows, MetricType Metric, size_t Columns>
inline DistanceFunc InterleavedKernel() {
  return [](const void *m, const void *q, size_t dim, float *out) {
    InterleavedDistanceMatrix<T, Rows, Metric>::template Compute<Columns>(
        static_cast<const T *>(m), static_cast<const T *>(q), dim, out);
  };
}

template <typename T, size_t Rows, MetricType Metric>
inline DistanceFunc InterleavedKernel(size_t columns) {
  switch (columns) {
    case 1:
      return InterleavedKernel<T, Rows, Metric, 1>();
    case 2:
      return InterleavedKernel<T, Rows, Metric, 2>();
    case 4:
      return InterleavedKernel<T, Rows, Metric, 4>();
    case 8:
      return InterleavedKernel<T, Rows, Metric, 8>();
    case 16:
      return InterleavedKernel<T, Rows, Metric, 16>();
    case 32:
      return InterleavedKernel<T, Rows, Metric, 32>();
    default:
      return {};
  }
}

template <typename T, MetricType Metric>
inline DistanceFunc SelectInterleavedKernel(size_t rows, size_t columns) {
  switch (rows) {
    case 2:
      return InterleavedKernel<T, 2, Metric>(columns);
    case 4:
      return InterleavedKernel<T, 4, Metric>(columns);
    case 8:
      return InterleavedKernel<T, 8, Metric>(columns);
    case 16:
      return InterleavedKernel<T, 16, Metric>(columns);
    case 32:
      return InterleavedKernel<T, 32, Metric>(columns);
    default:
      return {};
  }
}

template <MetricType Metric>
inline DistanceFunc MakeRawDistanceMatrix(size_t rows, size_t columns,
                                          DataType type) {
  if (rows == 0 || columns == 0 || rows > 32 || columns > rows ||
      (rows & (rows - 1)) || (columns & (columns - 1)))
    return {};
  if (rows == 1) return get_distance_func(Metric, type, QuantizeType::kRaw);
  switch (type) {
    case DataType::kFp32:
      return SelectInterleavedKernel<float, Metric>(rows, columns);
    case DataType::kFp16:
      return SelectInterleavedKernel<ailego::Float16, Metric>(rows, columns);
    case DataType::kFp64:
      return SelectInterleavedKernel<double, Metric>(rows, columns);
    case DataType::kInt16:
      return SelectInterleavedKernel<int16_t, Metric>(rows, columns);
    case DataType::kInt8:
      return SelectInterleavedKernel<int8_t, Metric>(rows, columns);
    case DataType::kInt4:
      return SelectInterleavedKernel<uint8_t, Metric>(rows, columns);
    default:
      return {};
  }
}

template <typename T, size_t M, size_t N, MetricType Metric>
struct DistanceMatrix {
  static void Compute(const T *m, const T *q, size_t dim, float *out) {
    static_assert(
        M > 0 && M <= 32 && N > 0 && N <= M && !(M & (M - 1)) && !(N & (N - 1)),
        "Unsupported matrix shape");
    if constexpr (M == 1) {
      static const auto distance =
          get_distance_func(Metric, MatrixDataType<T>(), QuantizeType::kRaw);
      distance(m, q, dim, out);
    } else {
      InterleavedDistanceMatrix<T, M, Metric>::template Compute<N>(m, q, dim,
                                                                   out);
    }
  }
};

template <typename T, size_t M, size_t N, typename = void>
using SquaredEuclideanDistanceMatrix =
    DistanceMatrix<T, M, N, MetricType::kSquaredEuclidean>;
template <typename T, size_t M, size_t N, typename = void>
using MinusInnerProductMatrix =
    DistanceMatrix<T, M, N, MetricType::kInnerProduct>;

}  // namespace zvec::turbo
