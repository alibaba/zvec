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
  static auto Value(const T *data, size_t d, size_t row, size_t rows) {
    if constexpr (std::is_same_v<T, uint8_t>) {
      const size_t offset = (d / 8 * rows + row) * 4 + d % 8 / 2;
      const int code = (data[offset] >> (4 * (d % 2))) & 15;
      return float(code < 8 ? code : code - 16);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return float(data[(d / 4 * rows + row) * 4 + d % 4]);
    } else if constexpr (std::is_same_v<T, double>) {
      return data[d * rows + row];
    } else {
      return float(data[d * rows + row]);
    }
  }

  static void Compute(const T *m, const T *q, size_t dim, size_t columns,
                      float *out) {
    for (size_t j = 0; j < columns; ++j) {
      float sums[Rows]{};
      for (size_t d = 0; d < dim; ++d) {
        const auto query = Value(q, d, j, columns);
        for (size_t i = 0; i < Rows; ++i) {
          const auto value = Value(m, d, i, Rows);
          if constexpr (Metric == MetricType::kSquaredEuclidean) {
            const auto delta = value - query;
            sums[i] += delta * delta;
          } else {
            sums[i] -= value * query;
          }
        }
      }
      std::memcpy(out + j * Rows, sums, sizeof(sums));
    }
  }
};

template <typename T, size_t Rows, MetricType Metric>
inline DistanceFunc InterleavedKernel(size_t columns) {
  return [columns](const void *m, const void *q, size_t dim, float *out) {
    InterleavedDistanceMatrix<T, Rows, Metric>::Compute(
        static_cast<const T *>(m), static_cast<const T *>(q), dim, columns,
        out);
  };
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
      InterleavedDistanceMatrix<T, M, Metric>::Compute(m, q, dim, N, out);
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
