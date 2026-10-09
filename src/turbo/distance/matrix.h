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

template <typename T, size_t M, size_t N, MetricType Metric>
struct DistanceMatrix {
  static void Compute(const T *m, const T *q, size_t dim, float *out) {
    static const auto distance = MakeDistanceMatrix(
        get_distance_func(Metric, MatrixDataType<T>(), QuantizeType::kRaw), M,
        N, MatrixDataType<T>());
    distance(m, q, dim, out);
  }
};

template <typename T, size_t M, size_t N, typename = void>
using SquaredEuclideanDistanceMatrix =
    DistanceMatrix<T, M, N, MetricType::kSquaredEuclidean>;
template <typename T, size_t M, size_t N, typename = void>
using MinusInnerProductMatrix =
    DistanceMatrix<T, M, N, MetricType::kInnerProduct>;

}  // namespace zvec::turbo
