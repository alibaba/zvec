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
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>
#include <ailego/utility/matrix_helper.h>
#include <zvec/ailego/utility/time_helper.h>

namespace zvec {
namespace core {

#ifndef ivf_check_error_code
#define ivf_check_error_code(code) \
  if (ailego_unlikely((code) != 0)) return code
#endif

#ifndef ivf_assert
#define ivf_assert(cond, code) \
  if (ailego_unlikely(!(cond))) return code
#endif

#ifndef ivf_check_with_msg
#define ivf_check_with_msg(code, fmt, args...) \
  do {                                         \
    if (ailego_unlikely((code) != 0)) {        \
      LOG_ERROR(fmt, ##args);                  \
      return code;                             \
    }                                          \
  } while (0)
#endif

#ifndef ivf_assert_with_msg
#define ivf_assert_with_msg(cond, err, fmt, args...) \
  do {                                               \
    if (ailego_unlikely(!(cond))) {                  \
      LOG_ERROR(fmt, ##args);                        \
      return err;                                    \
    }                                                \
  } while (0)
#endif

/*! Quantized Clustering Utility
 */
class IVFUtility {
 public:
  //! Generator a random path with specificed prefix
  static inline std::string GenerateRandomPath(const std::string &prefix) {
    uint64_t timestamp = ailego::Monotime::MicroSeconds();
    return prefix + std::to_string(timestamp);
  }

  //! Compute the default scan ratio for total vectors
  static inline float ComputeScanRatio(size_t vector_count) {
    // the fitting function for the follow points: 1000000(0.02)
    // 10000000(0.01) 50000000(0.005) 100000000(0.001)
    float scan_ratio = -0.004 * std::log(vector_count) + 0.0751;
    scan_ratio = std::max(scan_ratio, 0.0001f);
    return scan_ratio;
  }

  //! Transpose the vectors in row major order to column major order
  static inline void Transpose(size_t align_size, const void *src, size_t m,
                               size_t dim, void *dst);

  //! Transpose the vectors in column major order to row major order
  static inline void ReverseTranspose(size_t align_size, const void *src,
                                      size_t m, size_t dim, void *dst);

  //! Aligned size of a block vectors buffer
  static inline size_t AlignedSize(size_t fnum, size_t element_size);

  //! Aligned size of one vector buffer
  static inline size_t AlignedSize(size_t element_size);

  //! Sort arr with size in ascending order, and keep the index postion
  //! n2o keep the mapping: new position => origin postion
  //! For example, the input arr = [5, 3, 9, 6, 7], size = 5, after sort
  //      arr = [3, 5, 6, 7, 9]
  //      n2o = [1, 0, 3, 4, 2]
  //! To save memory, no extra memory is allocated
  template <typename T, typename I>
  static void Sort(T *arr, std::vector<I> *n2o, size_t size) {
    std::vector<I> o2n;
    o2n.resize(size);
    n2o->resize(size);

    std::iota(n2o->begin(), n2o->end(), 0U);
    std::sort(n2o->begin(), n2o->end(),
              [&](I i, I j) { return arr[i] < arr[j]; });
    for (I i = 0U; i < size; ++i) {
      o2n[(*n2o)[i]] = i;
    }
    //! reorder arr in place, according to given n2o index
    for (I i = 0; i < size; ++i) {
      if (i != (*n2o)[i]) {
        T tmp = arr[i];
        I j = i, k;
        while (i != (k = (*n2o)[j])) {
          arr[j] = arr[k];
          (*n2o)[j] = j;
          j = k;
        }
        arr[j] = tmp;
        (*n2o)[j] = j;
      }
    }

    for (I i = 0U; i < size; ++i) {
      (*n2o)[o2n[i]] = i;
    }
  }

  //! Transpose one vector in block
  template <typename T>
  static inline void TransposeOne(const void *src, size_t M, size_t N,
                                  void *dst) {
    for (size_t i = 0; i < N; ++i) {
      reinterpret_cast<T *>(dst)[i] = reinterpret_cast<const T *>(src)[i * M];
    }
  }
};

void IVFUtility::Transpose(size_t align_size, const void *src, size_t m,
                           size_t dim, void *dst) {
  switch (align_size) {
    case 2:
      ailego::MatrixHelper::Transpose<uint16_t>(src, m, dim, dst);
      break;
    case 4:
      ailego::MatrixHelper::Transpose<uint32_t>(src, m, dim, dst);
      break;
    case 8:
      ailego::MatrixHelper::Transpose<uint64_t>(src, m, dim, dst);
      break;
  }
}

void IVFUtility::ReverseTranspose(size_t align_size, const void *src, size_t m,
                                  size_t dim, void *dst) {
  switch (align_size) {
    case 2:
      ailego::MatrixHelper::ReverseTranspose<uint16_t>(src, m, dim, dst);
      break;
    case 4:
      ailego::MatrixHelper::ReverseTranspose<uint32_t>(src, m, dim, dst);
      break;
    case 8:
      ailego::MatrixHelper::ReverseTranspose<uint64_t>(src, m, dim, dst);
      break;
  }
}

size_t IVFUtility::AlignedSize(size_t fnum, size_t element_size) {
  return ailego_align(fnum * element_size, 32);
}

size_t IVFUtility::AlignedSize(size_t element_size) {
  return ailego_align(element_size, 32);
}

}  // namespace core
}  // namespace zvec
