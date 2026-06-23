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

#include "matrix_rotator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace zvec {
namespace core {

namespace {

// Generate a dim x dim random Gaussian matrix (row-major) without Eigen.
void random_gaussian_matrix(float *mat, size_t dim) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < dim * dim; ++i) {
    mat[i] = dist(gen);
  }
}

// Householder QR decomposition: A = Q * R.
// Computes the orthogonal matrix Q from input matrix A (row-major, dim x dim).
// Result is stored in q (row-major, dim x dim).
//
// Implemented manually to avoid rabitqlib/Eigen dependency whose ISA-sensitive
// inline functions cause ODR violations (duplicate codegen with different
// -march flags) leading to SEGFAULT on linux-x64-clang.
void householder_qr(const float *A_in, float *q, size_t dim) {
  // R starts as a copy of A
  std::vector<float> R(A_in, A_in + dim * dim);

  // Q starts as identity
  std::fill(q, q + dim * dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) {
    q[i * dim + i] = 1.0f;
  }

  std::vector<float> v(dim);

  for (size_t k = 0; k < dim; ++k) {
    // x = R[k:dim, k]  (sub-column below and including diagonal)
    float norm_x_sq = 0.0f;
    for (size_t i = k; i < dim; ++i) {
      norm_x_sq += R[i * dim + k] * R[i * dim + k];
    }
    if (norm_x_sq == 0.0f) continue;

    float norm_x = std::sqrt(norm_x_sq);

    // alpha = -sign(R[k][k]) * ||x||  (choose sign to avoid cancellation)
    float alpha = (R[k * dim + k] >= 0.0f) ? -norm_x : norm_x;

    // v = x - alpha * e1  (only the sub-vector [k, dim) is non-zero)
    for (size_t i = k; i < dim; ++i) {
      v[i - k] = R[i * dim + k];
    }
    v[0] -= alpha;

    // Normalize v
    float v_norm_sq = 0.0f;
    for (size_t i = 0; i < dim - k; ++i) {
      v_norm_sq += v[i] * v[i];
    }
    if (v_norm_sq == 0.0f) continue;
    float inv_v_norm = 1.0f / std::sqrt(v_norm_sq);
    for (size_t i = 0; i < dim - k; ++i) {
      v[i] *= inv_v_norm;
    }

    // Apply Householder reflection to R: R[k:dim, k:dim] -= 2*v*(v^T * R)
    for (size_t j = k; j < dim; ++j) {
      float dot = 0.0f;
      for (size_t i = 0; i < dim - k; ++i) {
        dot += v[i] * R[(k + i) * dim + j];
      }
      dot *= 2.0f;
      for (size_t i = 0; i < dim - k; ++i) {
        R[(k + i) * dim + j] -= v[i] * dot;
      }
    }

    // Accumulate Q: Q[:, k:dim] -= 2*(Q[:, k:dim] * v) * v^T
    for (size_t i = 0; i < dim; ++i) {
      float dot = 0.0f;
      for (size_t j = 0; j < dim - k; ++j) {
        dot += q[i * dim + k + j] * v[j];
      }
      dot *= 2.0f;
      for (size_t j = 0; j < dim - k; ++j) {
        q[i * dim + k + j] -= dot * v[j];
      }
    }
  }
}

}  // anonymous namespace

void MatrixRotatorImpl::init(size_t dim) {
  // Generate dim x dim random Gaussian matrix
  std::vector<float> rand_mat(dim * dim);
  random_gaussian_matrix(rand_mat.data(), dim);

  // Householder QR: A = Q * R, use Q^T as the rotation matrix
  std::vector<float> Q(dim * dim);
  householder_qr(rand_mat.data(), Q.data(), dim);

  // Store Q^T (transpose) as the rotation matrix
  matrix.resize(dim * dim);
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      matrix[j * dim + i] = Q[i * dim + j];
    }
  }
}

void MatrixRotatorImpl::rotate(const float *in, float *out, size_t dim) const {
  // out = in * matrix  (1 x dim) * (dim x dim) -> (1 x dim)
  for (size_t j = 0; j < dim; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      sum += in[i] * matrix[i * dim + j];
    }
    out[j] = sum;
  }
}

void MatrixRotatorImpl::unrotate(const float *in, float *out,
                                 size_t dim) const {
  // out = in * matrix^T  (1 x dim) * (dim x dim)^T -> (1 x dim)
  for (size_t j = 0; j < dim; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      sum += in[i] * matrix[j * dim + i];
    }
    out[j] = sum;
  }
}

void MatrixRotatorImpl::save(char *data) const {
  std::memcpy(data, matrix.data(), matrix.size() * sizeof(float));
}

void MatrixRotatorImpl::load(const char *data) {
  std::memcpy(matrix.data(), data, matrix.size() * sizeof(float));
}

size_t MatrixRotatorImpl::dump_bytes() const {
  return matrix.size() * sizeof(float);
}

}  // namespace core
}  // namespace zvec
