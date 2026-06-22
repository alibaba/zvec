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
#include <cstring>
#include <random>
#include <rabitqlib/third/Eigen/Core>
#include <rabitqlib/third/Eigen/QR>

namespace zvec {
namespace core {

namespace {

template <typename T>
using RowMajorMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;

template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(size_t rows, size_t cols) {
  RowMajorMatrix<T> rand(rows, cols);
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<T> dist(0, 1);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      rand(i, j) = dist(gen);
    }
  }

  return rand;
}

}  // anonymous namespace

void MatrixRotatorImpl::init(size_t dim) {
  // Generate dim x dim random Gaussian matrix
  RowMajorMatrix<float> rand_mat = random_gaussian_matrix<float>(dim, dim);

  // Householder QR: numerically stable orthogonalisation
  Eigen::HouseholderQR<RowMajorMatrix<float>> qr(rand_mat);
  RowMajorMatrix<float> q_inv = qr.householderQ().transpose();

  matrix.resize(dim * dim);
  std::memcpy(matrix.data(), &q_inv(0, 0), sizeof(float) * dim * dim);
}

void MatrixRotatorImpl::rotate(const float *in, float *out, size_t dim) const {
  // v (1 x dim) * M (dim x dim) -> rv (1 x dim)
  ConstRowMajorMatrixMap<float> v(in, 1, dim);
  RowMajorMatrixMap<float> rv(out, 1, dim);
  rv = v * ConstRowMajorMatrixMap<float>(matrix.data(), dim, dim);
}

void MatrixRotatorImpl::unrotate(const float *in, float *out,
                                 size_t dim) const {
  // in (1 x dim) * M^T (dim x dim) -> out (1 x dim)
  ConstRowMajorMatrixMap<float> v(in, 1, dim);
  RowMajorMatrixMap<float> rv(out, 1, dim);
  rv = v * ConstRowMajorMatrixMap<float>(matrix.data(), dim, dim).transpose();
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
