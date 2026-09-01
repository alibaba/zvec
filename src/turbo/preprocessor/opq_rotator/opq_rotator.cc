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

#include "opq_rotator.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <random>
// Eigen is used ONLY inside this translation unit: the header exposes no Eigen
// type, and this file is compiled with the baseline ISA flags (it lives outside
// distance/{sse,avx2,avx512*}), so the ISA-sensitive inline functions of Eigen
// cannot be emitted twice with different -march flags.  See the comment in
// src/core/quantizer/rotator/matrix_rotator.cc for the ODR hazard this avoids.
#include <rabitqlib/third/Eigen/Dense>

namespace zvec {
namespace turbo {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstRowMajorMap = Eigen::Map<const RowMajorMatrix>;
using RowMajorMap = Eigen::Map<RowMajorMatrix>;

//! Rows accumulated per block when building the d x d cross-covariance.
//! Blocking keeps the double-precision cast buffer bounded while still
//! accumulating the sum in double (a plain float GEMM over 65k rows loses
//! several digits, which shows up as a non-orthogonal fit result).
constexpr Eigen::Index kAccumBlockRows = 1024;

//! Generate a random orthogonal dim x dim matrix (row-major) as the Q factor
//! of the Householder QR decomposition of a random Gaussian matrix.
void random_orthogonal_matrix(float *out, int dim, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  RowMajorMatrix rand(dim, dim);
  for (Eigen::Index i = 0; i < rand.size(); ++i) {
    rand.data()[i] = dist(gen);
  }

  Eigen::HouseholderQR<RowMajorMatrix> qr(rand);
  RowMajorMap(out, dim, dim) = qr.householderQ();
}

}  // namespace

// ============================================================================
// OpqRotator method implementations
// ============================================================================

OpqRotator::Pointer OpqRotator::create(int dim, uint64_t seed) {
  if (dim <= 0) return nullptr;

  Pointer r(new OpqRotator());
  r->dim_ = dim;
  r->matrix_.resize(static_cast<size_t>(dim) * static_cast<size_t>(dim));
  random_orthogonal_matrix(r->matrix_.data(), dim, seed);
  r->kernels_ = get_rotator_kernels(RotateType::kOpq);
  return r;
}

// ---------------------------------------------------------------------------
// train / fit  (OPQ step 1: fixed codebook -> rotation matrix)
// ---------------------------------------------------------------------------

void OpqRotator::train(const void *data, void *ctx, size_t num, size_t stride) {
  // Packed fp32 (x, x_hat) pairs only; the typed fit() wrapper enforces the
  // same contract with an error code.
  assert(data && ctx && num > 0 && dim_ > 0 && stride == 0);
  (void)stride;  // assert compiles out under NDEBUG
  const float *x = reinterpret_cast<const float *>(data);
  const float *x_hat = reinterpret_cast<const float *>(ctx);

  const Eigen::Index dim = static_cast<Eigen::Index>(dim_);
  const Eigen::Index rows = static_cast<Eigen::Index>(num);

  // M = X^T * X_hat, accumulated in double over row blocks.
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(dim, dim);
  for (Eigen::Index begin = 0; begin < rows; begin += kAccumBlockRows) {
    const Eigen::Index block = std::min(kAccumBlockRows, rows - begin);
    ConstRowMajorMap xb(x + begin * dim, block, dim);
    ConstRowMajorMap xhb(x_hat + begin * dim, block, dim);
    m.noalias() += xb.cast<double>().transpose() * xhb.cast<double>();
  }

  // maximize trace(R * M) subject to R^T R = I  =>  R = V * U^T for the SVD
  // M = U * S * V^T, which is the minimizer of sum_i ||R * x_i - x_hat_i||^2.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::MatrixXd r = svd.matrixV() * svd.matrixU().transpose();

  RowMajorMap(matrix_.data(), dim, dim) = r.cast<float>();
}

int OpqRotator::fit(const float *x, const float *x_hat, size_t num) {
  if (!x || !x_hat || num == 0 || dim_ <= 0) return kErrInvalidArgument;
  train(x, const_cast<float *>(x_hat), num, 0);
  return 0;
}

void OpqRotator::train(const void * /*data*/, size_t /*num*/,
                       size_t /*stride*/) {
  // No-op: OPQ needs (x, x_hat) pairs, see fit().
}

// ---------------------------------------------------------------------------
// apply / apply_inverse
// ---------------------------------------------------------------------------

void OpqRotator::apply(const float *in, float *out) const {
  // ctx is the dim x dim row-major rotation matrix itself.
  void *ctx = const_cast<float *>(matrix_.data());
  kernels_.rotate(in, out, static_cast<size_t>(dim_), static_cast<size_t>(dim_),
                  ctx);
}

void OpqRotator::apply_inverse(const float *in, float *out) const {
  void *ctx = const_cast<float *>(matrix_.data());
  kernels_.unrotate(in, out, static_cast<size_t>(dim_),
                    static_cast<size_t>(dim_), ctx);
}

// ---------------------------------------------------------------------------
// serialize / deserialize
// ---------------------------------------------------------------------------

int OpqRotator::serialize(std::string *out) const {
  if (!out) return kErrInvalidArgument;
  if (dim_ <= 0 || matrix_.empty()) return kErrRuntime;

  const size_t matrix_bytes = matrix_.size() * sizeof(float);

  RotatorSerHeader hdr{};
  hdr.magic = kRotatorMagic;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kOpq);
  hdr.in_dim = static_cast<uint32_t>(dim_);
  hdr.out_dim = static_cast<uint32_t>(dim_);
  hdr.payload_size = static_cast<uint32_t>(matrix_bytes);
  hdr.reserved = 0;

  out->resize(sizeof(hdr) + matrix_bytes);
  std::memcpy(&(*out)[0], &hdr, sizeof(hdr));
  std::memcpy(&(*out)[sizeof(hdr)], matrix_.data(), matrix_bytes);
  return 0;
}

int OpqRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return kErrInvalidArgument;

  // Aligned header copy, see from_blob().
  RotatorSerHeader hdr;
  std::memcpy(&hdr, data, sizeof(RotatorSerHeader));
  if (hdr.magic != kRotatorMagic) return kErrUnsupported;
  if (hdr.version != kRotatorSerVersion) return kErrUnsupported;
  if (static_cast<RotateType>(hdr.rotator_type) != RotateType::kOpq) {
    return kErrUnsupported;
  }

  // OPQ keeps dimensionality unchanged, and the dimension must be
  // representable as int (the matrix is dim x dim floats).
  if (hdr.in_dim == 0 ||
      hdr.in_dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return kErrInvalidArgument;
  }
  if (hdr.out_dim != hdr.in_dim) return kErrInvalidArgument;

  // Length check via subtraction to avoid size_t overflow on 32-bit
  // (len >= sizeof(header) was guaranteed above, so the subtraction is safe).
  if (hdr.payload_size > len - sizeof(RotatorSerHeader)) {
    return kErrInvalidArgument;
  }

  // The payload must hold exactly the dim x dim rotation matrix: apply() reads
  // all of it, so any shorter payload would read out of bounds.
  const size_t elements =
      static_cast<size_t>(hdr.in_dim) * static_cast<size_t>(hdr.in_dim);
  const size_t expected_bytes = elements * sizeof(float);
  if (hdr.payload_size != expected_bytes) return kErrInvalidArgument;

  // Commit only after the payload copy, so a malformed blob cannot leave a
  // half-updated rotator behind (deserialize may target a live object).
  std::vector<float> new_matrix(elements);
  std::memcpy(new_matrix.data(),
              reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader),
              expected_bytes);

  matrix_.swap(new_matrix);
  dim_ = static_cast<int>(hdr.in_dim);
  kernels_ = get_rotator_kernels(RotateType::kOpq);
  return 0;
}

}  // namespace turbo
}  // namespace zvec
