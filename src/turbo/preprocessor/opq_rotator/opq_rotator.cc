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
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include "paged_opq_workspace.h"
// Eigen stays inside this translation unit (baseline ISA flags) to avoid the
// ODR hazard described in src/core/quantizer/rotator/matrix_rotator.cc.
#include <rabitqlib/third/Eigen/Dense>

namespace zvec {
namespace turbo {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstRowMajorMap = Eigen::Map<const RowMajorMatrix>;
using RowMajorMap = Eigen::Map<RowMajorMatrix>;

//! Rows per block when accumulating the cross-covariance in double; a plain
//! float GEMM over many rows loses enough precision to break orthogonality.
constexpr Eigen::Index kAccumBlockRows = 1024;

//! Random orthogonal dim x dim matrix (Q factor of a Gaussian Householder QR).
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

OpqRotator::Pointer OpqRotator::create(int dim, uint64_t seed) try {
  if (dim <= 0) return nullptr;

  Pointer r(new OpqRotator());
  r->dim_ = dim;
  if (ailego::MemoryLimitPool::get_instance().capacity()) {
    r->paged_matrix_ = std::make_unique<core::PagedBuildBuffer>();
    PagedOpqWorkspace workspace(dim);
    if (workspace.random_rotation(seed, r->paged_matrix_.get()) != 0)
      return nullptr;
  } else {
    r->matrix_.resize(static_cast<size_t>(dim) * static_cast<size_t>(dim));
    random_orthogonal_matrix(r->matrix_.data(), dim, seed);
  }
  r->kernels_ = get_rotator_kernels(RotateType::kOpq);
  return r;
} catch (const std::exception &) {
  return nullptr;
}

OpqRotator::Pointer OpqRotator::from_serialized(const void *data, size_t len) {
  Pointer r(new OpqRotator());
  if (r->deserialize(data, len) != 0) return nullptr;
  return r;
}

// OPQ step 1: fixed codebook -> rotation matrix.  data/ctx are packed fp32
// (x, x_hat) pairs; stride must be 0.
void OpqRotator::train(const void *data, void *ctx, size_t num, size_t stride) {
  assert(data && ctx && num > 0 && dim_ > 0 && stride == 0);
  (void)stride;  // assert compiles out under NDEBUG
  const float *x = reinterpret_cast<const float *>(data);
  const float *x_hat = reinterpret_cast<const float *>(ctx);
  if (ailego::MemoryLimitPool::get_instance().capacity()) {
    if (fit(x, x_hat, num) != 0)
      throw std::runtime_error("Paged OPQ training failed");
    return;
  }

  const Eigen::Index dim = static_cast<Eigen::Index>(dim_);
  const Eigen::Index rows = static_cast<Eigen::Index>(num);

  // M = X^T * X_hat
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(dim, dim);
  for (Eigen::Index begin = 0; begin < rows; begin += kAccumBlockRows) {
    const Eigen::Index block = std::min(kAccumBlockRows, rows - begin);
    ConstRowMajorMap xb(x + begin * dim, block, dim);
    ConstRowMajorMap xhb(x_hat + begin * dim, block, dim);
    m.noalias() += xb.cast<double>().transpose() * xhb.cast<double>();
  }

  // Orthogonal Procrustes: R = V * U^T for the SVD M = U * S * V^T.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::MatrixXd r = svd.matrixV() * svd.matrixU().transpose();

  RowMajorMap(matrix_.data(), dim, dim) = r.cast<float>();
}

int OpqRotator::fit(const float *x, const float *x_hat, size_t num) {
  if (!x || !x_hat || num == 0 || dim_ <= 0) return kErrInvalidArgument;
  if (ailego::MemoryLimitPool::get_instance().capacity()) {
    return fit_rows(num, [&](size_t first, size_t count, float *a, float *b) {
      std::memcpy(a, x + first * dim_, count * dim_ * sizeof(float));
      std::memcpy(b, x_hat + first * dim_, count * dim_ * sizeof(float));
      return 0;
    });
  }
  train(x, const_cast<float *>(x_hat), num, 0);
  return 0;
}

int OpqRotator::fit_rows(size_t num, const ReadPairs &read) try {
  if (!read || num == 0 || dim_ <= 0) return kErrInvalidArgument;
  if (ailego::MemoryLimitPool::get_instance().capacity()) {
    auto output = std::make_unique<core::PagedBuildBuffer>();
    PagedOpqWorkspace workspace(dim_);
    int ret = workspace.fit(num, read, output.get());
    if (ret != 0) return ret;
    paged_matrix_ = std::move(output);
    std::vector<float>().swap(matrix_);
    return 0;
  }
  const Eigen::Index dim = static_cast<Eigen::Index>(dim_);
  const size_t tile = std::min<size_t>(num, kAccumBlockRows);
  std::vector<float> x(tile * dim_), x_hat(x.size());
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(dim, dim);
  for (size_t first = 0; first < num; first += tile) {
    const size_t count = std::min(tile, num - first);
    int ret = read(first, count, x.data(), x_hat.data());
    if (ret != 0) return ret;
    ConstRowMajorMap xb(x.data(), count, dim);
    ConstRowMajorMap xhb(x_hat.data(), count, dim);
    m.noalias() += xb.cast<double>().transpose() * xhb.cast<double>();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::MatrixXd r = svd.matrixV() * svd.matrixU().transpose();
  RowMajorMap(matrix_.data(), dim, dim) = r.cast<float>();
  return 0;
} catch (const std::bad_alloc &) {
  return core::IndexError_NoMemory;
} catch (const std::exception &) {
  return core::IndexError_Runtime;
}

void OpqRotator::train(const void * /*data*/, size_t /*num*/,
                       size_t /*stride*/) {
  // OPQ needs (x, x_hat) pairs, see the two-input train() overload.
}

void OpqRotator::apply(const float *in, float *out) const {
  if (paged_matrix_) {
    std::vector<float> row(dim_);
    for (int i = 0; i < dim_; ++i) {
      if (paged_matrix_->read(static_cast<size_t>(i) * dim_ * sizeof(float),
                              row.data(), row.size() * sizeof(float)) != 0)
        throw std::runtime_error("Failed to read OPQ rotation");
      float value = 0;
      for (int j = 0; j < dim_; ++j) value += row[j] * in[j];
      out[i] = value;
    }
    return;
  }
  void *ctx = const_cast<float *>(matrix_.data());
  kernels_.rotate(in, out, static_cast<size_t>(dim_), static_cast<size_t>(dim_),
                  ctx);
}

void OpqRotator::apply_inverse(const float *in, float *out) const {
  if (paged_matrix_) {
    std::vector<float> row(dim_);
    std::fill(out, out + dim_, 0);
    for (int i = 0; i < dim_; ++i) {
      if (paged_matrix_->read(static_cast<size_t>(i) * dim_ * sizeof(float),
                              row.data(), row.size() * sizeof(float)) != 0)
        throw std::runtime_error("Failed to read OPQ rotation");
      for (int j = 0; j < dim_; ++j) out[j] += row[j] * in[i];
    }
    return;
  }
  void *ctx = const_cast<float *>(matrix_.data());
  kernels_.unrotate(in, out, static_cast<size_t>(dim_),
                    static_cast<size_t>(dim_), ctx);
}

int OpqRotator::serialize(std::string *out) const {
  return serialize(out, nullptr);
}

size_t OpqRotator::serialized_size() const {
  return dim_ > 0 ? sizeof(RotatorSerHeader) +
                        static_cast<size_t>(dim_) * dim_ * sizeof(float)
                  : 0;
}

int OpqRotator::serialize_to(const SerializationSink &sink) const {
  if (!sink) return kErrInvalidArgument;
  if (dim_ <= 0 || (!paged_matrix_ && matrix_.empty())) return kErrRuntime;

  const size_t matrix_bytes = static_cast<size_t>(dim_) * dim_ * sizeof(float);
  if (matrix_bytes > std::numeric_limits<uint32_t>::max())
    return kErrInvalidArgument;

  RotatorSerHeader hdr{};
  hdr.magic = kRotatorMagic;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kOpq);
  hdr.in_dim = static_cast<uint32_t>(dim_);
  hdr.out_dim = static_cast<uint32_t>(dim_);
  hdr.payload_size = static_cast<uint32_t>(matrix_bytes);
  hdr.reserved = 0;

  int ret = sink(&hdr, sizeof(hdr));
  if (ret != 0) return ret;
  if (!paged_matrix_)
    return WriteSerialized(sink, matrix_.data(), matrix_bytes);
  std::array<char, 64 * 1024> scratch{};
  for (size_t offset = 0; offset < matrix_bytes; offset += scratch.size()) {
    const size_t n = std::min(scratch.size(), matrix_bytes - offset);
    ret = paged_matrix_->read(offset, scratch.data(), n);
    if (ret == 0) ret = sink(scratch.data(), n);
    if (ret != 0) return ret;
  }
  return 0;
}

int OpqRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return kErrInvalidArgument;

  RotatorSerHeader hdr;
  std::memcpy(&hdr, data, sizeof(RotatorSerHeader));
  if (hdr.magic != kRotatorMagic) return kErrUnsupported;
  if (hdr.version != kRotatorSerVersion) return kErrUnsupported;
  if (static_cast<RotateType>(hdr.rotator_type) != RotateType::kOpq) {
    return kErrUnsupported;
  }
  if (hdr.in_dim == 0 ||
      hdr.in_dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return kErrInvalidArgument;
  }
  if (hdr.out_dim != hdr.in_dim) return kErrInvalidArgument;

  // Subtraction is safe: len >= sizeof(header) was checked above.
  if (hdr.payload_size > len - sizeof(RotatorSerHeader)) {
    return kErrInvalidArgument;
  }

  // apply() reads the whole matrix, so the payload must match exactly.
  const size_t elements =
      static_cast<size_t>(hdr.in_dim) * static_cast<size_t>(hdr.in_dim);
  const size_t expected_bytes = elements * sizeof(float);
  if (hdr.payload_size != expected_bytes) return kErrInvalidArgument;

  // Copy before committing so a malformed blob never half-updates the object.
  const void *payload =
      reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader);
  if (ailego::MemoryLimitPool::get_instance().capacity()) {
    auto candidate = std::make_unique<core::PagedBuildBuffer>();
    int ret = candidate->init(expected_bytes, 64 * 1024);
    if (!ret) ret = candidate->write(0, payload, expected_bytes);
    if (ret) return ret;
    paged_matrix_ = std::move(candidate);
    std::vector<float>().swap(matrix_);
  } else {
    std::vector<float> new_matrix(elements);
    std::memcpy(new_matrix.data(), payload, expected_bytes);
    matrix_.swap(new_matrix);
    paged_matrix_.reset();
  }
  dim_ = static_cast<int>(hdr.in_dim);
  kernels_ = get_rotator_kernels(RotateType::kOpq);
  return 0;
}

}  // namespace turbo
}  // namespace zvec
