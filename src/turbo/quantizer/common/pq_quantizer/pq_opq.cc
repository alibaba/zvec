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

#include "pq_opq.h"
#include <turbo/preprocessor/opq_rotator/opq_rotator.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/logger/logger.h>
#include "core/utility/paged_build_buffer.h"
#include "turbo/training_workspace.h"

namespace zvec {
namespace turbo {

int PqOpq::init(uint32_t dim, DataType data_type,
                const ailego::Params &params) {
  preprocessor_.reset();
  std::string type;
  params.get("rotate_type", &type);
  if (type.empty() || type == "none") return 0;
  if (type != "opq" || data_type != DataType::kFp32) return kErrUnsupported;

  params.get("opq_iter", &opq_iter_);
  params.get("opq_pq_iter", &opq_pq_iter_);
  if (opq_iter_ == 0 || opq_pq_iter_ == 0) return kErrInvalidArgument;
  preprocessor_ = OpqRotator::create(static_cast<int>(dim));
  return preprocessor_ ? 0 : kErrInvalidArgument;
}

int PqOpq::train(void *data, size_t num, const char *name,
                 const TrainCodebook &train_codebook,
                 const Reconstruct &reconstruct,
                 const ReconstructSum &reconstruct_sum) try {
  if (!enabled() || num == 0) return 0;
  if (!data) return kErrInvalidArgument;
  const size_t dim = static_cast<size_t>(preprocessor_->in_dim());
  if (!dim || num > std::numeric_limits<size_t>::max() / sizeof(float) / dim)
    return kErrInvalidArgument;
  if (reconstruct_sum &&
      ailego::MemoryLimitPool::get_instance().capacity() != 0) {
    core::PagedBuildBuffer original;
    const size_t row_bytes = dim * sizeof(float);
    int ret = original.init(num * row_bytes, 64 * 1024);
    if (ret != 0) return ret;
    ret = original.write(0, data, num * row_bytes);
    if (ret != 0) return ret;
    return train_rows(
        num, name,
        [&](size_t first, size_t count, float *rows) {
          return original.read(first * row_bytes, rows, count * row_bytes);
        },
        [&](size_t first, size_t count, const float *rows) {
          std::memcpy(static_cast<char *>(data) + first * row_bytes, rows,
                      count * row_bytes);
          return 0;
        },
        [&](uint32_t iters) {
          train_codebook(iters);
          return 0;
        },
        reconstruct_sum);
  }
  float *rotated = reinterpret_cast<float *>(data);
  const size_t total = num * static_cast<size_t>(preprocessor_->in_dim());
  std::vector<float> x(rotated, rotated + total);
  std::vector<float> x_hat(total);
  float prev_mse = 0.0f;

  for (uint32_t it = 0; it < opq_iter_; ++it) {
    rotate_batch(x.data(), num, rotated);
    train_codebook(opq_pq_iter_);
    float mse = reconstruct(rotated, num, x_hat.data());
    LOG_INFO("%s OPQ round %u/%u: reconstruction mse=%f", name, it + 1,
             opq_iter_, mse);
    if (it > 0 &&
        (mse >= prev_mse || (prev_mse - mse) <= prev_mse * kMinImprovement)) {
      break;
    }
    prev_mse = mse;
    preprocessor_->train(x.data(), x_hat.data(), num, 0);
  }

  // The final PQ codebook must be trained using the final rotation.
  rotate_batch(x.data(), num, rotated);
  return 0;
} catch (const std::bad_alloc &) {
  return core::IndexError_NoMemory;
} catch (const std::exception &) {
  return core::IndexError_Runtime;
}

int PqOpq::train_rows(size_t num, const char *name,
                      const ReadRows &read_original,
                      const WriteRows &write_rotated,
                      const TrainPagedCodebook &train_codebook,
                      const ReconstructSum &reconstruct) try {
  if (!enabled() || num == 0) return 0;
  if (!read_original || !write_rotated || !train_codebook || !reconstruct)
    return kErrInvalidArgument;
  auto rotator = std::dynamic_pointer_cast<OpqRotator>(preprocessor_);
  if (!rotator) return kErrUnsupported;
  const size_t dim = static_cast<size_t>(preprocessor_->in_dim());
  if (dim == 0 ||
      num > std::numeric_limits<size_t>::max() / sizeof(float) / dim)
    return kErrInvalidArgument;
  const size_t row_bytes = dim * sizeof(float);
  core::PagedBuildBuffer reconstruction;
  int ret = reconstruction.init(num * row_bytes, 64 * 1024);
  if (ret != 0) return ret;
  // Reserve backing metadata first, then shrink tiles under budget pressure.
  const size_t tile =
      TrainingTileRows(std::min<size_t>(num, 1024), row_bytes, 3);
  std::vector<float, ailego::BudgetedAllocator<float>> original(tile * dim),
      rotated(original.size()), reconstructed(original.size());
  auto publish_rotation = [&]() -> int {
    for (size_t first = 0; first < num; first += tile) {
      const size_t count = std::min(tile, num - first);
      int status = read_original(first, count, original.data());
      if (status != 0) return status;
      rotate_batch(original.data(), count, rotated.data());
      status = write_rotated(first, count, rotated.data());
      if (status != 0) return status;
    }
    return 0;
  };
  float previous = 0.0f;
  for (uint32_t iter = 0; iter < opq_iter_; ++iter) {
    ret = publish_rotation();
    if (ret != 0) return ret;
    ret = train_codebook(opq_pq_iter_);
    if (ret != 0) return ret;
    double error = 0.0;
    for (size_t first = 0; first < num; first += tile) {
      const size_t count = std::min(tile, num - first);
      ret = read_original(first, count, original.data());
      if (ret != 0) return ret;
      rotate_batch(original.data(), count, rotated.data());
      error += reconstruct(rotated.data(), count, reconstructed.data());
      ret = reconstruction.write(first * row_bytes, reconstructed.data(),
                                 count * row_bytes);
      if (ret != 0) return ret;
    }
    const float mse = static_cast<float>(error / num);
    LOG_INFO("%s OPQ round %u/%u: reconstruction mse=%f", name, iter + 1,
             opq_iter_, mse);
    if (iter > 0 &&
        (mse >= previous || previous - mse <= previous * kMinImprovement))
      break;
    previous = mse;
    ret = rotator->fit_rows(
        num, [&](size_t first, size_t count, float *x, float *x_hat) {
          int status = read_original(first, count, x);
          return status != 0 ? status
                             : reconstruction.read(first * row_bytes, x_hat,
                                                   count * row_bytes);
        });
    if (ret != 0) return ret;
  }
  return publish_rotation();
} catch (const std::bad_alloc &) {
  return core::IndexError_NoMemory;
} catch (const std::exception &) {
  return core::IndexError_Runtime;
}

void PqOpq::rotate_batch(const float *input, size_t num, float *output) const {
  const size_t dim = static_cast<size_t>(preprocessor_->in_dim());
  for (size_t i = 0; i < num; ++i) {
    rotate(input + i * dim, output + i * dim);
  }
}

const void *PqOpq::apply(const void *input, std::vector<float> *buffer) const {
  if (!enabled()) return input;
  buffer->resize(preprocessor_->out_dim());
  rotate(reinterpret_cast<const float *>(input), buffer->data());
  return buffer->data();
}

void PqOpq::rotate(const float *input, float *output) const {
  preprocessor_->apply(input, output);
}

void PqOpq::rotate_inverse(const float *input, float *output) const {
  preprocessor_->apply_inverse(input, output);
}

int PqOpq::serialize(std::string *out) const {
  if (!out) return kErrInvalidArgument;
  out->clear();
  return enabled() ? preprocessor_->serialize(out) : 0;
}

int PqOpq::deserialize(uint8_t type, uint32_t dim, const void *data,
                       size_t len) {
  if (type == 0) {
    preprocessor_.reset();
    return 0;
  }
  if (type != static_cast<uint8_t>(RotateType::kOpq)) return kErrUnsupported;
  if (len == 0) return kErrInvalidArgument;
  preprocessor_ = OpqRotator::from_serialized(data, len);
  if (!preprocessor_ || preprocessor_->in_dim() != static_cast<int>(dim)) {
    preprocessor_.reset();
    return kErrInvalidArgument;
  }
  return 0;
}

}  // namespace turbo
}  // namespace zvec
