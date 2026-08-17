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
#include <zvec/ailego/logger/logger.h>

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

void PqOpq::train(void *data, size_t num, const char *name,
                  const TrainCodebook &train_codebook,
                  const Reconstruct &reconstruct) {
  if (!enabled() || num == 0) return;
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
