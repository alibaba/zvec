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
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {

int MatrixRotator::init_impl(size_t dim) {
  return matrix_.generate(dim, dim);
}
void MatrixRotator::rotate(const float *in, float *out) const {
  matrix_.apply(in, out);
}
void MatrixRotator::unrotate(const float *in, float *out) const {
  matrix_.apply(in, out, true);
}
RotatorType MatrixRotator::rotator_type() const {
  return RotatorType::Matrix;
}
void MatrixRotator::save_blob(char *data) const {
  PagedRotationMatrix::check(matrix_.read(0, data, matrix_.size()));
}
void MatrixRotator::load_blob(const char *data) {
  PagedRotationMatrix::check(matrix_.load(
      dimension_, dimension_, [data](size_t offset, void *out, size_t bytes) {
        std::memcpy(out, data + offset, bytes);
        return 0;
      }));
}
int MatrixRotator::read_blob(size_t offset, void *data, size_t bytes) const {
  return matrix_.read(offset, data, bytes);
}
int MatrixRotator::load_reader(const BlobReader &reader, size_t bytes) {
  if (dimension_ == 0 || dimension_ > bytes / sizeof(float) / dimension_ ||
      bytes != dimension_ * dimension_ * sizeof(float))
    return IndexError_InvalidFormat;
  return matrix_.load(dimension_, dimension_, reader);
}
size_t MatrixRotator::blob_bytes() const {
  return matrix_.size();
}
}  // namespace core
}  // namespace zvec
