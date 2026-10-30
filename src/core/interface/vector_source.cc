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

#include <algorithm>
#include <limits>
#include <utility>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/interface/vector_source.h>

namespace zvec {
namespace core {

namespace {

bool ValidContiguousView(const ContiguousBatch &view) {
  if (!view.count) return true;
  constexpr auto kInvalidId = std::numeric_limits<uint32_t>::max();
  if (!view.data || !view.stride || view.first_id >= kInvalidId ||
      view.count > kInvalidId - view.first_id)
    return false;
  const auto max_span = std::numeric_limits<uintptr_t>::max() -
                        reinterpret_cast<uintptr_t>(view.data);
  return view.count <= max_span / view.stride;
}

const void *GetContiguousVector(const ContiguousBatch &view, uint32_t id) {
  if (id < view.first_id || id - view.first_id >= view.count) return nullptr;
  return static_cast<const char *>(view.data) +
         size_t(id - view.first_id) * view.stride;
}

class ContiguousVectorIterator final : public VectorIterator {
 public:
  explicit ContiguousVectorIterator(ContiguousBatch view)
      : view_(std::move(view)), valid_(ValidContiguousView(view_)) {}

  int next_batch(uint32_t max_count, VectorBatch *out) override {
    if (!out || !max_count) return IndexError_InvalidArgument;
    auto &batch = out->emplace<ContiguousBatch>();
    if (!valid_) return IndexError_InvalidArgument;
    if (offset_ == view_.count) return 0;
    const auto count = std::min(max_count, view_.count - offset_);
    batch = {
        static_cast<const char *>(view_.data) + size_t(offset_) * view_.stride,
        view_.first_id + offset_, count, view_.stride, view_.lease};
    offset_ += count;
    return 0;
  }

 private:
  const ContiguousBatch view_;
  const bool valid_;
  uint32_t offset_{0};
};

}  // namespace

VectorSource::VectorSource() = default;

VectorSource::~VectorSource() = default;

VectorIterator::Pointer VectorSource::create_iterator() const {
  return nullptr;
}

void VectorSource::get_vectors(const uint32_t *ids, uint32_t count,
                               const void **out) const {
  for (uint32_t i = 0; i < count; ++i) {
    out[i] = get_vector(ids[i]);
  }
}

const void *ContiguousVectorSource::get_vector(uint32_t id) const {
  const auto view = get_contiguous_view();
  return ValidContiguousView(view) ? GetContiguousVector(view, id) : nullptr;
}

void ContiguousVectorSource::get_vectors(const uint32_t *ids, uint32_t count,
                                         const void **out) const {
  if (!count) return;
  const auto view = get_contiguous_view();
  const bool valid = ValidContiguousView(view);
  for (uint32_t i = 0; i < count; ++i) {
    out[i] = valid ? GetContiguousVector(view, ids[i]) : nullptr;
  }
}

VectorIterator::Pointer ContiguousVectorSource::create_iterator() const {
  return std::make_unique<ContiguousVectorIterator>(get_contiguous_view());
}

}  // namespace core
}  // namespace zvec
