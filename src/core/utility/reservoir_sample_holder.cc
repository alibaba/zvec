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
#include "reservoir_sample_holder.h"
#include <random>
#include "paged_build_buffer.h"

namespace zvec::core {
namespace {
class ReservoirHolder final
    : public IndexHolder,
      public OrdinalAccessHolder,
      public std::enable_shared_from_this<ReservoirHolder> {
 public:
  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return dim_;
  }
  size_t element_size() const override {
    return bytes_;
  }
  IndexMeta::DataType data_type() const override {
    return type_;
  }
  bool multipass() const override {
    return true;
  }

  class Cursor : public Iterator {
   public:
    explicit Cursor(std::shared_ptr<ReservoirHolder> holder)
        : holder_(std::move(holder)), row_((holder_->bytes_ + 7) / 8) {
      load();
    }
    const void *data() const override {
      return status_ == 0 ? row_.data() : nullptr;
    }
    uint64_t key() const override {
      return position_;
    }
    bool is_valid() const override {
      return position_ < holder_->count_ && status_ == 0;
    }
    int status() const override {
      return status_;
    }
    void next() override {
      ++position_;
      load();
    }

   private:
    void load() {
      if (position_ < holder_->count_ && status_ == 0)
        status_ = holder_->buffer_.read(position_ * holder_->bytes_,
                                        row_.data(), holder_->bytes_);
    }
    std::shared_ptr<ReservoirHolder> holder_;
    ailego::BudgetedVector<uint64_t> row_;
    size_t position_{0};
    int status_{0};
  };
  Iterator::Pointer create_iterator() override {
    return std::make_unique<Cursor>(shared_from_this());
  }

  // The sample is already pageable and immutable. Expose ordinal reads so
  // multi-pass consumers can borrow it instead of spooling a second copy.
  int create_ordinal_reader(Reader::Pointer *out) override try {
    if (!out) return IndexError_InvalidArgument;
    class SampleReader : public Reader {
     public:
      explicit SampleReader(std::shared_ptr<ReservoirHolder> holder)
          : holder_(std::move(holder)) {}
      int read(size_t ordinal, uint64_t *key, const void **data) override try {
        if (!key || !data) return IndexError_InvalidArgument;
        *data = nullptr;
        if (ordinal >= holder_->count_) return IndexError_OutOfRange;
        row_.resize((holder_->bytes_ + 7) / 8);
        int ret = holder_->buffer_.read(ordinal * holder_->bytes_, row_.data(),
                                        holder_->bytes_);
        if (ret != 0) return ret;
        *key = ordinal;
        *data = row_.data();
        return 0;
      } catch (const std::bad_alloc &) {
        return IndexError_NoMemory;
      } catch (const std::exception &) {
        return IndexError_ReadData;
      }
      void reset() override {
        ailego::BudgetedVector<uint64_t>().swap(row_);
      }

     private:
      std::shared_ptr<ReservoirHolder> holder_;
      ailego::BudgetedVector<uint64_t> row_;
    };
    *out = std::make_unique<SampleReader>(shared_from_this());
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }

  int init(const IndexHolder::Pointer &input, size_t limit) {
    dim_ = input->dimension();
    type_ = input->data_type();
    bytes_ = input->element_size();
    const size_t expected = input->count();
    const size_t capacity = std::min(expected, limit);
    if (!capacity || !bytes_ ||
        capacity > std::numeric_limits<size_t>::max() / bytes_)
      return IndexError_InvalidArgument;
    int ret = buffer_.init(capacity * bytes_);
    if (ret != 0) return ret;
    auto iter = input->create_iterator();
    if (!iter) return IndexError_Runtime;
    std::mt19937 rng;
    size_t total = 0;
    for (; iter->is_valid(); iter->next(), ++total) {
      if (iter->status() != 0) return iter->status();
      if (total >= expected) return IndexError_InvalidLength;
      // Consume every row, even if not selected, so source errors cannot be
      // hidden by sampling and reusable iterator storage never escapes.
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      const size_t slot =
          total < capacity
              ? total
              : std::uniform_int_distribution<size_t>(0, total)(rng);
      if (slot < capacity) {
        ret = buffer_.write(slot * bytes_, data, bytes_);
        if (ret != 0) return ret;
      }
    }
    if (iter->status() != 0) return iter->status();
    if (expected != std::numeric_limits<size_t>::max() && total != expected)
      return IndexError_InvalidLength;
    count_ = std::min(total, capacity);
    return count_ ? 0 : IndexError_InvalidLength;
  }

 private:
  PagedBuildBuffer buffer_;
  size_t count_{0};
  size_t dim_{0};
  size_t bytes_{0};
  IndexMeta::DataType type_{};
};
}  // namespace

int CreateReservoirSampleHolder(const IndexHolder::Pointer &input, size_t limit,
                                IndexHolder::Pointer *out) try {
  if (!input || !out || !limit) return IndexError_InvalidArgument;
  auto candidate = std::make_shared<ReservoirHolder>();
  int ret = candidate->init(input, limit);
  if (ret == 0) *out = std::move(candidate);
  return ret;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}
}  // namespace zvec::core
