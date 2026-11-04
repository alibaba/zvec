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

#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>
#include <gtest/gtest.h>
#include "core/utility/prefix_index_holder.h"
#include "quantizer/common/pq_training_samples.h"
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"

namespace zvec::turbo {
namespace {

using core::IndexHolder;
using core::IndexMeta;

// Immutable MultiPass holders keep every row stable. The lifetime directly
// owns the source, so tests can destroy the wrapper and its iterator.
class StableHolder : public IndexHolder, public core::StableVectorHolder {
 public:
  explicit StableHolder(IndexHolder::Pointer source)
      : source_(std::move(source)) {}
  size_t count() const override {
    return source_->count();
  }
  size_t dimension() const override {
    return source_->dimension();
  }
  IndexMeta::DataType data_type() const override {
    return source_->data_type();
  }
  size_t element_size() const override {
    return source_->element_size();
  }
  bool multipass() const override {
    return true;
  }
  Iterator::Pointer create_iterator() override {
    ++iterator_calls;
    return source_->create_iterator();
  }
  int retain_stable_data(std::shared_ptr<const void> *lifetime) override {
    ++retain_calls;
    if (retain_result != 0) return retain_result;
    if (!null_lifetime) *lifetime = source_;
    return 0;
  }

  int retain_result = 0;
  bool null_lifetime = false;
  size_t retain_calls = 0;
  size_t iterator_calls = 0;

 protected:
  IndexHolder::Pointer source_;
};

template <IndexMeta::DataType Type>
IndexHolder::Pointer MakeSource(size_t count, size_t dim) {
  using T =
      std::conditional_t<Type == IndexMeta::DT_FP32, float, ailego::Float16>;
  auto holder = std::make_shared<core::MultiPassIndexHolder<Type>>(dim);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    ailego::NumericalVector<T> row(dim);
    for (size_t d = 0; d < dim; ++d) row[d] = T(distribution(rng));
    holder->emplace(i, std::move(row));
  }
  return holder;
}

IndexHolder::Pointer MakeSource(IndexMeta::DataType type, size_t count,
                                size_t dim) {
  return type == IndexMeta::DT_FP32
             ? MakeSource<IndexMeta::DT_FP32>(count, dim)
             : MakeSource<IndexMeta::DT_FP16>(count, dim);
}

TEST(PqTrainingSamples, BorrowedOrderMatchesCopiedSamples) {
  constexpr size_t kCount = 65573, kLimit = 65536, kDim = 2;
  auto source = MakeSource<IndexMeta::DT_FP32>(kCount, kDim);
  auto stable = std::make_shared<StableHolder>(source);
  for (size_t limit : {kLimit, kCount, kCount + 1}) {
    SCOPED_TRACE(limit);
    std::vector<uint8_t> copied;
    ASSERT_TRUE(
        CollectPqTrainingSamples(source, limit, kDim * sizeof(float), &copied));
    std::vector<const void *> rows;
    std::shared_ptr<const void> lifetime;
    ASSERT_EQ(0, BorrowPqTrainingSamples(stable, limit, kDim * sizeof(float),
                                         &rows, &lifetime));
    ASSERT_TRUE(lifetime);
    ASSERT_EQ(std::min(kCount, limit), rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      ASSERT_EQ(0,
                std::memcmp(rows[i], copied.data() + i * kDim * sizeof(float),
                            kDim * sizeof(float)));
    }
  }
}

TEST(PqTrainingSamples, PrefixForwardsLifetimeAndBounds) {
  auto source = MakeSource<IndexMeta::DT_FP32>(512, 3);
  std::weak_ptr<IndexHolder> weak = source;
  auto stable = std::make_shared<StableHolder>(source);
  auto prefix = std::make_shared<core::PrefixIndexHolder>(
      stable, 300, IndexMeta(IndexMeta::DT_FP32, 3));
  std::vector<uint8_t> copied;
  ASSERT_TRUE(
      CollectPqTrainingSamples(prefix, 300, 3 * sizeof(float), &copied));
  std::vector<const void *> rows;
  std::shared_ptr<const void> lifetime;
  ASSERT_EQ(0, BorrowPqTrainingSamples(prefix, 65536, 3 * sizeof(float), &rows,
                                       &lifetime));
  ASSERT_EQ(300u, rows.size());
  EXPECT_EQ(1u, stable->retain_calls);
  prefix.reset();
  stable.reset();
  source.reset();
  ASSERT_FALSE(weak.expired());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(0, std::memcmp(rows[i], copied.data() + i * 3 * sizeof(float),
                             3 * sizeof(float)));
  }
  lifetime.reset();
  EXPECT_TRUE(weak.expired());
}

// Deliberately recycles the iterator's scratch row and has no stable-data
// capability. Copying must happen before the next row is read.
class ReusingHolder : public IndexHolder {
 public:
  size_t count() const override {
    return 512;
  }
  size_t dimension() const override {
    return 4;
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  size_t element_size() const override {
    return 4 * sizeof(float);
  }
  bool multipass() const override {
    return true;
  }
  Iterator::Pointer create_iterator() override {
    ++iterator_calls;
    struct ReusingIterator : Iterator {
      const void *data() const override {
        for (size_t d = 0; d < row.size(); ++d) row[d] = float(index * 4 + d);
        return row.data();
      }
      bool is_valid() const override {
        return index < 512;
      }
      uint64_t key() const override {
        return index;
      }
      void next() override {
        ++index;
      }
      size_t index = 0;
      mutable std::array<float, 4> row{};
    };
    return std::make_unique<ReusingIterator>();
  }
  size_t iterator_calls = 0;
};

TEST(PqTrainingSamples, ReusedIteratorBuffersRequireCopying) {
  auto holder = std::make_shared<ReusingHolder>();
  int sentinel = 0;
  std::vector<const void *> rows{&sentinel};
  std::shared_ptr<const void> lifetime = std::make_shared<int>(7);
  const auto owner = lifetime;
  EXPECT_EQ(core::IndexError_NotImplemented,
            BorrowPqTrainingSamples(holder, 65536, holder->element_size(),
                                    &rows, &lifetime));
  EXPECT_EQ(0u, holder->iterator_calls);
  ASSERT_EQ(1u, rows.size());
  EXPECT_EQ(&sentinel, rows[0]);
  EXPECT_EQ(owner, lifetime);
  std::vector<uint8_t> copied;
  ASSERT_TRUE(
      CollectPqTrainingSamples(holder, 65536, holder->element_size(), &copied));
  for (size_t i = 0; i < holder->count(); ++i) {
    std::array<float, 4> row;
    std::memcpy(row.data(), copied.data() + i * sizeof(row), sizeof(row));
    for (size_t d = 0; d < row.size(); ++d) EXPECT_EQ(float(i * 4 + d), row[d]);
  }
}

class FailingHolder : public StableHolder {
 public:
  using StableHolder::StableHolder;
  Iterator::Pointer create_iterator() override {
    struct FailingIterator : Iterator {
      FailingIterator(Iterator::Pointer input, bool on_data)
          : input_(std::move(input)), on_data_(on_data) {}
      const void *data() const override {
        if (on_data_) error_ = core::IndexError_ReadData;
        return input_->data();
      }
      bool is_valid() const override {
        bool valid = input_->is_valid();
        if (!valid) error_ = core::IndexError_ReadData;
        return valid;
      }
      int status() const override {
        return error_;
      }
      uint64_t key() const override {
        return input_->key();
      }
      void next() override {
        input_->next();
      }
      Iterator::Pointer input_;
      bool on_data_;
      mutable int error_ = 0;
    };
    return std::make_unique<FailingIterator>(source_->create_iterator(),
                                             fail_on_data);
  }
  bool fail_on_data = false;
};

TEST(PqTrainingSamples, ReadErrorsLeaveBothOutputsUnchanged) {
  for (bool on_data : {false, true}) {
    auto holder =
        std::make_shared<FailingHolder>(MakeSource<IndexMeta::DT_FP32>(512, 4));
    holder->fail_on_data = on_data;
    IndexHolder::Pointer input = holder;
    if (on_data) {
      input = std::make_shared<core::PrefixIndexHolder>(
          input, 300, IndexMeta(IndexMeta::DT_FP32, 4));
    }
    int sentinel = 0;
    std::vector<const void *> rows{&sentinel};
    std::shared_ptr<const void> lifetime = std::make_shared<int>(7);
    const auto owner = lifetime;
    EXPECT_EQ(core::IndexError_ReadData,
              BorrowPqTrainingSamples(input, 65536, 4 * sizeof(float), &rows,
                                      &lifetime));
    ASSERT_EQ(1u, rows.size());
    EXPECT_EQ(&sentinel, rows[0]);
    EXPECT_EQ(owner, lifetime);
    std::vector<uint8_t> copied{0x5a};
    EXPECT_FALSE(
        CollectPqTrainingSamples(input, 65536, 4 * sizeof(float), &copied));
    EXPECT_EQ(std::vector<uint8_t>{0x5a}, copied);
  }
}

TEST(PqTrainingSamples, ReadOnlyTrainingBorrowsForBothPrecisionsAndMetrics) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16}) {
    for (const char *metric : {"SquaredEuclidean", "InnerProduct"}) {
      SCOPED_TRACE(static_cast<int>(type));
      SCOPED_TRACE(metric);
      auto source = MakeSource(type, 512, 17);
      auto stable = std::make_shared<StableHolder>(source);
      IndexMeta meta(type, 17);
      meta.set_metric(metric, 0, ailego::Params());
      ailego::Params params;
      params.set("num_chunk", uint32_t{4});
      params.set("thread_count", uint32_t{2});
      PqInt8Quantizer quantizer;
      ASSERT_EQ(0, quantizer.init(meta, params));
      std::vector<uint8_t> before;
      ASSERT_TRUE(CollectPqTrainingSamples(source, 512, source->element_size(),
                                           &before));
      ASSERT_EQ(0, quantizer.train(stable));
      EXPECT_EQ(1u, stable->retain_calls);
      std::vector<uint8_t> after;
      ASSERT_TRUE(CollectPqTrainingSamples(source, 512, source->element_size(),
                                           &after));
      EXPECT_EQ(before, after);
      auto iter = source->create_iterator();
      std::vector<uint8_t> code(quantizer.quantized_datapoint_vector_length());
      std::vector<float> lut(quantizer.quantized_query_vector_length() / 4);
      quantizer.quantize_data(iter->data(), code.data());
      quantizer.quantize_query(iter->data(), lut.data());
      EXPECT_TRUE(std::isfinite(
          quantizer.calc_distance_dp_query(code.data(), lut.data())));
      std::string blob;
      ASSERT_EQ(0, quantizer.serialize(&blob));
      PqInt8Quantizer restored;
      ASSERT_EQ(0, restored.init(meta, params));
      ASSERT_EQ(0, restored.deserialize(blob));
      EXPECT_EQ(quantizer.calc_distance_dp_query(code.data(), lut.data()),
                restored.calc_distance_dp_query(code.data(), lut.data()));
    }
  }
}

TEST(PqTrainingSamples, MutatingTrainingKeepsTheCopyingPath) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16}) {
    for (int mode : {0, 1, 2}) {
      if (mode == 2 && type == IndexMeta::DT_FP16) continue;
      auto source = MakeSource(type, 512, 8);
      auto stable = std::make_shared<StableHolder>(source);
      stable->retain_result = core::IndexError_ReadData;
      IndexMeta meta(type, 8);
      meta.set_metric(mode == 0 ? "Cosine" : "SquaredEuclidean", 0,
                      ailego::Params());
      ailego::Params params;
      params.set("num_chunk", uint32_t{4});
      params.set("thread_count", uint32_t{1});
      if (mode == 1) params.set("use_zero_mean", true);
      if (mode == 2) {
        params.set("rotate_type", "opq");
        params.set("opq_iter", uint32_t{2});
        params.set("opq_pq_iter", uint32_t{1});
      }
      PqInt8Quantizer quantizer;
      ASSERT_EQ(0, quantizer.init(meta, params));
      std::vector<uint8_t> before, after;
      ASSERT_TRUE(CollectPqTrainingSamples(source, 512, source->element_size(),
                                           &before));
      ASSERT_EQ(0, quantizer.train(stable));
      EXPECT_EQ(0u, stable->retain_calls);
      ASSERT_TRUE(CollectPqTrainingSamples(source, 512, source->element_size(),
                                           &after));
      EXPECT_EQ(before, after);
    }
  }
}

TEST(PqTrainingSamples, OnlyUnsupportedCapabilitiesPermitFallback) {
  for (int error : {core::IndexError_NotImplemented, core::IndexError_ReadData,
                    core::IndexError_Canceled}) {
    auto stable =
        std::make_shared<StableHolder>(MakeSource<IndexMeta::DT_FP32>(512, 8));
    stable->retain_result = error;
    IndexMeta meta(IndexMeta::DT_FP32, 8);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ailego::Params params;
    params.set("num_chunk", uint32_t{4});
    PqInt8Quantizer quantizer;
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(error == core::IndexError_NotImplemented ? 0 : error,
              quantizer.train(stable));
    EXPECT_EQ(error == core::IndexError_NotImplemented ? 1u : 0u,
              stable->iterator_calls);
  }
  auto stable =
      std::make_shared<StableHolder>(MakeSource<IndexMeta::DT_FP32>(512, 8));
  stable->null_lifetime = true;
  std::vector<const void *> rows;
  std::shared_ptr<const void> lifetime;
  EXPECT_EQ(core::IndexError_Runtime,
            BorrowPqTrainingSamples(stable, 65536, 8 * sizeof(float), &rows,
                                    &lifetime));
  EXPECT_EQ(0u, stable->iterator_calls);
}

TEST(PqTrainingSamples, TrainingReleasesBorrowedSourceAfterWorkersFinish) {
  auto source = MakeSource<IndexMeta::DT_FP32>(512, 8);
  std::weak_ptr<IndexHolder> weak = source;
  auto stable = std::make_shared<StableHolder>(source);
  source.reset();
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", uint32_t{4});
  params.set("thread_count", uint32_t{2});
  PqInt8Quantizer quantizer;
  ASSERT_EQ(0, quantizer.init(meta, params));
  ASSERT_EQ(0, quantizer.train(std::move(stable)));
  EXPECT_TRUE(weak.expired());
}

}  // namespace
}  // namespace zvec::turbo
