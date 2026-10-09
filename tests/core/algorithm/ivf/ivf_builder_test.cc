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
#include "ivf_builder.h"
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_streamer.h>

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

class IVFBuilderTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  void prepare_index_holder(uint32_t base_key, uint32_t num);

  IndexMeta index_meta_;
  Params params_;
  uint32_t dimension_;
  IndexHolder::Pointer holder_;
  IndexThreads::Pointer threads_{};
};

void IVFBuilderTest::SetUp() {
  dimension_ = 8U;

  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());

  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "8");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  std::mt19937 gen((std::random_device())());
  bool v = std::uniform_int_distribution<size_t>(0, 1)(gen);
  if (v) {
    threads_ = std::make_shared<SingleQueueIndexThreads>();
  }
}

void IVFBuilderTest::TearDown() {}

void IVFBuilderTest::prepare_index_holder(uint32_t base_key, uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 1.0f * i;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

enum class IteratorErrorOperation { kValidity, kKey, kData };

class IteratorErrorHolder : public IndexHolder {
 public:
  IteratorErrorHolder(IndexHolder::Pointer delegate, size_t failed_ordinal,
                      IteratorErrorOperation operation)
      : delegate_(std::move(delegate)),
        failed_ordinal_(failed_ordinal),
        operation_(operation) {}

  size_t count() const override {
    return delegate_->count();
  }
  size_t dimension() const override {
    return delegate_->dimension();
  }
  IndexMeta::DataType data_type() const override {
    return delegate_->data_type();
  }
  size_t element_size() const override {
    return delegate_->element_size();
  }
  bool multipass() const override {
    return true;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<ErrorIterator>(delegate_->create_iterator(),
                                           failed_ordinal_, operation_);
  }

 private:
  class ErrorIterator : public IndexHolder::Iterator {
   public:
    ErrorIterator(IndexHolder::Iterator::Pointer delegate,
                  size_t failed_ordinal, IteratorErrorOperation operation)
        : delegate_(std::move(delegate)),
          failed_ordinal_(failed_ordinal),
          operation_(operation) {}

    bool is_valid() const override {
      return !failed(IteratorErrorOperation::kValidity) &&
             delegate_->is_valid();
    }
    const void *data() const override {
      return failed(IteratorErrorOperation::kData) ? nullptr
                                                   : delegate_->data();
    }
    uint64_t key() const override {
      return failed(IteratorErrorOperation::kKey)
                 ? std::numeric_limits<uint64_t>::max()
                 : delegate_->key();
    }
    void next() override {
      ++ordinal_;
      delegate_->next();
    }
    int status() const override {
      return status_;
    }

   private:
    bool failed(IteratorErrorOperation operation) const {
      if (operation == operation_ && ordinal_ == failed_ordinal_) {
        status_ = IndexError_ReadData;
      }
      return status_ != 0;
    }
    IndexHolder::Iterator::Pointer delegate_;
    size_t failed_ordinal_;
    IteratorErrorOperation operation_;
    size_t ordinal_{0};
    mutable int status_{0};
  };

  IndexHolder::Pointer delegate_;
  size_t failed_ordinal_;
  IteratorErrorOperation operation_;
};

class SinglePassIteratorErrorHolder : public IteratorErrorHolder {
 public:
  using IteratorErrorHolder::IteratorErrorHolder;

  bool multipass() const override {
    return false;
  }
};

TEST_F(IVFBuilderTest, TwoPassHolderDoesNotExposePartialCacheAfterReadError) {
  prepare_index_holder(0, 8);
  for (auto operation :
       {IteratorErrorOperation::kValidity, IteratorErrorOperation::kKey,
        IteratorErrorOperation::kData}) {
    for (size_t failed_ordinal : {0U, 3U}) {
      SCOPED_TRACE(static_cast<int>(operation));
      SCOPED_TRACE(failed_ordinal);
      auto source = std::make_shared<SinglePassIteratorErrorHolder>(
          holder_, failed_ordinal, operation);
      auto two_pass = IndexHelper::MakeTwoPassHolder(source);
      ASSERT_NE(nullptr, two_pass);
      ASSERT_NE(source.get(), two_pass.get());
      auto first = two_pass->create_iterator();
      ASSERT_NE(nullptr, first);
      size_t read_count = 0;
      for (; first->is_valid(); first->next()) {
        (void)first->key();
        if (first->status() != 0) {
          break;
        }
        (void)first->data();
        if (first->status() != 0) {
          break;
        }
        ++read_count;
      }
      EXPECT_EQ(failed_ordinal, read_count);
      EXPECT_EQ(IndexError_ReadData, first->status());
      first.reset();

      auto second = two_pass->create_iterator();
      ASSERT_NE(nullptr, second);
      // The first pass may have cached a prefix, but it is not a valid input.
      EXPECT_FALSE(second->is_valid());
      EXPECT_EQ(IndexError_ReadData, second->status());
    }
  }
}

TEST_F(IVFBuilderTest, TwoPassHolderKeepsSuccessfulSecondPass) {
  prepare_index_holder(0, 8);
  auto source = std::make_shared<SinglePassIteratorErrorHolder>(
      holder_, holder_->count() + 1, IteratorErrorOperation::kValidity);
  auto two_pass = IndexHelper::MakeTwoPassHolder(source);
  ASSERT_NE(nullptr, two_pass);
  ASSERT_NE(source.get(), two_pass.get());
  for (size_t pass = 0; pass < 2; ++pass) {
    SCOPED_TRACE(pass);
    auto iter = two_pass->create_iterator();
    ASSERT_NE(nullptr, iter);
    size_t read_count = 0;
    for (; iter->is_valid(); iter->next()) {
      EXPECT_EQ(read_count, iter->key());
      const auto *data = static_cast<const float *>(iter->data());
      ASSERT_NE(nullptr, data);
      EXPECT_FLOAT_EQ(static_cast<float>(read_count), data[0]);
      EXPECT_EQ(0, iter->status());
      ++read_count;
    }
    EXPECT_EQ(holder_->count(), read_count);
    EXPECT_EQ(0, iter->status());
  }
}

TEST_F(IVFBuilderTest, MaterializationPreservesIteratorReadError) {
  dimension_ = 16;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  prepare_index_holder(0, 128);
  threads_ = std::make_shared<SingleQueueIndexThreads>(1, false);
  for (auto operation :
       {IteratorErrorOperation::kValidity, IteratorErrorOperation::kKey,
        IteratorErrorOperation::kData}) {
    for (size_t failed_ordinal : {0U, 65U}) {
      SCOPED_TRACE(static_cast<int>(operation));
      SCOPED_TRACE(failed_ordinal);
      IVFBuilder builder;
      ASSERT_EQ(0, builder.init(index_meta_, params_));
      ASSERT_EQ(0, builder.train(threads_, holder_));
      auto failing = std::make_shared<IteratorErrorHolder>(
          holder_, failed_ordinal, operation);
      ASSERT_EQ(IndexError_ReadData, builder.build(threads_, failing));
      // The partial copy must not become a successfully built index.
      EXPECT_EQ(IndexError_Runtime, builder.dump(nullptr));
    }
  }
}

TEST_F(IVFBuilderTest, ConvertedIteratorsPreserveSourceReadError) {
  prepare_index_holder(0, 8);
  for (const char *name : {"HalfFloatConverter", "CosineFp32Converter"}) {
    for (auto operation :
         {IteratorErrorOperation::kValidity, IteratorErrorOperation::kKey,
          IteratorErrorOperation::kData}) {
      for (size_t failed_ordinal : {0U, 3U}) {
        SCOPED_TRACE(name);
        SCOPED_TRACE(static_cast<int>(operation));
        SCOPED_TRACE(failed_ordinal);
        auto converter = IndexFactory::CreateConverter(name);
        ASSERT_NE(nullptr, converter);
        ASSERT_EQ(0, converter->init(index_meta_, Params{}));
        auto failing = std::make_shared<IteratorErrorHolder>(
            holder_, failed_ordinal, operation);
        ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, failing));
        auto converted = converter->result();
        ASSERT_NE(nullptr, converted);
        auto iter = converted->create_iterator();
        ASSERT_NE(nullptr, iter);
        size_t read_count = 0;
        for (; iter->is_valid(); iter->next()) {
          (void)iter->key();
          if (iter->status() != 0) {
            break;
          }
          ASSERT_NE(nullptr, iter->data());
          if (iter->status() != 0) {
            break;
          }
          ++read_count;
        }
        EXPECT_EQ(failed_ordinal, read_count);
        EXPECT_EQ(IndexError_ReadData, iter->status());
        iter->next();
        EXPECT_FALSE(iter->is_valid());
        EXPECT_EQ(IndexError_ReadData, iter->status());
      }
    }
  }
}

// Defer execution until wait_finish() to exercise the worst case: producers
// outrun all consumers. Track how many copied-vector batches remain live.
class DeferredLabelThreads : public IndexThreads {
 public:
  class Group : public IndexThreads::TaskGroup {
   public:
    void submit(ClosureHandler &&task) override {
      tasks_.emplace_back(std::move(task));
      max_pending = std::max(max_pending, tasks_.size());
      ++submitted;
    }
    bool is_finished() const override {
      return tasks_.empty();
    }
    void wait_finish() override {
      for (auto &task : tasks_) {
        task->run();
      }
      tasks_.clear();
    }
    size_t max_pending{0};
    size_t submitted{0};

   private:
    std::vector<ClosureHandler> tasks_;
  };

  size_t count() const override {
    return 1;
  }
  int indexof_this() const override {
    return 0;
  }
  void stop() override {}
  void submit(ClosureHandler &&task) override {
    task->run();
  }
  IndexThreads::TaskGroup::Pointer make_group() override {
    auto group = std::make_shared<Group>();
    groups.emplace_back(group);
    return group;
  }
  std::vector<std::shared_ptr<Group>> groups;
};

TEST_F(IVFBuilderTest, LabelQueueIsBoundedForHighDimensionalVectors) {
  constexpr uint32_t kVectorCount = 4103;
  for (const bool convert : {false, true}) {
    SCOPED_TRACE(convert ? "FP16" : "FP32");
    dimension_ = 1024;
    index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
    params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
    params_.set(PARAM_IVF_BUILDER_TRAIN_SAMPLE_COUNT, 8u);
    if (convert) {
      params_.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "HalfFloatConverter");
    }
    // Covers multiple byte windows and a final partial batch. The converter
    // also exercises a holder whose iterator reuses a temporary vector.
    // Keep coordinates bounded to avoid native FP16 distance accumulation
    // overflow: this test targets queue memory, not numerical range limits.
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
            dimension_);
    for (uint32_t i = 0; i < kVectorCount; ++i) {
      NumericalVector<float> vec(dimension_);
      for (size_t j = 0; j < dimension_; ++j) {
        vec[j] = static_cast<float>(i) / kVectorCount;
      }
      holder->emplace(i, vec);
    }
    holder_ = std::move(holder);
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(index_meta_, params_));
    ASSERT_EQ(0,
              builder.train(std::make_shared<SingleQueueIndexThreads>(1, false),
                            holder_));
    auto deferred = std::make_shared<DeferredLabelThreads>();
    ASSERT_EQ(0, builder.build(deferred, holder_));
    ASSERT_FALSE(deferred->groups.empty());
    const auto &group = deferred->groups.front();
    // 4 MiB holds 1024 FP32 vectors or 2048 FP16 vectors at this dimension.
    EXPECT_LE(group->max_pending, convert ? 205u : 103u);
    EXPECT_GT(group->submitted, group->max_pending);
    EXPECT_TRUE(group->is_finished());
    EXPECT_EQ(kVectorCount, builder.stats().built_count());
    auto dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create("label_queue"));
    ASSERT_EQ(0, builder.dump(dumper));
    EXPECT_EQ(kVectorCount, builder.stats().dumped_count());
    ASSERT_EQ(0, dumper->close());
  }
}

// An optional ordinal reader deliberately returns one reused, unaligned
// buffer. The builder must retain the source, not retain these data pointers.
class OrdinalTestHolder : public IndexHolder, public OrdinalAccessHolder {
 public:
  explicit OrdinalTestHolder(IndexHolder::Pointer delegate)
      : delegate_(std::move(delegate)) {
    for (auto iter = delegate_->create_iterator(); iter->is_valid();
         iter->next()) {
      keys_.push_back(iter->key());
      vectors_.emplace_back(static_cast<const char *>(iter->data()),
                            delegate_->element_size());
    }
  }
  size_t count() const override {
    return vectors_.size() + extra_count;
  }
  size_t dimension() const override {
    return delegate_->dimension();
  }
  IndexMeta::DataType data_type() const override {
    return delegate_->data_type();
  }
  size_t element_size() const override {
    return delegate_->element_size();
  }
  bool multipass() const override {
    return true;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    ++iterations;
    return delegate_->create_iterator();
  }
  class OrdinalReader : public OrdinalAccessHolder::Reader {
   public:
    explicit OrdinalReader(OrdinalTestHolder *owner) : owner_(owner) {}
    int read(size_t id, uint64_t *key, const void **data) override {
      ++owner_->reads;
      if (owner_->read_error) return owner_->read_error;
      if (id >= owner_->vectors_.size()) return IndexError_OutOfRange;
      if (owner_->null_data) {
        *data = nullptr;
        return 0;
      }
      *key = owner_->keys_[id];
      if (owner_->aligned_data) {
        // Natural float alignment is enough; deliberately avoid SIMD alignment.
        aligned_buffer_.resize(owner_->dimension() + 1);
        float *values = aligned_buffer_.data();
        if (reinterpret_cast<uintptr_t>(values) % 16 == 0) {
          ++values;
        }
        std::memcpy(values, owner_->vectors_[id].data(),
                    owner_->element_size());
        *data = values;
      } else {
        buffer_.assign(1, '\0');
        buffer_.append(owner_->vectors_[id]);
        *data = buffer_.data() + 1;
      }
      return 0;
    }
    void reset() override {
      ++owner_->resets;
      buffer_.clear();
      aligned_buffer_.clear();
    }

   private:
    OrdinalTestHolder *owner_;
    std::string buffer_;
    std::vector<float> aligned_buffer_;
  };
  int create_ordinal_reader(
      OrdinalAccessHolder::Reader::Pointer *reader) override {
    ++reader_creations;
    if (create_error) return create_error;
    if (null_reader) return 0;
    reader->reset(new OrdinalReader(this));
    return 0;
  }
  size_t reader_creations{0}, iterations{0}, reads{0}, resets{0},
      extra_count{0};
  int create_error{0}, read_error{0};
  bool null_reader{false}, null_data{false}, aligned_data{false};

 private:
  IndexHolder::Pointer delegate_;
  std::vector<uint64_t> keys_;
  std::vector<std::string> vectors_;
};

TEST_F(IVFBuilderTest, HalfFloatOrdinalReaderConvertsLazilyAndRetainsSource) {
  // Exercise vectorized conversion plus a tail, negative/fractional values,
  // and reused source storage with natural and byte-unaligned addresses.
  dimension_ = 129;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  auto input =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dimension_);
  for (size_t i = 0; i < 7; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = (static_cast<float>(i * 13 + j) - 60.0f) / 1000.0f;
    }
    input->emplace(100 + i * 3, vec);
  }
  auto source = std::make_shared<OrdinalTestHolder>(input);
  std::weak_ptr<OrdinalTestHolder> weak = source;
  auto converter = IndexFactory::CreateConverter("HalfFloatConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(index_meta_, Params()));
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, source));
  auto converted = converter->result();
  auto *ordinal = dynamic_cast<OrdinalAccessHolder *>(converted.get());
  ASSERT_NE(nullptr, ordinal);
  std::vector<std::string> expected;
  for (auto iter = converted->create_iterator(); iter->is_valid();
       iter->next()) {
    expected.emplace_back(static_cast<const char *>(iter->data()),
                          converted->element_size());
  }
  source->iterations = 0;
  OrdinalAccessHolder::Reader::Pointer reader, other;
  ASSERT_EQ(0, ordinal->create_ordinal_reader(&reader));
  ASSERT_EQ(0, ordinal->create_ordinal_reader(&other));
  EXPECT_EQ(0u, source->iterations);
  EXPECT_EQ(0u, source->reads);
  EXPECT_EQ(2u, source->reader_creations);
  source.reset();
  converted.reset();
  converter.reset();
  EXPECT_FALSE(weak.expired());
  for (bool aligned_data : {true, false, true}) {
    SCOPED_TRACE(aligned_data);
    weak.lock()->aligned_data = aligned_data;
    for (size_t id : {6u, 0u, 3u, 0u}) {
      uint64_t key = 0;
      const void *data = nullptr;
      ASSERT_EQ(0, reader->read(id, &key, &data));
      ASSERT_NE(nullptr, data);
      EXPECT_EQ(100 + id * 3, key);
      EXPECT_EQ(expected[id], std::string(static_cast<const char *>(data),
                                          expected[id].size()));
    }
  }
  uint64_t key = 0;
  const void *data = nullptr, *other_data = nullptr;
  ASSERT_EQ(0, reader->read(1, &key, &data));
  ASSERT_EQ(0, other->read(2, &key, &other_data));
  // Another reader must not overwrite the first reader's converted output.
  EXPECT_EQ(expected[1],
            std::string(static_cast<const char *>(data), expected[1].size()));
  reader->reset();
  EXPECT_EQ(1u, weak.lock()->resets);
  ASSERT_EQ(0, reader->read(6, &key, &data));
  EXPECT_EQ(expected[6],
            std::string(static_cast<const char *>(data), expected[6].size()));
  other.reset();
  reader.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(IVFBuilderTest, HalfFloatOrdinalReaderPreservesErrorsAndFallback) {
  prepare_index_holder(100, 7);
  auto converter = IndexFactory::CreateConverter("HalfFloatConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(index_meta_, Params()));
  auto source = std::make_shared<OrdinalTestHolder>(holder_);
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, source));
  auto converted = converter->result();
  auto *ordinal = dynamic_cast<OrdinalAccessHolder *>(converted.get());
  ASSERT_NE(nullptr, ordinal);
  OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(0, ordinal->create_ordinal_reader(&reader));
  auto *original = reader.get();
  EXPECT_EQ(IndexError_InvalidArgument,
            ordinal->create_ordinal_reader(nullptr));
  for (int error :
       {IndexError_NotImplemented, IndexError_Runtime, IndexError_Canceled}) {
    source->create_error = error;
    EXPECT_EQ(error, ordinal->create_ordinal_reader(&reader));
    EXPECT_EQ(original, reader.get());
  }
  source->create_error = 0;
  source->null_reader = true;
  EXPECT_EQ(IndexError_Runtime, ordinal->create_ordinal_reader(&reader));
  EXPECT_EQ(original, reader.get());
  source->null_reader = false;
  EXPECT_EQ(0u, source->iterations);
  EXPECT_EQ(0u, source->reads);

  uint64_t key = 999;
  const void *data = nullptr;
  EXPECT_EQ(IndexError_InvalidArgument, reader->read(0, nullptr, &data));
  EXPECT_EQ(IndexError_InvalidArgument, reader->read(0, &key, nullptr));
  EXPECT_EQ(IndexError_OutOfRange, reader->read(7, &key, &data));
  EXPECT_EQ(0u, source->reads);
  for (int error : {IndexError_Runtime, IndexError_Canceled}) {
    source->read_error = error;
    EXPECT_EQ(error, reader->read(0, &key, &data));
    EXPECT_EQ(nullptr, data);
    EXPECT_EQ(999u, key);
  }
  source->read_error = 0;
  source->null_data = true;
  EXPECT_EQ(IndexError_Runtime, reader->read(0, &key, &data));
  EXPECT_EQ(nullptr, data);
  source->null_data = false;
  ASSERT_EQ(0, reader->read(0, &key, &data));
  ASSERT_NE(nullptr, data);

  // An ordinary multipass source has no ordinal capability. Declining it
  // must leave the existing reader intact and sequential conversion usable.
  ASSERT_EQ(0, converter->transform(holder_));
  auto fallback = converter->result();
  auto *unsupported = dynamic_cast<OrdinalAccessHolder *>(fallback.get());
  ASSERT_NE(nullptr, unsupported);
  EXPECT_EQ(IndexError_NotImplemented,
            unsupported->create_ordinal_reader(&reader));
  EXPECT_EQ(original, reader.get());
  auto iter = fallback->create_iterator();
  size_t count = 0;
  for (; iter->is_valid(); iter->next()) {
    ASSERT_NE(nullptr, iter->data());
    ++count;
  }
  EXPECT_EQ(7u, count);
}

TEST_F(IVFBuilderTest, HalfFloatOrdinalSourceIsBorrowedAndDumpCanRetry) {
  for (bool store_original : {false, true}) {
    // Exactly representable, bounded coordinates avoid FP16 distance overflow.
    prepare_index_holder(100, 7);
    params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
    params_.set(PARAM_IVF_BUILDER_TRAIN_SAMPLE_COUNT, 4u);
    params_.set(PARAM_IVF_BUILDER_STORE_ORIGINAL_FEATURES, store_original);
    auto source = std::make_shared<OrdinalTestHolder>(holder_);
    std::weak_ptr<OrdinalTestHolder> weak = source;
    auto converter = IndexFactory::CreateConverter("HalfFloatConverter");
    ASSERT_NE(nullptr, converter);
    ASSERT_EQ(0, converter->init(index_meta_, Params()));
    ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, source));
    auto converted = converter->result();
    auto meta = converter->meta();
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(meta, params_));
    ASSERT_EQ(0, builder.train(threads_, converted));
    source->iterations = 0;
    source->create_error = IndexError_Runtime;
    EXPECT_EQ(IndexError_Runtime, builder.build(threads_, converted));
    EXPECT_EQ(0u, source->iterations);
    source->create_error = 0;
    source->reader_creations = 0;
    ASSERT_EQ(0, builder.build(threads_, converted));
    EXPECT_EQ(1u, source->reader_creations);
    EXPECT_EQ(1u, source->iterations);  // Labels only, no materialization pass.
    EXPECT_EQ(0u, source->reads);

    source->read_error = IndexError_Runtime;
    auto failed_dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_EQ(0, failed_dumper->create("fp16_ordinal_failure"));
    EXPECT_EQ(IndexError_Runtime, builder.dump(failed_dumper));
    EXPECT_EQ(1u, source->resets);
    EXPECT_EQ(0u, builder.stats().dumped_count());
    ASSERT_EQ(0, failed_dumper->close());
    source->read_error = 0;
    source->reads = 0;
    source->resets = 0;
    source.reset();
    converter.reset();
    converted.reset();
    EXPECT_FALSE(weak.expired());
    for (size_t pass = 1; pass <= 2; ++pass) {
      const std::string path = "ivf_fp16_ordinal_source.index";
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      ASSERT_EQ(0, dumper->create(path));
      ASSERT_EQ(0, builder.dump(dumper));
      ASSERT_EQ(0, dumper->close());
      EXPECT_EQ(7u * pass * (store_original ? 2 : 1), weak.lock()->reads);
      EXPECT_EQ(pass, weak.lock()->resets);
      auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
      ASSERT_EQ(0, storage->init(Params()));
      ASSERT_EQ(0, storage->open(path, false));
      auto streamer = IndexFactory::CreateStreamer("IVFStreamer");
      ASSERT_EQ(0, streamer->init(meta, Params()));
      ASSERT_EQ(0, streamer->open(storage));
      auto provider = streamer->create_provider();
      ASSERT_NE(nullptr, provider);
      auto expected_converter =
          IndexFactory::CreateConverter("HalfFloatConverter");
      ASSERT_EQ(0, expected_converter->init(index_meta_, Params()));
      ASSERT_EQ(0,
                IndexConverter::TrainAndTransform(expected_converter, holder_));
      auto expected = expected_converter->result();
      for (auto iter = expected->create_iterator(); iter->is_valid();
           iter->next()) {
        const void *actual = provider->get_vector(iter->key());
        ASSERT_NE(nullptr, actual);
        EXPECT_EQ(0,
                  std::memcmp(actual, iter->data(), expected->element_size()));
      }
      provider.reset();
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, storage->close());
      File::RemovePath(path);
    }
    ASSERT_EQ(0, builder.cleanup());
    EXPECT_TRUE(weak.expired());
  }
}

TEST_F(IVFBuilderTest, OrdinalSourceIsReadAtDumpAndRetainedForRepeatedDumps) {
  for (bool turbo : {false, true}) {
    SCOPED_TRACE(turbo);
    for (bool store_original : {false, true}) {
      prepare_index_holder(100, 103);
      params_.set(PARAM_IVF_BUILDER_STORE_ORIGINAL_FEATURES, store_original);
      IVFBuilder builder;
      auto meta = index_meta_;
      zvec::turbo::Quantizer::Pointer quantizer;
      if (turbo) {
        meta.set_quantizer("Fp32Quantizer", 0, Params());
        quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
        ASSERT_NE(nullptr, quantizer);
        ASSERT_EQ(0, quantizer->init(meta, Params()));
      }
      ASSERT_EQ(0, builder.init(meta, params_, quantizer));
      ASSERT_EQ(0, builder.train(threads_, holder_));
      auto source = std::make_shared<OrdinalTestHolder>(holder_);
      std::weak_ptr<OrdinalTestHolder> weak = source;
      ASSERT_EQ(0, builder.build(threads_, source));
      EXPECT_EQ(1u, source->reader_creations);
      EXPECT_EQ(1u, source->iterations);
      EXPECT_EQ(0u, source->reads);
      source.reset();
      EXPECT_FALSE(weak.expired());
      for (size_t pass = 1; pass <= 2; ++pass) {
        const std::string path = "ivf_ordinal_source.index";
        auto dumper = IndexFactory::CreateDumper("FileDumper");
        ASSERT_EQ(0, dumper->create(path));
        ASSERT_EQ(0, builder.dump(dumper));
        ASSERT_EQ(0, dumper->close());
        EXPECT_EQ(103u * pass * (store_original ? 2 : 1), weak.lock()->reads);
        EXPECT_EQ(pass, weak.lock()->resets);
        auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
        ASSERT_EQ(0, storage->init(Params()));
        ASSERT_EQ(0, storage->open(path, false));
        auto streamer = IndexFactory::CreateStreamer("IVFStreamer");
        ASSERT_EQ(0, streamer->init(index_meta_, Params()));
        ASSERT_EQ(0, streamer->open(storage));
        auto provider = streamer->create_provider();
        ASSERT_NE(nullptr, provider);
        for (auto iter = holder_->create_iterator(); iter->is_valid();
             iter->next()) {
          const void *actual = provider->get_vector(iter->key());
          ASSERT_NE(nullptr, actual);
          EXPECT_EQ(0,
                    std::memcmp(actual, iter->data(), holder_->element_size()));
        }
        provider.reset();
        ASSERT_EQ(0, streamer->close());
        ASSERT_EQ(0, storage->close());
        File::RemovePath(path);
      }
      ASSERT_EQ(0, builder.cleanup());
      EXPECT_TRUE(weak.expired());
    }
  }
}

TEST_F(IVFBuilderTest, OrdinalSourceErrorsAreNotMaterializedOrSilentlyDumped) {
  prepare_index_holder(100, 103);
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(index_meta_, params_));
  ASSERT_EQ(0, builder.train(threads_, holder_));
  auto source = std::make_shared<OrdinalTestHolder>(holder_);
  source->create_error = IndexError_Runtime;
  EXPECT_EQ(IndexError_Runtime, builder.build(threads_, source));
  EXPECT_EQ(0u, source->iterations);
  source->create_error = 0;
  source->extra_count = 1;
  EXPECT_EQ(IndexError_Mismatch, builder.build(threads_, source));
  source->extra_count = 0;
  ASSERT_EQ(0, builder.build(threads_, source));
  source->read_error = IndexError_Runtime;
  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
  ASSERT_EQ(0, dumper->create("ordinal_error"));
  EXPECT_EQ(IndexError_Runtime, builder.dump(dumper));
  EXPECT_EQ(1u, source->resets);
  EXPECT_EQ(0u, builder.stats().dumped_count());
  ASSERT_EQ(0, dumper->close());
}

TEST_F(IVFBuilderTest, OrdinalSourceFallsBackForUnsupportedTransforms) {
  for (int mode : {0, 1, 2}) {
    Params params = params_;
    if (mode == 1)
      params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "HalfFloatConverter");
    if (mode == 2)
      params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
    prepare_index_holder(100, 103);
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(index_meta_, params));
    ASSERT_EQ(0, builder.train(threads_, holder_));
    auto source = std::make_shared<OrdinalTestHolder>(holder_);
    source->create_error = IndexError_NotImplemented;
    std::weak_ptr<OrdinalTestHolder> weak = source;
    ASSERT_EQ(0, builder.build(threads_, source));
    EXPECT_EQ(mode == 0 ? 1u : 0u, source->reader_creations);
    source.reset();
    EXPECT_TRUE(weak.expired());
    auto dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_EQ(0, dumper->create("ordinal_fallback"));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    EXPECT_EQ(103u, builder.stats().dumped_count());
  }
}

TEST_F(IVFBuilderTest, TestInitSuccess) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
}

TEST_F(IVFBuilderTest, TestInitFailedWithInvalidMetric) {
  IVFBuilder builder;
  index_meta_.set_metric("invalid", 0, Params());
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(IndexError_NoExist, ret);
}

TEST_F(IVFBuilderTest, TestInitFailedWithInvalidCentroidsNum) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(IndexError_InvalidArgument, ret);
}

TEST_F(IVFBuilderTest, TestTrainWithHolder1Level) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_GT(centroid_index->centroids_count(), 0u);
}

TEST_F(IVFBuilderTest, TestTrainWithHolder2Level) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 8);
}

TEST_F(IVFBuilderTest, TestTrainWithTrainer2Level) {
  IndexTrainer::Pointer trainer =
      IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_TRUE(!!trainer);

  prepare_index_holder(0, 1000);

  Params params;
  params.set("zvec.stratified.trainer.cluster_count", "4*2");
  ASSERT_EQ(0, trainer->init(index_meta_, params));
  ASSERT_EQ(0, trainer->train(threads_, holder_));

  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);


  ret = builder.train(trainer);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 8);
}

TEST_F(IVFBuilderTest, TestTrainWithTrainer1Level) {
  IVFBuilder builder;

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  IndexTrainer::Pointer trainer =
      IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_TRUE(!!trainer);

  prepare_index_holder(0, 1000);

  Params params1;
  params1.set("zvec.stratified.trainer.cluster_count", "4");
  ASSERT_EQ(0, trainer->init(index_meta_, params1));
  ASSERT_EQ(0, trainer->train(threads_, holder_));

  ret = builder.train(trainer);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 4);
}

TEST_F(IVFBuilderTest, TestBuildWith2Level) {
  IVFBuilder builder;

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  EXPECT_EQ((size_t)1000, builder.stats().built_count());
}

TEST_F(IVFBuilderTest, TestBuildWith1Level) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  EXPECT_EQ((size_t)1000, builder.stats().built_count());
}

TEST_F(IVFBuilderTest, TestDump) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

#if 0
TEST_F(IVFBuilderTest, TestBuildWithNoEnoughMemory)
{
    IVFBuilder builder;
    Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

    dimension_ = 256;
    index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);

    int ret = builder.init(index_meta_, params);
    EXPECT_EQ(0, ret);

    prepare_index_holder(0, 1000);

    ret = builder.train(threads_, holder_);
    EXPECT_EQ(0, ret);

    ret = builder.build(threads_, holder_);
    EXPECT_EQ(IndexError_IndexFull, ret);
}
#endif

TEST_F(IVFBuilderTest, TestBuildWithEnoughMemory) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

#if 0
TEST_F(IVFBuilderTest, TestBuildWithRowMajorAndNoEnoughMemory)
{
    IVFBuilder builder;
    Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

    dimension_ = 256;
    index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
    index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

    int ret = builder.init(index_meta_, params);
    EXPECT_EQ(0, ret);

    prepare_index_holder(0, 1000);

    ret = builder.train(threads_, holder_);
    EXPECT_EQ(0, ret);

    ret = builder.build(threads_, holder_);
    EXPECT_EQ(IndexError_IndexFull, ret);
}
#endif

TEST_F(IVFBuilderTest, TestBuildWithRowMajorAndMemory) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

TEST_F(IVFBuilderTest, TestBuildWithEmptyCentroid) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  size_t doc_cnt = 10;

  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 1.0f;
    }
    holder->emplace(i, vec);
  }
  holder_.reset(holder);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)10, builder.stats().built_count());
  EXPECT_EQ((size_t)10, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

TEST_F(IVFBuilderTest, TestTrainClusterParams) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  prepare_index_holder(0, 1000);
  EXPECT_EQ(0, builder.init(index_meta_, params));
  EXPECT_EQ(0, builder.train(threads_, holder_));
  EXPECT_EQ(0, builder.build(threads_, holder_));

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  EXPECT_EQ(0, dumper->create("test.index"));
  EXPECT_EQ(0, builder.dump(dumper));
}

TEST_F(IVFBuilderTest, TestBuildWithConverterClass) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "HalfFloatConverter");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_GT(centroid_index->centroids_count(), 0u);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

TEST_F(IVFBuilderTest, TestBuildWithConverterClassMultiLevel) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "HalfFloatConverter");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 8);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create("./ivf_converter_test.index");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());
  File::RemovePath("./ivf_converter_test.index");
}

TEST_F(IVFBuilderTest, TestIndexThreads) {
  IndexBuilder::Pointer builder1 = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder1, nullptr);
  IndexBuilder::Pointer builder2 = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder2, nullptr);

  size_t dim = 128UL;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  std::srand(Realtime::MilliSeconds());
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  ASSERT_EQ(0, builder1->init(meta, params));
  ASSERT_EQ(0, builder2->init(meta, params));

  auto threads =
      std::make_shared<SingleQueueIndexThreads>(std::rand() % 4, false);
  auto build_index1 = [&]() {
    ASSERT_EQ(0, builder1->train(threads, holder));
    ASSERT_EQ(0, builder1->build(threads, holder));
  };
  auto build_index2 = [&]() {
    ASSERT_EQ(0, builder2->train(threads, holder));
    ASSERT_EQ(0, builder2->build(threads, holder));
  };

  auto t1 = std::async(std::launch::async, build_index1);
  auto t2 = std::async(std::launch::async, build_index2);
  t1.wait();
  t2.wait();


  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  std::string path = "./hc_index";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder1->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder2->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats1 = builder1->stats();
  ASSERT_EQ(doc_cnt, stats1.built_count());
  auto &stats2 = builder2->stats();
  ASSERT_EQ(doc_cnt, stats2.built_count());
}

namespace zvec {
namespace core {
namespace {

TEST(IVFTurboConfiguration, RejectsColumnOrderAndLegacyPostingQuantization) {
  constexpr uint32_t kDimension = 18;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  meta.set_quantizer("Int8Quantizer", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");

  {
    IVFBuilder builder;
    auto column_meta = meta;
    column_meta.set_major_order(IndexMeta::MO_COLUMN);
    EXPECT_NE(0, builder.init(column_meta, params, quantizer));
  }
  {
    IVFBuilder builder;
    auto conflicting_params = params;
    conflicting_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS,
                           "Int8QuantizerConverter");
    EXPECT_NE(0, builder.init(meta, conflicting_params, quantizer));
  }
  {
    IVFBuilder builder;
    auto conflicting_params = params;
    conflicting_params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
    EXPECT_NE(0, builder.init(meta, conflicting_params, quantizer));
  }
}

}  // namespace
}  // namespace core
}  // namespace zvec
