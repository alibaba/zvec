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
#include <atomic>
#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include "ailego/pattern/defer.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_memory.h"
#include "zvec/core/framework/index_provider.h"
#include "ivf_rabitq_builder.h"
#include "ivf_rabitq_entity.h"
#include "ivf_rabitq_params.h"

namespace zvec::core {
namespace {

class ObservedBuildInput : public IndexHolder, public OrdinalAccessHolder {
 public:
  explicit ObservedBuildInput(IndexHolder::Pointer input)
      : input_(std::move(input)) {}
  size_t count() const override {
    return input_->count();
  }
  size_t dimension() const override {
    return input_->dimension();
  }
  size_t element_size() const override {
    return input_->element_size();
  }
  IndexMeta::DataType data_type() const override {
    return input_->data_type();
  }
  bool multipass() const override {
    return true;
  }
  Iterator::Pointer create_iterator() override {
    ++iterator_calls;
    return nullptr;  // build() must not materialize an ordinal-capable source.
  }
  std::atomic<size_t> reads{0};
  size_t iterator_calls{0};
  size_t fail_after{std::numeric_limits<size_t>::max()};

  class InputReader : public Reader {
   public:
    InputReader(ObservedBuildInput *input, Reader::Pointer reader)
        : input_(input), reader_(std::move(reader)), row_(input->dimension()) {}
    int read(size_t ordinal, uint64_t *key, const void **data) override {
      if (input_->reads.fetch_add(1) >= input_->fail_after)
        return IndexError_ReadData;
      const void *source = nullptr;
      int ret = reader_->read(ordinal, key, &source);
      if (ret != 0) return ret;
      std::memcpy(row_.data(), source, input_->element_size());
      *data = row_.data();  // Valid only until the next read on this reader.
      return 0;
    }
    void reset() override {
      reader_->reset();
    }

   private:
    ObservedBuildInput *input_;
    Reader::Pointer reader_;
    std::vector<float> row_;
  };
  int create_ordinal_reader(Reader::Pointer *out) override {
    Reader::Pointer reader;
    int ret = dynamic_cast<OrdinalAccessHolder *>(input_.get())
                  ->create_ordinal_reader(&reader);
    if (ret == 0) *out = std::make_unique<InputReader>(this, std::move(reader));
    return ret;
  }

 private:
  IndexHolder::Pointer input_;
};

TEST(IvfRabitqBuildMemoryTest,
     BorrowedRowsPropagateErrorsAndSurviveSourceRelease) {
  constexpr size_t kDim = 128;
  constexpr size_t kCount = 259;
  auto holder =
      std::make_shared<MultiPassIndexProvider<IndexMeta::DT_FP32>>(kDim);
  for (size_t i = 0; i < kCount; ++i) {
    ailego::NumericalVector<float> row(kDim);
    for (size_t d = 0; d < kDim; ++d)
      row[d] = float((i * 17 + d * 13) % 257) / 257.0f;
    ASSERT_TRUE(holder->emplace((uint64_t{1} << 40) + i % 7, std::move(row)));
  }
  IndexMeta meta(IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(PARAM_IVF_RABITQ_NLIST, 4U);
  IvfRabitqBuilder builder;
  auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
  ASSERT_EQ(builder.init(meta, params), 0);
  ASSERT_EQ(builder.train(threads, holder), 0);
  auto observed = std::make_shared<ObservedBuildInput>(holder);
  for (size_t fail_after : {size_t{0}, kCount}) {
    SCOPED_TRACE(fail_after);  // Assignment failure, then encoding failure.
    observed->reads = 0;
    observed->fail_after = fail_after;
    EXPECT_EQ(builder.build(threads, observed), IndexError_ReadData);
    EXPECT_EQ(builder.stats().built_count(), 0u);
  }
  observed->reads = 0;
  observed->fail_after = std::numeric_limits<size_t>::max();
  ASSERT_EQ(builder.build(threads, observed), 0);
  EXPECT_EQ(builder.stats().built_count(), kCount);
  EXPECT_EQ(observed->reads.load(), 2 * kCount);
  EXPECT_EQ(observed->iterator_calls, 0u);
  observed.reset();
  holder.reset();

  // Only encoded output/trained state may remain after build. Repeated dump
  // must work after releasing the input and the training converter.
  for (size_t attempt = 0; attempt < 2; ++attempt) {
    const std::string id = "ivf_rabitq_borrowed_build";
    AILEGO_DEFER([&]() { IndexMemory::Instance()->remove(id); });
    auto dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(dumper->init(ailego::Params()), 0);
    ASSERT_EQ(dumper->create(id), 0);
    ASSERT_EQ(builder.dump(dumper), 0);
    ASSERT_EQ(dumper->close(), 0);
    auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
    ASSERT_NE(storage, nullptr);
    ASSERT_EQ(storage->open(id, false), 0);
    IvfRabitqEntity entity;
    ASSERT_EQ(entity.load(storage), 0);
    ASSERT_EQ(entity.total_vector_count(), kCount);
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < kCount; ++i) keys.push_back(entity.get_key(i));
    std::sort(keys.begin(), keys.end());
    std::vector<uint64_t> expected;
    for (size_t i = 0; i < kCount; ++i)
      expected.push_back((uint64_t{1} << 40) + i % 7);
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(keys, expected);
  }
  EXPECT_EQ(builder.cleanup(), 0);
}

}  // namespace
}  // namespace zvec::core
