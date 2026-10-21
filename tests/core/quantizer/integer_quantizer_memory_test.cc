// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
// Generate on demand so the fixture itself never materializes the dataset.
class GeneratedHolder : public IndexHolder {
 public:
  explicit GeneratedHolder(bool repeatable, size_t fail_at = SIZE_MAX)
      : repeatable_(repeatable), fail_at_(fail_at) {}
  size_t count() const override {
    return 40003;
  }
  size_t dimension() const override {
    return 64;
  }
  size_t element_size() const override {
    return 64 * sizeof(float);
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  bool multipass() const override {
    return repeatable_;
  }
  class Cursor : public IndexHolder::Iterator {
   public:
    explicit Cursor(GeneratedHolder &owner) : owner_(owner) {}
    const void *data() const override {
      owner_.peak_charge = std::max(
          owner_.peak_charge, ailego::PoolPayloadBudget::get_instance().used());
      for (size_t d = 0; d < row_.size(); ++d)
        row_[d] =
            (static_cast<float>((id_ * 17 + d * 31) % 997) - 498.0f) / 997.0f;
      return row_.data();
    }
    bool is_valid() const override {
      return id_ < owner_.count() && status() == 0;
    }
    int status() const override {
      return id_ >= owner_.fail_at_ ? IndexError_ReadData : 0;
    }
    uint64_t key() const override {
      return id_;
    }
    void next() override {
      ++id_;
    }

   private:
    GeneratedHolder &owner_;
    size_t id_{0};
    mutable std::array<float, 64> row_{};
  };
  Iterator::Pointer create_iterator() override {
    if (!repeatable_ && iterations != 0) return nullptr;
    ++iterations;
    return std::make_unique<Cursor>(*this);
  }
  size_t iterations{0};
  size_t peak_charge{0};

 private:
  bool repeatable_;
  size_t fail_at_;
};

TEST(IntegerQuantizerMemoryTest, OnePassMatchesMultipassWithAndWithoutPool) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  const size_t budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(
          40003 * 64 * sizeof(float) / ailego::kVectorPageSize + 512, true) +
      64 * ailego::kVectorPageSize;
  for (const auto *name :
       {"Int8QuantizerConverter", "Int4QuantizerConverter"}) {
    std::string expected;
    for (size_t mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(mode);
      ASSERT_EQ(0, pool.init(mode == 2 ? budget : 0));
      auto converter = IndexFactory::CreateConverter(name);
      ASSERT_NE(nullptr, converter);
      ASSERT_EQ(0, converter->init(meta, {}));
      auto holder = std::make_shared<GeneratedHolder>(mode == 0);
      ASSERT_EQ(0, converter->train(holder));
      EXPECT_EQ(mode == 0 ? 2u : 1u, holder->iterations);
      EXPECT_EQ(holder->count(), converter->stats().trained_count());
      std::string serialized;
      converter->meta().serialize(&serialized);
      if (!mode)
        expected = serialized;
      else
        EXPECT_EQ(expected, serialized);
      if (mode == 2) {
        EXPECT_GT(holder->peak_charge, 0u);
        EXPECT_LE(holder->peak_charge, pool.capacity());
      }
      EXPECT_EQ(0u, pool.used());
    }
  }
}

TEST(IntegerQuantizerMemoryTest, PropagatesInputAndAdmissionFailures) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  const size_t writable_budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(512, true) +
      64 * ailego::kVectorPageSize;
  for (const size_t budget : {size_t{0}, writable_budget, size_t{1}}) {
    ASSERT_EQ(0, pool.init(budget));
    auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
    ASSERT_EQ(0, converter->init(IndexMeta(IndexMeta::DT_FP32, 64), {}));
    const int ret =
        converter->train(std::make_shared<GeneratedHolder>(false, 17));
    if (budget == 1)
      EXPECT_NE(0, ret);
    else
      EXPECT_EQ(IndexError_ReadData, ret);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0u, converter->stats().trained_count());
  }
}
}  // namespace
}  // namespace zvec::core
