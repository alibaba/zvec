// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "utility/paged_rotation_matrix.h"
#include <array>
#include <gtest/gtest.h>
#include "utility/storage_segment_dumper.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class PagedRotationMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};
TEST_F(PagedRotationMatrixTest, MatrixLargerThanPoolRotatesAndReleases) {
  const size_t budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(1024, true) +
      32 * ailego::kVectorPageSize;
  ASSERT_LT(budget, 1024u * 1024u * sizeof(float));
  ASSERT_EQ(0, pool_.init(budget));
  {
    constexpr size_t dim = 1024;
    PagedRotationMatrix matrix;
    ASSERT_EQ(0,
              matrix.load(dim, dim, [](size_t offset, void *out, size_t bytes) {
                auto *values = static_cast<float *>(out);
                EXPECT_TRUE(ailego::BudgetedAllocator<float>::charged(values));
                const size_t row = offset / sizeof(float) / dim;
                std::fill(values, values + bytes / sizeof(float), 0.0f);
                values[(row + 1) % dim] = 1;
                return 0;
              }));
    ASSERT_TRUE(matrix.paged());
    std::vector<float> input(dim), rotated(dim), restored(dim);
    for (size_t i = 0; i < dim; ++i) input[i] = static_cast<float>(i);
    matrix.apply(input.data(), rotated.data());
    for (size_t i = 0; i < dim; ++i)
      EXPECT_EQ(input[i], rotated[(i + 1) % dim]);
    matrix.apply(rotated.data(), restored.data(), true);
    EXPECT_EQ(input, restored);
    EXPECT_LE(pool_.used(), pool_.capacity());
  }
  EXPECT_EQ(0u, pool_.used());
}
TEST_F(PagedRotationMatrixTest, QrPreservesNormAndInverse) {
  for (size_t budget : {size_t{0}, size_t{16 * 1024 * 1024}}) {
    ASSERT_EQ(0, pool_.init(budget));
    PagedRotationMatrix matrix;
    ASSERT_EQ(0, matrix.generate(64, 64));
    std::vector<float> input(64, 1.0f), output(64), restored(64);
    matrix.apply(input.data(), output.data());
    matrix.apply(output.data(), restored.data(), true);
    for (float value : restored) EXPECT_NEAR(1.0f, value, 1e-5f);
    PagedRotationMatrix loaded;
    ASSERT_EQ(0,
              loaded.load(64, 64, [&](size_t offset, void *out, size_t bytes) {
                return matrix.read(offset, out, bytes);
              }));
    std::vector<float> replay(64);
    loaded.apply(input.data(), replay.data());
    EXPECT_EQ(output, replay);
  }
}
TEST_F(PagedRotationMatrixTest, RectangularQrHandlesPartialTiles) {
  ASSERT_EQ(0, pool_.init(16 * 1024 * 1024));
  PagedRotationMatrix matrix;
  ASSERT_EQ(0, matrix.generate(39, 77));
  std::vector<float> input(39, 0.5f), output(77), restored(39);
  matrix.apply(input.data(), output.data());
  matrix.apply(output.data(), restored.data(), true);
  for (float value : restored) EXPECT_NEAR(0.5f, value, 1e-5f);
  EXPECT_EQ(39u * 77u * sizeof(float), matrix.size());
}

TEST_F(PagedRotationMatrixTest, ReadFailureAndTinyBudgetDoNotFallback) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  PagedRotationMatrix matrix;
  EXPECT_EQ(IndexError_ReadData,
            matrix.load(64, 64, [](size_t, void *, size_t) {
              return IndexError_ReadData;
            }));
  EXPECT_EQ(0u, matrix.size());
  ASSERT_EQ(0u, pool_.used());
  ASSERT_EQ(0, pool_.init(1));
  EXPECT_NE(0, matrix.generate(64, 64));
  EXPECT_EQ(0u, matrix.size());
}
TEST_F(PagedRotationMatrixTest, TemporaryHandoffOwnsFileUntilLastReader) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  IndexStorage::Pointer storage;
  ASSERT_EQ(0, CreateTemporaryIndexStorage(&storage));
  auto reader = storage;
  {
    StorageSegmentDumper dumper;
    ASSERT_EQ(0, dumper.prepare(storage, "payload", 32));
    const std::array<char, 32> payload{{'o', 'k'}};
    ASSERT_EQ(32u, dumper.write(payload.data(), payload.size()));
    ASSERT_EQ(0, dumper.append("payload", 2, 30, 0));
  }
  storage.reset();
  char out[2]{};
  ASSERT_EQ(2u, reader->get("payload")->fetch(0, out, 2));
  EXPECT_EQ('o', out[0]);
  EXPECT_EQ('k', out[1]);
  reader.reset();
  EXPECT_EQ(0u, pool_.used());
}
}  // namespace
}  // namespace zvec::core


namespace zvec::core {
TEST_F(PagedRotationMatrixTest, ApplyBudgetFailureAndInPlaceInverse) {
  ASSERT_EQ(0, pool_.init(8 * 1024 * 1024));
  PagedRotationMatrix matrix;
  ASSERT_EQ(0, matrix.generate(64, 64));
  std::vector<float> input(64, 0.25f), output(64);
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool_.capacity() - pool_.metadata_used() -
                 pool_.external_used());
    EXPECT_THROW(matrix.apply(input.data(), output.data()), std::bad_alloc);
  }
  matrix.apply(input.data(), output.data());
  matrix.apply(output.data(), output.data(), true);
  for (float value : output) EXPECT_NEAR(0.25f, value, 1e-5f);
  EXPECT_THROW(PagedRotationMatrix::check(IndexError_NoMemory), std::bad_alloc);
  EXPECT_THROW(PagedRotationMatrix::check(IndexError_ReadData),
               std::runtime_error);
}
}  // namespace zvec::core
