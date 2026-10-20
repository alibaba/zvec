// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "quantizer/rotator/rotator.h"
#include "utility/storage_segment_dumper.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
TEST(RotationStorageTest, PagedMatrixAndFhtRoundTrip) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(16 * 1024 * 1024));
  for (auto type : {RotatorType::Matrix, RotatorType::FhtKac}) {
    std::shared_ptr<Rotator> original, loaded;
    ASSERT_EQ(0, Rotator::create(&original, 64, type));
    IndexStorage::Pointer storage;
    ASSERT_EQ(0, CreateTemporaryIndexStorage(&storage));
    ASSERT_EQ(0, original->dump(storage));
    ASSERT_EQ(0, Rotator::open(&loaded, storage));
    std::vector<float> input(64, 0.25f);
    auto output = original->rotate(input.data());
    EXPECT_EQ(output, loaded->rotate(input.data()));
    auto restored = loaded->unrotate(output.data());
    for (float value : restored) EXPECT_NEAR(0.25f, value, 1e-5f);
  }
}
}  // namespace zvec::core
