// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#if DISKANN_SUPPORTED
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/diskann/diskann_context.h"
#include "query_context_oom_fixture.h"

namespace zvec {
namespace {
TEST_P(QueryContextOomTest, DiskAnnReadersReleaseFailedContexts) {
  namespace ci = core_interface;
  const auto path = (directory_ / "diskann.index").string();
  auto param = ci::DiskAnnIndexParamBuilder()
                   .with_metric_type(ci::MetricType::kL2sq)
                   .with_data_type(ci::DataType::DT_FP32)
                   .with_dimension(1)
                   .with_max_degree(8)
                   .with_list_size(32)
                   .with_pq_chunk_num(1)
                   .build();
  {
    auto writer = ci::IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, writer);
    ASSERT_EQ(0, writer->open(
                     path, {ci::StorageOptions::StorageType::kMMAP, true}));
    for (uint32_t id = 0; id < 512; ++id) {
      const auto value = static_cast<float>(id);
      ASSERT_EQ(0, writer->add(ci::VectorData{ci::DenseVector{&value}}, id));
    }
    ASSERT_EQ(0, writer->train());
    ASSERT_EQ(0, writer->close());
  }

  core::IndexStorage::Pointer searcher_storage;
  core::IndexStorage::Pointer streamer_storage;
  ASSERT_NO_FATAL_FAILURE(OpenReadStorage(path, &searcher_storage));
  ASSERT_NO_FATAL_FAILURE(OpenReadStorage(path, &streamer_storage));
  auto searcher = core::IndexFactory::CreateSearcher("DiskAnnSearcher");
  auto streamer = core::IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(nullptr, searcher);
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, searcher->init(ailego::Params{}));
  ASSERT_EQ(0, searcher->load(searcher_storage, nullptr));
  core::IndexMeta meta(core::IndexMeta::DataType::DT_FP32, 1);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ASSERT_EQ(0, streamer->init(meta, ailego::Params{}));
  ASSERT_EQ(0, streamer->open(streamer_storage));

  const core::IndexQueryMeta query_meta(core::IndexMeta::DataType::DT_FP32, 1);
  const float query = 256.0F;
  auto check_reader = [&](auto &reader) {
    ASSERT_NO_FATAL_FAILURE(
        CheckContextFailure(reader, sizeof(core::DiskAnnContext),
                            sizeof(core::VisitByteMap::Context)));
    auto context = reader.create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(1);
    ASSERT_EQ(0, reader.search_bf_impl(&query, query_meta, context));
    ASSERT_EQ(1u, context->result().size());
    EXPECT_EQ(256u, context->result()[0].key());
  };
  ASSERT_NO_FATAL_FAILURE(check_reader(*searcher));
  ASSERT_NO_FATAL_FAILURE(check_reader(*streamer));
}
INSTANTIATE_TEST_SUITE_P(StorageBackends, QueryContextOomTest, ::testing::Bool());
}  // namespace
}  // namespace zvec
#endif
