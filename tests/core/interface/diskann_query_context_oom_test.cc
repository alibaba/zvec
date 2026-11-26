// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#if DISKANN_SUPPORTED
#include <array>
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
  ASSERT_NO_FATAL_FAILURE(open_read_storage(path, &searcher_storage));
  ASSERT_NO_FATAL_FAILURE(open_read_storage(path, &streamer_storage));
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
        check_context_failure(reader, sizeof(core::DiskAnnContext),
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
TEST_P(QueryContextOomTest, DiskAnnKeepsIdsWhenScoreNormalizationFails) {
  namespace ci = core_interface;
  using StorageType = ci::StorageOptions::StorageType;
  const auto path = (directory_ / "fp16.index").string();
  const auto param =
      ci::DiskAnnIndexParamBuilder()
          .with_metric_type(ci::MetricType::kL2sq)
          .with_data_type(ci::DataType::DT_FP32)
          .with_quantizer_param(ci::QuantizerParam(ci::QuantizerType::kFP16))
          .with_dimension(8)
          .with_max_degree(8)
          .with_list_size(32)
          .with_pq_chunk_num(1)
          .build();
  auto index = ci::IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageType::kMMAP, true}));
  for (uint32_t id = 0; id < 512; ++id) {
    std::array<float, 8> values;
    values.fill(static_cast<float>(id));
    ASSERT_EQ(0,
              index->add(ci::VectorData{ci::DenseVector{values.data()}}, id));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->close());
  ASSERT_EQ(0, index->open(path, {GetParam() ? StorageType::kBufferPool
                                             : StorageType::kMMAP,
                                  false, true}));

  constexpr size_t topk = 3;
  auto query = std::make_shared<ci::DiskAnnQueryParam>();
  query->topk = topk;
  query->list_size = 64;
  std::array<float, 8> values;
  values.fill(256.25F);
  const ci::VectorData input{ci::DenseVector{values.data()}};
  std::array<int64_t, topk> ids;
  std::array<float, topk> scores;
  ASSERT_EQ(
      0, index->search_internal_ids(input, query, ids.data(), scores.data()));
  const auto expected_ids = ids;
  const auto expected_scores = scores;
  ids.fill(-999);
  scores.fill(-999.0F);
  const auto previous_ids = ids;
  const auto previous_scores = scores;

  // The reformer's normalization reserves one IndexDocument per match.
  refused_allocation_size = topk * sizeof(core::IndexDocument);
  const int status =
      index->search_internal_ids(input, query, ids.data(), scores.data());
  const bool refused = refused_allocation_size == 0;
  refused_allocation_size = 0;
  ASSERT_TRUE(refused);
  EXPECT_EQ(core::IndexError_NoMemory, status);
  EXPECT_EQ(previous_ids, ids);
  EXPECT_EQ(previous_scores, scores);
  ASSERT_EQ(
      0, index->search_internal_ids(input, query, ids.data(), scores.data()));
  EXPECT_EQ(expected_ids, ids);
  EXPECT_EQ(expected_scores, scores);
  ASSERT_EQ(0, index->close());
}
INSTANTIATE_TEST_SUITE_P(StorageBackends, QueryContextOomTest, ::testing::Bool());
}  // namespace
}  // namespace zvec
#endif
