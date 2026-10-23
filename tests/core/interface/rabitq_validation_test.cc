// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

template <typename IndexT>
class TestIndex : public IndexT {
 public:
  using IndexT::init;
};

template <typename IndexT, typename ParamT>
void RejectRotation() {
  for (auto metric :
       {MetricType::kL2sq, MetricType::kInnerProduct, MetricType::kCosine}) {
    for (bool typed : {false, true}) {
      ParamT params;
      params.dimension = 128;
      params.data_type = DataType::DT_FP32;
      params.metric_type = metric;
      params.quantizer_param =
          typed ? std::make_shared<RabitqQuantizerParam>()
                : std::make_shared<QuantizerParam>(QuantizerType::kRabitq);
      params.quantizer_param->enable_rotate = true;
      TestIndex<IndexT> index;
      EXPECT_EQ(core::IndexError_Unsupported, index.init(params));
#if RABITQ_SUPPORTED
      params.quantizer_param->enable_rotate = false;
      EXPECT_EQ(0, index.init(params));
#endif
    }
  }
}

TEST(RabitqValidation, CoreRejectsAdditionalRotation) {
  RejectRotation<HNSWIndex, HNSWIndexParam>();
  RejectRotation<IVFIndex, IVFIndexParam>();
#if DISKANN_SUPPORTED
  RejectRotation<DiskAnnIndex, DiskAnnIndexParam>();
#endif
}

template <typename IndexT, typename ParamT>
void RejectUnsupportedIndex() {
  for (auto metric :
       {MetricType::kL2sq, MetricType::kInnerProduct, MetricType::kCosine}) {
    ParamT params;
    params.dimension = 128;
    params.data_type = DataType::DT_FP32;
    params.metric_type = metric;
    params.quantizer_param = std::make_shared<RabitqQuantizerParam>();
    TestIndex<IndexT> index;
    EXPECT_EQ(core::IndexError_Unsupported, index.init(params));
    params.quantizer_param = std::make_shared<QuantizerParam>();
    EXPECT_EQ(0, index.init(params));
  }
}

TEST(RabitqValidation, CoreRejectsUnsupportedIndexWithoutBreakingRawVectors) {
  RejectUnsupportedIndex<FlatIndex, FlatIndexParam>();
  RejectUnsupportedIndex<VamanaIndex, VamanaIndexParam>();
}

// Older untrained RaBitQ blocks used the legacy FP32/Cosine converter path.
class LegacyRawFlatIndex : public TestIndex<FlatIndex> {
 protected:
  int create_and_init_converter_reformer(
      const QuantizerParam &quantizer, const BaseIndexParam &params) override {
    return Index::create_and_init_converter_reformer(quantizer, params);
  }
};

TEST(RabitqValidation, RawFlatFallbackReopensLegacyUntrainedBlocks) {
  const std::string path = "rabitq_legacy_raw_flat.index";
  for (auto metric :
       {MetricType::kL2sq, MetricType::kInnerProduct, MetricType::kCosine}) {
    zvec::test_util::RemoveTestFiles(path);
    FlatIndexParam params;
    params.dimension = 128;
    params.data_type = DataType::DT_FP32;
    params.metric_type = metric;
    params.quantizer_param = std::make_shared<QuantizerParam>();
    std::vector<float> raw(128, 0);
    raw[0] = 1;
    VectorData vector;
    vector.vector = DenseVector{raw.data()};
    {
      LegacyRawFlatIndex legacy;
      ASSERT_EQ(0, legacy.init(params));
      ASSERT_EQ(0,
                legacy.open(path, {StorageOptions::StorageType::kMMAP, true}));
      ASSERT_EQ(0, legacy.add(vector, 17));
      ASSERT_EQ(0, legacy.flush());
      ASSERT_EQ(0, legacy.close());
    }
    auto index = IndexFactory::CreateAndInitIndex(params);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0,
              index->open(path, {StorageOptions::StorageType::kMMAP, false}));
    auto query = std::make_shared<FlatQueryParam>();
    query->topk = 1;
    SearchResult result;
    ASSERT_EQ(0, index->search(vector, query, &result));
    ASSERT_EQ(1u, result.doc_list_.size());
    EXPECT_EQ(17u, result.doc_list_[0].key());
    // Reopened legacy blocks must remain writable as well as searchable.
    ASSERT_EQ(0, index->add(vector, 18));
    ASSERT_EQ(0, index->close());
    zvec::test_util::RemoveTestFiles(path);
  }
}

}  // namespace
}  // namespace zvec::core_interface
