// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/interface/index.h>

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

}  // namespace
}  // namespace zvec::core_interface
