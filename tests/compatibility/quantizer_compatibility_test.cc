// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_flow.h>
#include <zvec/core/interface/index_factory.h>
namespace zvec::core {
class CompatibilityQuantizer : public turbo::Fp32Quantizer {
 public:
  static int initializations;
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    ++initializations;
    return Fp32Quantizer::init(meta, params);
  }
};
int CompatibilityQuantizer::initializations = 0;
INDEX_FACTORY_REGISTER_QUANTIZER(CompatibilityQuantizer);
INDEX_FACTORY_REGISTER_QUANTIZER_ALIAS(CompatibilityAlias,
                                       CompatibilityQuantizer);
}  // namespace zvec::core
namespace {
using namespace zvec::core;
using zvec::ailego::Factory;
TEST(QuantizerCompatibility, OldAPIsAndCustomAliases) {
  for (const auto &name : IndexFactory::AllQuantizers()) {
    EXPECT_TRUE(IndexFactory::HasQuantizer(name));
    EXPECT_NE(nullptr, IndexFactory::CreateQuantizer(name));
  }
  EXPECT_TRUE(IndexFactory::HasQuantizer("CompatibilityQuantizer"));
  EXPECT_TRUE(IndexFactory::HasQuantizer("CompatibilityAlias"));
  EXPECT_NE(nullptr,
            dynamic_cast<CompatibilityQuantizer *>(
                IndexFactory::CreateQuantizer("CompatibilityAlias").get()));
  EXPECT_FALSE(IndexFactory::HasQuantizer("UnknownCompatibilityQuantizer"));
  EXPECT_EQ(nullptr,
            IndexFactory::CreateQuantizer("UnknownCompatibilityQuantizer"));
}
struct EncodedHolder : MultiPassNumericalIndexHolder<uint8_t> {
  explicit EncodedHolder(const IndexMeta &m)
      : MultiPassNumericalIndexHolder<uint8_t>(m.element_size()), meta(m) {}
  size_t dimension() const override {
    return meta.dimension();
  }
  IndexMeta::DataType data_type() const override {
    return meta.data_type();
  }
  IndexMeta meta;
};
TEST(QuantizerCompatibility, CustomNameAndAliasPersistAndReopen) {
  for (bool legacy_reformer : {false, true}) {
    for (const char *name : {"CompatibilityQuantizer", "CompatibilityAlias"}) {
      SCOPED_TRACE(std::string(name) +
                   (legacy_reformer ? " reformer" : " quantizer"));
      auto path = (std::filesystem::path(::testing::TempDir()) /
                   (std::string(name) + ".index"))
                      .string();
      IndexMeta raw(IndexMeta::DT_FP32, 32);
      raw.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});
      auto q = IndexFactory::CreateQuantizer(name);
      ASSERT_NE(nullptr, q);
      ASSERT_EQ(0, q->init(raw, {}));
      auto meta = q->meta();
      if (legacy_reformer) {
        meta.set_reformer(name, 0, zvec::ailego::Params{});
      } else {
        meta.set_quantizer(name, 0, zvec::ailego::Params{});
      }
      auto holder = std::make_shared<EncodedHolder>(meta);
      std::vector<float> vector(32, 0.25f);
      zvec::ailego::NumericalVector<uint8_t> code(meta.element_size());
      q->quantize_data(vector.data(), &code[0]);
      ASSERT_TRUE(holder->emplace(17, code));
      auto builder = IndexFactory::CreateBuilder("FlatBuilder");
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      ASSERT_NE(nullptr, builder);
      ASSERT_NE(nullptr, dumper);
      ASSERT_EQ(0, builder->init(meta, {}, q));
      ASSERT_EQ(0, dumper->create(path));
      ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
      ASSERT_EQ(0, dumper->close());
      int before = CompatibilityQuantizer::initializations;
      IndexFlow flow;
      ASSERT_EQ(0, flow.load(path));
      EXPECT_GT(CompatibilityQuantizer::initializations, before);
      auto context = flow.create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(1);
      ASSERT_EQ(0, flow.search_bf_impl(vector.data(),
                                       IndexQueryMeta(IndexMeta::DT_FP32, 32),
                                       context));
      ASSERT_EQ(1u, context->result().size());
      EXPECT_EQ(17u, context->result()[0].key());
      ASSERT_EQ(0, flow.unload());
      std::filesystem::remove(path);
    }
  }
}
TEST(QuantizerCompatibility, SameNameOverrideUsedOnBuildAndReopen) {
  struct Restore {
    ~Restore() {
      Factory<zvec::turbo::Quantizer>::Register<zvec::turbo::Fp32Quantizer> r(
          "Fp32Quantizer");
    }
  } restore;
  Factory<zvec::turbo::Quantizer>::Register<CompatibilityQuantizer> override(
      "Fp32Quantizer");
  namespace ci = zvec::core_interface;
  ci::FlatIndexParam param;
  param.dimension = 32;
  param.data_type = IndexMeta::DT_FP32;
  param.metric_type = ci::MetricType::kL2sq;
  auto path =
      (std::filesystem::path(::testing::TempDir()) / "compat_override.index")
          .string();
  for (bool create : {true, false}) {
    auto before = CompatibilityQuantizer::initializations;
    auto index = ci::IndexFactory::CreateAndInitIndex(param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(
        0, index->open(path, {ci::StorageOptions::StorageType::kMMAP, create}));
    EXPECT_GT(CompatibilityQuantizer::initializations, before);
    if (create) {
      std::vector<float> vector(32, 0.5f);
      ci::VectorData data;
      data.vector = ci::DenseVector{vector.data()};
      ASSERT_EQ(0, index->add(data, 7));
    }
    ASSERT_EQ(0, index->close());
  }
  std::filesystem::remove_all(path);
}
}  // namespace
