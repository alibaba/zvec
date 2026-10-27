// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
// Compile this unchanged with ea60a361 to write fixtures; current builds only
// read the checked-in files. Never regenerate fixtures with the code under
// test.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <turbo/distance/common/fast_scan_common.h>
#include <turbo/quantizer/common/pq_quantizer/packed_code_quantizer.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_flow.h>
using namespace zvec::core;
using namespace zvec::ailego;
namespace zvec::core {
INDEX_FACTORY_REGISTER_QUANTIZER_ALIAS(LegacyCustomFp32, turbo::Fp32Quantizer);
}
namespace {
void Check(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
constexpr size_t kDim = 32;
std::vector<float> Vector(size_t i) {
  std::vector<float> v(kDim);
  for (size_t j = 0; j < kDim; ++j)
    v[j] = std::sin(float(i * 37 + j * 13) * 0.1f);
  return v;
}
struct EncodedHolder : MultiPassNumericalIndexHolder<uint8_t> {
  explicit EncodedHolder(const IndexMeta &meta)
      : MultiPassNumericalIndexHolder<uint8_t>(meta.element_size()),
        meta_(meta) {}
  size_t dimension() const override {
    return meta_.dimension();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  IndexMeta meta_;
};
std::string Read(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  Check(bool(file), "Cannot read " + path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}
void Write(const std::string &path, const std::string &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), bytes.size());
  Check(bool(file), "Cannot write " + path);
}
float Distance(const zvec::turbo::Quantizer::Pointer &q) {
  auto v = Vector(0);
  std::string code(q->quantized_datapoint_vector_length(), '\0');
  std::string query(q->quantized_query_vector_length(), '\0');
  q->quantize_data(v.data(), code.data());
  q->quantize_query(v.data(), query.data());
  auto packed = dynamic_cast<const zvec::turbo::PackedCodeQuantizer *>(q.get());
  if (packed) {
    std::vector<uint8_t> block(zvec::turbo::fast_scan_packed_block_size(4));
    Check(packed->pack_codes(code.data(), 1, code.size(), block.data()) == 0,
          "pack code");
    float result = 0;
    packed->calc_distance_packed_block(block.data(), 1, query.data(), &result);
    return result;
  }
  return q->calc_distance_dp_query(code.data(), query.data());
}
void Run(const std::string &directory, bool write) {
  if (write) std::filesystem::create_directories(directory);
  const char *names[] = {"Fp32Quantizer",   "Fp16Quantizer",
                         "Int8Quantizer",   "Int4Quantizer",
                         "PqInt8Quantizer", "PqInt4Quantizer",
                         "PqFastQuantizer", "LegacyCustomFp32"};
  for (const char *name : names) {
    const std::string path = directory + "/" + name;
    IndexMeta raw(IndexMeta::DT_FP32, kDim);
    raw.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});
    Params params;
    params.set("num_chunk", uint32_t{4});
    auto q = IndexFactory::CreateQuantizer(name);
    Check(q && q->init(raw, params) == 0, std::string("init ") + name);
    if (write) {
      if (q->require_train()) {
        auto training =
            std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(kDim);
        for (size_t i = 0; i < 1024; ++i) {
          auto v = Vector(i);
          NumericalVector<float> row(kDim);
          std::copy(v.begin(), v.end(), &row[0]);
          Check(training->emplace(i, row), "training holder");
        }
        Check(q->train(training) == 0, std::string("train ") + name);
      }
      std::string blob;
      Check(q->serialize(&blob) == 0, "serialize");
      Write(path + ".quantizer", blob);
      // Store a numerical reference, not SIMD-specific encoded bytes.
      std::ofstream expected(path + ".distance");
      expected.precision(9);
      expected << Distance(q) << '\n';
    } else {
      auto blob = Read(path + ".quantizer");
      Check(q->deserialize(blob) == 0, "deserialize " + path);
      std::ifstream expected(path + ".distance");
      float distance = 0;
      expected >> distance;
      Check(bool(expected), "missing distance");
      Check(std::abs(Distance(q) - distance) < 0.01f,
            "distance mismatch " + path);
    }
    // Flat physical layouts cover the four scalar formats plus a legacy alias.
    if (std::string(name).find("Pq") == 0) continue;
    if (write) {
      IndexMeta meta = q->meta();
      meta.set_quantizer(name, 0, params);
      meta.set_major_order(IndexMeta::MO_ROW);
      auto holder = std::make_shared<EncodedHolder>(meta);
      for (size_t i = 0; i < 17; ++i) {
        NumericalVector<uint8_t> code(q->quantized_datapoint_vector_length());
        auto v = Vector(i);
        q->quantize_data(v.data(), &code[0]);
        Check(holder->emplace(i, code), "encoded holder");
      }
      auto builder = IndexFactory::CreateBuilder("FlatBuilder");
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      Check(builder && dumper, "builder/dumper");
      Check(builder->init(meta, {}, q) == 0, "builder init");
      Check(dumper->create(path + ".index") == 0, "dumper create");
      Check(IndexBuilder::TrainBuildAndDump(builder, holder, dumper) == 0,
            "dump");
      Check(dumper->close() == 0, "close");
    }
    IndexFlow flow;
    Check(flow.load(path + ".index") == 0, "flow load " + path);
    auto context = flow.create_context();
    Check(bool(context), "context");
    context->set_topk(3);
    auto query = Vector(0);
    Check(flow.search_bf_impl(query.data(),
                              IndexQueryMeta(IndexMeta::DT_FP32, kDim),
                              context) == 0,
          "search");
    Check(context->result().size() == 3 && context->result()[0].key() == 0,
          "top result");
    Check(flow.unload() == 0, "unload");
  }
}
}  // namespace
int main(int argc, char **argv) {
  try {
    Check(argc == 3, "Usage: index_format_fixture --write|--read directory");
    Check(std::string(argv[1]) == "--write" || std::string(argv[1]) == "--read",
          "invalid mode");
    Run(argv[2], std::string(argv[1]) == "--write");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
