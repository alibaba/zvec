// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#endif

using Builder = zvec::core::QuantizedGraphRouterBuilder;
namespace {
Builder::Data data(size_t count, bool identical = false) {
  Builder::Data result;
  result.docs = 4096;
  for (size_t i = 0; i < count; ++i) {
    Builder::Vector v{};
    v[identical ? 0 : i % 16] = 1.f;
    result.vectors.push_back(v);
    result.ids.push_back(4095 - uint32_t(i) * 3);
    for (size_t d = 0; d < v.size(); ++d) result.mean[d] += v[d] / count;
  }
  return result;
}

std::string bytes(const Builder::Model &model) {
  std::ostringstream out(std::ios::binary);
  model.write(out);
  return out.str();
}

// Independent traversal checks all leaves exactly once, IDs rather than row
// offsets, and each internal center against its descendant anchors' mean.
std::vector<uint32_t> leaves(const Builder::Model &model, size_t group,
                             const Builder::Data &input) {
  std::vector<uint32_t> result;
  const auto &g = model.groups.at(group);
  EXPECT_GE(g.count, 1u);
  EXPECT_LE(g.count, 16u);
  for (size_t j = 0; j < g.count; ++j) {
    std::vector<uint32_t> ids;
    if (g.targets[j] < 0) {
      EXPECT_NE(g.targets[j], INT32_MIN);
      ids.push_back(uint32_t(-int64_t(g.targets[j]) - 1));
    } else {
      EXPECT_GT(size_t(g.targets[j]), group);
      ids = leaves(model, g.targets[j], input);
    }
    std::array<double, Builder::kDim> sum{};
    for (auto id : ids) {
      auto it = std::find(input.ids.begin(), input.ids.end(), id);
      EXPECT_NE(it, input.ids.end());
      if (it == input.ids.end()) continue;
      const auto &v = input.vectors[it - input.ids.begin()];
      for (size_t d = 0; d < v.size(); ++d) sum[d] += v[d];
    }
    double norm = 0;
    for (double v : sum) norm += v * v;
    norm = std::max(std::sqrt(norm), 1e-30);
    for (size_t d = 0; d < sum.size(); ++d)
      EXPECT_NEAR(g.centers[j][d], sum[d] / norm, 2e-6);
    result.insert(result.end(), ids.begin(), ids.end());
  }
  return result;
}
}  // namespace

TEST(QuantizedGraphRouterBuilder, SphericalTreeAndDeterminism) {
  const auto input = data(256);
  const auto model = Builder::train(input, {});
  auto actual = leaves(model, 0, input), expected = input.ids;
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(actual, expected);
  EXPECT_GT(model.groups.size(), 1u);
  EXPECT_LT(model.max_depth, 16u);
  EXPECT_EQ(bytes(model), bytes(Builder::train(input, {})));
  // Empty clusters may disappear, as in the Python implementation.
  EXPECT_GT(model.groups[0].count, 1u);
  for (size_t j = 0; j < model.groups[0].count; ++j) {
    const auto &center = model.groups[0].centers[j];
    double norm = 0;
    for (float v : center) norm += double(v) * v;
    EXPECT_NEAR(norm, 1., 2e-6);
  }
}

TEST(QuantizedGraphRouterBuilder, SmallAndDegenerateInputs) {
  for (size_t n : {size_t{1}, size_t{16}, size_t{257}}) {
    const auto input = data(n, true);
    const auto model = Builder::train(input, {});
    auto ids = leaves(model, 0, input);
    EXPECT_EQ(std::set<uint32_t>(ids.begin(), ids.end()),
              std::set<uint32_t>(input.ids.begin(), input.ids.end()));
    EXPECT_EQ(ids.size(), n);
    EXPECT_LT(model.max_depth, 16u);
    EXPECT_FALSE(bytes(model).empty());
    if (n > 16) {
      EXPECT_GT(model.balanced_splits, 0u);
    }
  }
}

TEST(QuantizedGraphRouterBuilder, RejectsInvalidTrainingDataAndWriteErrors) {
  EXPECT_THROW(Builder::train({}, {}), std::invalid_argument);
  auto input = data(32);
  input.ids[1] = input.ids[0];
  EXPECT_THROW(Builder::train(input, {}), std::invalid_argument);
  input = data(32);
  input.ids[0] = input.docs;
  EXPECT_THROW(Builder::train(input, {}), std::invalid_argument);
  input = data(32);
  input.vectors[0].fill(0);
  EXPECT_THROW(Builder::train(input, {}), std::invalid_argument);
  input = data(32);
  input.vectors[0][0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(Builder::train(input, {}), std::invalid_argument);
  input = data(32);
  input.mean[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(Builder::train(input, {}), std::invalid_argument);
  input = data(32);
  Builder::Options options;
  options.iterations = 0;
  EXPECT_THROW(Builder::train(input, options), std::invalid_argument);
  auto model = Builder::train(input, {});
  std::ostringstream out;
  out.setstate(std::ios::badbit);
  EXPECT_THROW(model.write(out), std::runtime_error);
}

#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
TEST(QuantizedGraphRouterBuilder, ProductionLoaderAndEntrySelection) {
  if (!__builtin_cpu_supports("avx512vnni") ||
      !__builtin_cpu_supports("avx512bw")) {
    GTEST_SKIP() << "VNNI router unavailable";
  }
  const auto input = data(16);
  const auto model = Builder::train(input, {});
  std::string pattern =
      (std::filesystem::temp_directory_path() / "zvec-router-XXXXXX").string();
  std::vector<char> filename(pattern.begin(), pattern.end());
  filename.push_back('\0');
  const int fd = mkstemp(filename.data());
  ASSERT_GE(fd, 0);
  close(fd);
  struct Cleanup {
    const char *path;
    ~Cleanup() {
      unlink(path);
    }
  } cleanup{filename.data()};
  {
    std::ofstream out(filename.data(), std::ios::binary);
    model.write(out);
  }
  struct Valid {
    const std::vector<uint32_t> &ids;
    bool valid(uint32_t id) const {
      return std::find(ids.begin(), ids.end(), id) != ids.end();
    }
  } valid{input.ids};
  std::vector<float> mean(input.mean.begin(), input.mean.end()), rotated_mean;
  zvec::turbo::SymphonyQGRotation rotation(768);
  rotation.rotate(mean.data(), rotated_mean);
  zvec::core::QuantizedGraphIntegerRouter router;
  ASSERT_TRUE(
      router.load(filename.data(), mean, valid, input.docs, &rotated_mean));
  for (size_t i = 0; i < input.ids.size(); ++i) {
    std::vector<float> rotated;
    rotation.rotate(input.vectors[i].data(), rotated);
    zvec::core::IntegerCenter::Query query;
    ASSERT_TRUE(router.prepare_projected(rotated.data(), query));
    EXPECT_EQ(router.choose(query, 1), input.ids[i]);
  }
}
#endif
