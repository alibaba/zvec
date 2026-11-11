// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>

namespace {

constexpr int32_t kRows = 2048;
constexpr int32_t kBatchSize = 1024;
constexpr int kDimension = 8;
constexpr int kQueries = 16;
constexpr size_t kTopk = 8;

struct Options {
  std::string mode;
  std::string path;
  std::string output;
};

Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string argument = argv[i];
    if (argument == "--mode") {
      options.mode = argv[i + 1];
    } else if (argument == "--path") {
      options.path = argv[i + 1];
    } else if (argument == "--output") {
      options.output = argv[i + 1];
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }
  if (argc != 7 || options.path.empty() || options.output.empty() ||
      (options.mode != "build" && options.mode != "query")) {
    throw std::runtime_error(
        "Usage: scalar_jit_smoke --mode build|query --path COLLECTION "
        "--output JSON");
  }
  return options;
}

void Check(const zvec::Status &status) {
  if (!status.ok()) {
    throw std::runtime_error(status.message());
  }
}

template <class T>
T Take(zvec::Result<T> result) {
  if (!result) {
    throw std::runtime_error(result.error().message());
  }
  return std::move(result).value();
}

std::vector<float> Vector(float position) {
  std::vector<float> vector(kDimension, 0.0F);
  vector[0] = position;
  return vector;
}

void Build(const std::string &path) {
  if (std::filesystem::exists(path)) {
    throw std::runtime_error("Refusing to overwrite collection: " + path);
  }
  zvec::CollectionSchema schema("sdk_scalar_jit_smoke");
  // The inverted index lets the planner choose the forward callback for
  // predicates that match more than the configured 50% of documents.
  schema.add_field(std::make_shared<zvec::FieldSchema>(
      "value", zvec::DataType::INT32, true,
      std::make_shared<zvec::InvertIndexParams>()));
  schema.add_field(std::make_shared<zvec::FieldSchema>(
      "label", zvec::DataType::STRING, false,
      std::make_shared<zvec::InvertIndexParams>()));
  schema.add_field(std::make_shared<zvec::FieldSchema>(
      "rating", zvec::DataType::FLOAT, true,
      std::make_shared<zvec::InvertIndexParams>()));
  schema.add_field(
      std::make_shared<zvec::FieldSchema>("flag", zvec::DataType::BOOL, false));
  schema.add_field(std::make_shared<zvec::FieldSchema>(
      "vector", zvec::DataType::VECTOR_FP32, kDimension, false,
      std::make_shared<zvec::HnswIndexParams>(zvec::MetricType::L2)));
  auto collection = Take(zvec::Collection::CreateAndOpen(
      path, schema, zvec::CollectionOptions{false, true}));
  for (int32_t begin = 0; begin < kRows; begin += kBatchSize) {
    std::vector<zvec::Doc> docs;
    for (int32_t id = begin; id < begin + kBatchSize; ++id) {
      zvec::Doc doc;
      doc.set_pk("doc_" + std::to_string(id));
      if (id % 13 == 0) {
        doc.set_null("value");
      } else {
        doc.set<int32_t>("value", id);
      }
      doc.set<std::string>("label", "label_" + std::to_string(id));
      if (id % 17 == 0)
        doc.set_null("rating");
      else
        doc.set<float>("rating", static_cast<float>(id) / 4.0F);
      doc.set<bool>("flag", id % 7 != 0);
      doc.set<std::vector<float>>("vector", Vector(static_cast<float>(id)));
      docs.emplace_back(std::move(doc));
    }
    const auto statuses = Take(collection->insert(docs));
    if (statuses.size() != docs.size()) {
      throw std::runtime_error("Unexpected insertion result count");
    }
    for (const auto &status : statuses) {
      Check(status);
    }
  }
  Check(collection->flush());
  Check(collection->optimize());
  Check(collection->close());
}

void Query(const std::string &path, std::ostream &output) {
  auto collection =
      Take(zvec::Collection::Open(path, zvec::CollectionOptions{true, true}));
  // Catch singleton/visibility regressions across the public shared SDK.
  if (zvec::GlobalConfig::Instance().invert_to_forward_scan_ratio() != 0.5F) {
    throw std::runtime_error("SDK reset the caller's forward-filter ratio");
  }
  output << std::setprecision(9) << "{\"queries\":[";
  for (int q = 0; q < kQueries; ++q) {
    const int32_t threshold = 32 + q * 16;
    zvec::SearchQuery query;
    query.topk_ = kTopk;
    query.filter_ = "value > " + std::to_string(threshold);
    switch (q % 6) {
      case 1:
        query.filter_ += " AND (rating >= 0.0 OR value > 0)";
        break;
      case 2:
        query.filter_ += " AND (value > 0 OR value < 1536)";
        break;
      case 3:
        query.filter_ += " AND label >= 'label_0'";
        break;
      case 4:
        query.filter_ += " AND label LIKE 'label_%'";
        break;
      case 5:
        query.filter_ += " AND value IN (";
        for (int32_t value = threshold + 1; value < kRows; ++value) {
          if (value != threshold + 1) query.filter_ += ',';
          query.filter_ += std::to_string(value);
        }
        query.filter_ += ')';
        break;
      default:
        break;
    }
    query.output_fields_ = {"value", "label"};
    query.target_.field_name_ = "vector";
    query.target_.query_params_ =
        std::make_shared<zvec::HnswQueryParams>(kRows);
    // Exactly representable coordinates give unique distances and identical
    // scores across native and emulated CPUs, without relying on tie order.
    const auto vector = Vector(static_cast<float>(threshold) + 0.25F);
    query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));
    const auto docs = Take(collection->query(query));
    if (docs.size() != kTopk) {
      throw std::runtime_error("Unexpected query result count");
    }
    if (q != 0) {
      output << ',';
    }
    output << "{\"threshold\":" << threshold << ",\"docs\":[";
    int32_t expected = threshold;
    for (size_t i = 0; i < docs.size(); ++i) {
      do {
        ++expected;
      } while (expected % 13 == 0);
      const auto &doc = docs[i];
      if (doc->pk() != "doc_" + std::to_string(expected) ||
          doc->is_null("value") || doc->get_ref<int32_t>("value") != expected ||
          doc->get_ref<std::string>("label") !=
              "label_" + std::to_string(expected) ||
          !std::isfinite(doc->score())) {
        throw std::runtime_error(
            "Query IDs, scores or scalar fields are wrong");
      }
      if (i != 0) {
        output << ',';
      }
      output << "{\"id\":\"" << doc->pk() << "\",\"score\":" << doc->score()
             << ",\"value\":" << doc->get_ref<int32_t>("value")
             << ",\"label\":\"" << doc->get_ref<std::string>("label") << "\"}";
    }
    output << "]}";
  }
  output << "]}\n";
  Check(collection->close());
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto options = ParseOptions(argc, argv);
    zvec::GlobalConfig::ConfigData config;
    config.query_thread_count = 1;
    config.optimize_thread_count = 1;
    config.query_thread_binding = false;
    config.optimize_thread_binding = false;
    config.invert_to_forward_scan_ratio = 0.5F;
    config.log_config = std::make_shared<zvec::GlobalConfig::ConsoleLogConfig>(
        zvec::GlobalConfig::LogLevel::kDebug);
    Check(zvec::GlobalConfig::Instance().initialize(config));
    std::ofstream output(options.output);
    if (!output) {
      throw std::runtime_error("Cannot open output: " + options.output);
    }
    if (options.mode == "build") {
      Build(options.path);
      output << "{\"rows\":" << kRows << "}\n";
    } else {
      Query(options.path, output);
    }
    output.flush();
    if (!output) {
      throw std::runtime_error("Cannot write output: " + options.output);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "scalar_jit_smoke: " << error.what() << '\n';
    return 1;
  }
}
