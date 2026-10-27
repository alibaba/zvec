// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <zvec/core/interface/index.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"

using Builder = zvec::core::QuantizedGraphRouterBuilder;
namespace ci = zvec::core_interface;

namespace {
void check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
uint64_t number(const char *text) {
  const std::string s(text);
  check(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos,
        "expected an unsigned decimal integer");
  size_t end;
  uint64_t value = std::stoull(s, &end);
  check(end == s.size(), "invalid integer");
  return value;
}
void read(std::istream &in, void *p, size_t bytes) {
  in.read(static_cast<char *>(p), bytes);
  check(bool(in), "truncated or unreadable training input");
}
void eof(std::istream &in) {
  check(in.peek() == std::char_traits<char>::eof() && !in.bad(),
        "trailing bytes in training input");
}

// Compatibility input for migrating an existing Python training job. Records
// contain uint32 internal ID, uint64 external key, and 768 floats, without
// struct padding. External keys are not used as graph leaf IDs.
Builder::Data read_anchors(const std::string &path, const std::string &mean,
                           bool cosine) {
  std::ifstream in(path, std::ios::binary), mu(mean, std::ios::binary);
  check(bool(in) && bool(mu), "cannot open anchors or mean");
  uint32_t header[5];
  read(in, header, sizeof(header));
  check(header[0] == 0x51474159 && header[1] == 1 && header[2] > 0 &&
            header[2] <= Builder::kMaxAnchors && header[3] > 0 &&
            header[3] <= 4096 && header[4] > 0 &&
            header[4] <= uint32_t(INT32_MAX),
        "expected version-1 FP32 anchor export");
  check(
      std::filesystem::file_size(path) ==
          20 + uint64_t(header[2]) * (12 + uint64_t(header[3]) * sizeof(float)),
      "anchor file length does not match header");
  Builder::Data data;
  data.docs = header[4];
  data.cosine = cosine;
  data.mean.resize(header[3]);
  read(mu, data.mean.data(), data.mean.size() * sizeof(float));
  eof(mu);
  data.ids.resize(header[2]);
  data.vectors.assign(header[2], Builder::Vector(header[3]));
  for (size_t i = 0; i < data.ids.size(); ++i) {
    uint64_t key;
    read(in, &data.ids[i], sizeof(uint32_t));
    read(in, &key, sizeof(key));
    read(in, data.vectors[i].data(), data.mean.size() * sizeof(float));
  }
  eof(in);
  return data;
}

class Snapshot : public ci::HNSWIndex {
 public:
  using ci::HNSWIndex::init;
  void prepare_reader() {
    context_ = streamer_->create_context();
    hnsw_context_ = dynamic_cast<zvec::core::HnswContext *>(context_.get());
    check(hnsw_context_ != nullptr, "cannot create HNSW snapshot context");
  }
  int vector(uint32_t id, zvec::core::IndexStorage::MemoryBlock &block) const {
    if (streamer_->meta().data_type() != zvec::core::IndexMeta::DT_FP32 ||
        hnsw_context_->get_entity().get_key(id) == zvec::core::kInvalidKey)
      return zvec::core::IndexError_Mismatch;
    return streamer_->get_vector_by_id(id, block);
  }

 private:
  zvec::core::IndexStreamer::Context::Pointer context_;
  const zvec::core::HnswContext *hnsw_context_ = nullptr;
};

Builder::Data read_snapshot(const std::string &path, size_t count,
                            size_t dimension, bool cosine) {
  check(count > 0 && count <= Builder::kMaxAnchors,
        "anchor count must be in [1, 65536]");
  Snapshot index;
  ci::HNSWIndexParam param(
      cosine ? ci::MetricType::kCosine : ci::MetricType::kL2sq, dimension, 50,
      500);
  param.data_type = ci::DataType::DT_FP32;
  param.use_id_map = true;
  check(index.init(param) == 0, "cannot initialize snapshot reader");
  ci::StorageOptions storage;
  storage.type = ci::StorageOptions::StorageType::kMMAP;
  storage.read_only = true;
  check(index.open(path, storage) == 0,
        "snapshot does not match the requested inline FP32 metric/dimension");
  index.prepare_reader();
  Builder::Data data;
  const uint64_t docs = index.get_doc_count();
  check(docs > 0 && docs <= uint64_t(INT32_MAX), "unsupported document count");
  data.docs = docs;
  data.cosine = cosine;
  data.mean.resize(dimension);
  count = std::min(count, size_t(docs));
  data.ids.resize(count);
  data.vectors.assign(count, Builder::Vector(dimension));
  // Preserve the historical Cohere sampling sequence. Make the stride
  // coprime for other corpus sizes so anchors are always unique.
  uint64_t stride = 2654435761ULL;
  while (std::gcd(stride, docs) != 1) ++stride;
  std::vector<int32_t> selected(docs, -1);
  for (size_t i = 0; i < count; ++i) {
    const uint32_t id = (uint64_t(i) * stride + 7193) % docs;
    data.ids[i] = id;
    selected[id] = i;
  }
  std::vector<double> sum(dimension);
  for (uint32_t id = 0; id < docs; ++id) {
    zvec::core::IndexStorage::MemoryBlock block;
    check(index.vector(id, block) == 0 && block.data() != nullptr,
          "snapshot must contain readable vectors at every internal ID");
    const auto *v = static_cast<const float *>(block.data());
    double norm = 0;
    for (size_t d = 0; d < dimension; ++d) {
      check(std::isfinite(v[d]), "nonfinite snapshot vector");
      norm += double(v[d]) * v[d];
      sum[d] += v[d];
    }
    check(!cosine || std::abs(std::sqrt(norm) - 1.) <= 1e-4,
          "snapshot vectors must be normalized and nonzero");
    if (selected[id] >= 0)
      std::copy_n(v, dimension, data.vectors[selected[id]].begin());
  }
  for (size_t d = 0; d < dimension; ++d) data.mean[d] = float(sum[d] / docs);
  check(index.close() == 0, "closing snapshot failed");
  return data;
}

void publish(const Builder::Model &model, const std::string &path) {
  // A temporary sibling plus link() publishes only a complete model and
  // refuses to replace an existing file, including in concurrent invocations.
  std::string pattern = path + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const int fd = mkstemp(name.data());
  check(fd >= 0, "cannot create model temporary file");
  close(fd);
  struct Cleanup {
    const char *path;
    ~Cleanup() {
      unlink(path);
    }
  } cleanup{name.data()};
  std::ofstream out(name.data(), std::ios::binary | std::ios::trunc);
  check(bool(out), "cannot open model temporary file");
  model.write(out);
  out.close();
  check(bool(out), "flushing model failed");
  if (link(name.data(), path.c_str()) != 0)
    throw std::runtime_error("cannot publish model: " +
                             std::string(std::strerror(errno)));
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const uint32_t endian = 1;
    check(*reinterpret_cast<const uint8_t *>(&endian) == 1,
          "router tool requires little-endian host");
    size_t dimension = 768, projected = 0;
    bool cosine = true;
    std::vector<char *> positional{argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--dimension" || flag == "--projected" ||
          flag == "--metric") {
        check(i + 1 < argc, "missing layout option value");
        const char *value = argv[++i];
        if (flag == "--dimension")
          dimension = number(value);
        else if (flag == "--projected")
          projected = number(value);
        else {
          check(std::string(value) == "cosine" || std::string(value) == "l2",
                "metric must be cosine or l2");
          cosine = std::string(value) == "cosine";
        }
      } else
        positional.push_back(argv[i]);
    }
    check(dimension > 0 && dimension <= 4096 && projected <= 4096,
          "invalid dimension/projection");
    argc = positional.size();
    argv = positional.data();
    check(argc >= 4,
          "usage: quantized_graph_router build INDEX OUTPUT [ANCHORS [SEED]]\n"
          "   or: quantized_graph_router train ANCHORS.bin MEAN.f32 OUTPUT "
          "[SEED] [--dimension D --metric cosine|l2 --projected P]");
    const std::string mode = argv[1];
    Builder::Options options;
    options.projected = projected;
    Builder::Data data;
    std::string output;
    const auto start = std::chrono::steady_clock::now();
    if (mode == "build" && argc >= 4 && argc <= 6) {
      output = argv[3];
      check(!std::filesystem::exists(output), "output already exists");
      const auto anchors = argc >= 5 ? number(argv[4]) : Builder::kMaxAnchors;
      check(anchors > 0 && anchors <= Builder::kMaxAnchors, "invalid anchors");
      if (argc == 6) options.seed = number(argv[5]);
      data = read_snapshot(argv[2], anchors, dimension, cosine);
    } else if (mode == "train" && (argc == 5 || argc == 6)) {
      output = argv[4];
      check(!std::filesystem::exists(output), "output already exists");
      if (argc == 6) options.seed = number(argv[5]);
      data = read_anchors(argv[2], argv[3], cosine);
    } else {
      throw std::invalid_argument("invalid command line; use build or train");
    }
    const auto train_start = std::chrono::steady_clock::now();
    std::cout << "TRAIN_START docs=" << data.docs
              << " anchors=" << data.ids.size() << " seed=" << options.seed
              << " iterations=" << options.iterations << std::endl;
    const auto model = Builder::train(data, options);
    const auto train_end = std::chrono::steady_clock::now();
    publish(model, output);
    std::cout << "TRAIN_COMPLETE groups=" << model.groups.size()
              << " max_depth=" << model.max_depth
              << " balanced_splits=" << model.balanced_splits
              << " training_seconds="
              << std::chrono::duration<double>(train_end - train_start).count()
              << " total_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " model_bytes=" << std::filesystem::file_size(output)
              << " output=" << output << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return 1;
  }
}
