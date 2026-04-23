// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <utility>
#include <vector>
#include <ailego/pattern/defer.h>
#include <omega/ground_truth.h>
#include <omega/omega_trainer.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/omega/omega_params.h"
#include "algorithm/omega/omega_streamer.h"
#include "omega_training_session.h"

namespace zvec::core_interface {

namespace {

constexpr size_t kOmegaTrainingTopk = 100;
constexpr uint32_t kOmegaQueryCacheMagic = 0x4F514359;  // OQCY
constexpr uint32_t kOmegaQueryCacheVersion = 1;

int DefaultOmegaTrainerThreads() {
  const unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0) {
    return 8;
  }
  return static_cast<int>(std::max(1u, hc / 2));
}

struct OmegaTrainingArtifacts {
  std::vector<TrainingRecord> records;
  GtCmpsData gt_cmps_data;
  std::vector<std::vector<float>> training_queries;
  std::vector<uint64_t> query_doc_ids;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(std::vector<std::pair<std::string, int64_t>> *stats,
                       std::string name)
      : stats_(stats),
        name_(std::move(name)),
        start_(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer() {
    if (stats_ == nullptr) {
      return;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    stats_->emplace_back(
        name_,
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count());
  }

 private:
  std::vector<std::pair<std::string, int64_t>> *stats_;
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};

void WriteTimingStatsJson(
    const std::string &output_path,
    const std::vector<std::pair<std::string, int64_t>> &stats) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    LOG_WARN("Failed to write OMEGA timing stats: %s", output_path.c_str());
    return;
  }
  ofs << "{\n";
  for (size_t i = 0; i < stats.size(); ++i) {
    ofs << "  \"" << stats[i].first << "\": " << stats[i].second;
    if (i + 1 < stats.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "}\n";
}

std::filesystem::path GetOmegaModelDir(const std::string &index_path) {
  const std::filesystem::path path(index_path);
  const auto parent = path.parent_path();
  return parent.empty() ? std::filesystem::path("omega_model")
                        : parent / "omega_model";
}

std::filesystem::path GetOmegaQueryCachePath(const std::string &index_path) {
  return GetOmegaModelDir(index_path) / "training_queries.bin";
}

bool EnsureOmegaModelDir(const std::string &index_path) {
  const auto model_dir = GetOmegaModelDir(index_path);
  std::error_code ec;
  std::filesystem::create_directories(model_dir, ec);
  if (ec) {
    LOG_ERROR("Failed to create omega model dir %s: %s",
              model_dir.string().c_str(), ec.message().c_str());
    return false;
  }
  return true;
}

bool SaveOmegaTrainingQueryCache(const std::string &index_path,
                                 const std::vector<std::vector<float>> &queries,
                                 const std::vector<uint64_t> &query_doc_ids) {
  if (queries.empty() || queries.size() != query_doc_ids.size()) {
    return false;
  }

  if (!EnsureOmegaModelDir(index_path)) {
    return false;
  }

  const uint32_t dim = static_cast<uint32_t>(queries[0].size());
  for (const auto &query : queries) {
    if (query.size() != dim) {
      return false;
    }
  }

  std::ofstream ofs(GetOmegaQueryCachePath(index_path), std::ios::binary);
  if (!ofs.is_open()) {
    return false;
  }

  const uint64_t num_queries = queries.size();
  ofs.write(reinterpret_cast<const char *>(&kOmegaQueryCacheMagic),
            sizeof(kOmegaQueryCacheMagic));
  ofs.write(reinterpret_cast<const char *>(&kOmegaQueryCacheVersion),
            sizeof(kOmegaQueryCacheVersion));
  ofs.write(reinterpret_cast<const char *>(&num_queries), sizeof(num_queries));
  ofs.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  for (size_t i = 0; i < queries.size(); ++i) {
    ofs.write(reinterpret_cast<const char *>(&query_doc_ids[i]),
              sizeof(query_doc_ids[i]));
    ofs.write(reinterpret_cast<const char *>(queries[i].data()),
              static_cast<std::streamsize>(queries[i].size() * sizeof(float)));
  }
  return ofs.good();
}

bool LoadOmegaTrainingQueryCache(const std::string &index_path,
                                 std::vector<std::vector<float>> *queries,
                                 std::vector<uint64_t> *query_doc_ids) {
  std::ifstream ifs(GetOmegaQueryCachePath(index_path), std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t num_queries = 0;
  uint32_t dim = 0;
  ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
  ifs.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries));
  ifs.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!ifs.good() || magic != kOmegaQueryCacheMagic ||
      version != kOmegaQueryCacheVersion || num_queries == 0 || dim == 0) {
    return false;
  }

  queries->assign(num_queries, std::vector<float>(dim));
  query_doc_ids->assign(num_queries, 0);
  for (size_t i = 0; i < num_queries; ++i) {
    ifs.read(reinterpret_cast<char *>(&(*query_doc_ids)[i]), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char *>((*queries)[i].data()),
             static_cast<std::streamsize>(dim * sizeof(float)));
    if (!ifs.good()) {
      queries->clear();
      query_doc_ids->clear();
      return false;
    }
  }
  return true;
}

bool DecodeDenseVectorBuffer(const OmegaIndexParam &param,
                             const VectorDataBuffer &vector_data_buffer,
                             std::vector<float> *values) {
  if (!std::holds_alternative<DenseVectorBuffer>(
          vector_data_buffer.vector_buffer)) {
    LOG_ERROR("Expected dense vector buffer for OMEGA training");
    return false;
  }

  const auto &dense_buffer =
      std::get<DenseVectorBuffer>(vector_data_buffer.vector_buffer).data;
  values->assign(static_cast<size_t>(param.dimension), 0.0f);

  switch (param.data_type) {
    case DataType::DT_FP32: {
      if (dense_buffer.size() !=
          static_cast<size_t>(param.dimension) * sizeof(float)) {
        LOG_ERROR("Unexpected FP32 vector buffer size: %zu",
                  dense_buffer.size());
        return false;
      }
      std::memcpy(values->data(), dense_buffer.data(), dense_buffer.size());
      return true;
    }
    case DataType::DT_FP16: {
      if (dense_buffer.size() != static_cast<size_t>(param.dimension) *
                                     sizeof(zvec::ailego::Float16)) {
        LOG_ERROR("Unexpected FP16 vector buffer size: %zu",
                  dense_buffer.size());
        return false;
      }
      const auto *src =
          reinterpret_cast<const zvec::ailego::Float16 *>(dense_buffer.data());
      for (int i = 0; i < param.dimension; ++i) {
        (*values)[static_cast<size_t>(i)] = static_cast<float>(src[i]);
      }
      return true;
    }
    default:
      LOG_ERROR("OMEGA training only supports FP32/FP16 input vectors");
      return false;
  }
}

bool FetchDenseVector(const OmegaIndexParam &param, Index *index,
                      uint32_t doc_id, std::vector<float> *values) {
  VectorDataBuffer vector_data_buffer;
  if (index->Fetch(doc_id, &vector_data_buffer) != 0) {
    LOG_WARN("Failed to fetch vector %u for OMEGA training", doc_id);
    return false;
  }
  return DecodeDenseVectorBuffer(param, vector_data_buffer, values);
}

std::vector<std::vector<float>> LoadBaseVectors(const OmegaIndexParam &param,
                                                Index *index,
                                                uint32_t doc_count,
                                                size_t num_threads, bool *ok) {
  std::vector<std::vector<float>> base_vectors(doc_count);
  std::atomic<bool> load_error{false};
  size_t actual_threads =
      num_threads == 0 ? std::thread::hardware_concurrency() : num_threads;
  actual_threads = std::max<size_t>(1, actual_threads);
  actual_threads = std::min<size_t>(actual_threads, doc_count);
  const size_t docs_per_thread =
      (doc_count + actual_threads - 1) / actual_threads;

  std::vector<std::thread> threads;
  for (size_t t = 0; t < actual_threads; ++t) {
    const size_t start = t * docs_per_thread;
    const size_t end = std::min<size_t>(start + docs_per_thread, doc_count);
    if (start >= end) {
      continue;
    }
    threads.emplace_back([&, start, end]() {
      for (size_t doc_idx = start; doc_idx < end && !load_error.load();
           ++doc_idx) {
        std::vector<float> values;
        if (!FetchDenseVector(param, index, static_cast<uint32_t>(doc_idx),
                              &values)) {
          load_error = true;
          break;
        }
        base_vectors[doc_idx] = std::move(values);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  *ok = !load_error.load();
  return base_vectors;
}

std::vector<std::vector<uint64_t>> ComputeGroundTruthBruteForce(
    const OmegaIndexParam &param, Index *index,
    const std::vector<std::vector<float>> &queries, size_t topk,
    const std::vector<uint64_t> &query_doc_ids, size_t num_threads) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());
  if (queries.empty()) {
    return ground_truth;
  }

  bool loaded_ok = false;
  const uint32_t doc_count = index->GetDocCount();
  const auto base_vectors =
      LoadBaseVectors(param, index, doc_count, num_threads, &loaded_ok);
  if (!loaded_ok) {
    LOG_ERROR("Failed to load base vectors for OMEGA ground truth");
    return {};
  }

  std::vector<float> flat_base;
  flat_base.reserve(static_cast<size_t>(doc_count) * base_vectors[0].size());
  for (const auto &base : base_vectors) {
    flat_base.insert(flat_base.end(), base.begin(), base.end());
  }

  std::vector<float> flat_queries;
  flat_queries.reserve(queries.size() * queries[0].size());
  for (const auto &query : queries) {
    flat_queries.insert(flat_queries.end(), query.begin(), query.end());
  }

  omega::MetricType metric_type = omega::MetricType::IP;
  switch (param.metric_type) {
    case MetricType::kL2sq:
      metric_type = omega::MetricType::L2;
      break;
    case MetricType::kCosine:
      metric_type = omega::MetricType::COSINE;
      break;
    case MetricType::kInnerProduct:
    default:
      metric_type = omega::MetricType::IP;
      break;
  }

  const bool held_out_mode =
      !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  return omega::ComputeGroundTruth(flat_base.data(), flat_queries.data(),
                                   doc_count, queries.size(),
                                   static_cast<size_t>(param.dimension), topk,
                                   metric_type, held_out_mode, query_doc_ids);
}

std::vector<std::vector<uint64_t>> ComputeGroundTruthByHnsw(
    const OmegaIndexParam &param, OmegaIndex *index,
    const std::vector<std::vector<float>> &queries, size_t topk,
    const std::vector<uint64_t> &query_doc_ids, size_t num_threads) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());
  if (queries.empty()) {
    return ground_truth;
  }

  auto *omega_streamer =
      dynamic_cast<core::OmegaStreamer *>(index->index_searcher().get());
  if (omega_streamer == nullptr) {
    LOG_ERROR("OMEGA streamer is unavailable during ground truth collection");
    return {};
  }

  omega_streamer->EnableTrainingMode(true);
  AILEGO_DEFER([&]() { omega_streamer->EnableTrainingMode(false); });

  const bool held_out_mode =
      !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  const size_t actual_topk = held_out_mode ? topk + 1 : topk;
  size_t actual_threads =
      num_threads == 0 ? std::thread::hardware_concurrency() : num_threads;
  actual_threads = std::max<size_t>(1, actual_threads);
  actual_threads = std::min(actual_threads, queries.size());
  const size_t queries_per_thread =
      (queries.size() + actual_threads - 1) / actual_threads;

  std::vector<std::thread> threads;
  for (size_t t = 0; t < actual_threads; ++t) {
    const size_t start = t * queries_per_thread;
    const size_t end = std::min(start + queries_per_thread, queries.size());
    if (start >= end) {
      continue;
    }
    threads.emplace_back([&, start, end]() {
      for (size_t q = start; q < end; ++q) {
        SearchResult result;
        VectorData query{DenseVector{queries[q].data()}};
        auto query_param = OmegaQueryParamBuilder()
                               .with_topk(static_cast<uint32_t>(actual_topk))
                               .with_fetch_vector(false)
                               .with_ef_search(param.ef_groundtruth)
                               .with_training_query_id(-1)
                               .with_target_recall(1.0f)
                               .build();
        if (index->Search(query, query_param, &result) != 0) {
          continue;
        }

        auto &doc_ids = ground_truth[q];
        doc_ids.reserve(topk);
        for (const auto &doc : result.doc_list_) {
          if (held_out_mode && doc.key() == query_doc_ids[q]) {
            continue;
          }
          doc_ids.push_back(doc.key());
          if (doc_ids.size() >= topk) {
            break;
          }
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return ground_truth;
}

std::vector<std::vector<uint64_t>> ComputeGroundTruth(
    const OmegaIndexParam &param, OmegaIndex *index,
    const std::vector<std::vector<float>> &queries, size_t topk,
    const std::vector<uint64_t> &query_doc_ids, size_t num_threads) {
  if (param.ef_groundtruth > 0) {
    return ComputeGroundTruthByHnsw(param, index, queries, topk, query_doc_ids,
                                    num_threads);
  }
  return ComputeGroundTruthBruteForce(param, index, queries, topk,
                                      query_doc_ids, num_threads);
}

bool CollectTrainingData(
    OmegaIndex *index, const OmegaIndexParam &param,
    const std::vector<std::vector<float>> &training_queries,
    const std::vector<uint64_t> &query_doc_ids, OmegaTrainingArtifacts *result,
    std::vector<std::pair<std::string, int64_t>> *timing_stats) {
  std::vector<std::vector<uint64_t>> ground_truth;
  {
    ScopedTimer timer(timing_stats, "Step2: ComputeGroundTruth");
    ground_truth =
        ComputeGroundTruth(param, index, training_queries, kOmegaTrainingTopk,
                           query_doc_ids, DefaultOmegaTrainerThreads());
  }

  if (ground_truth.empty()) {
    LOG_ERROR("Failed to compute OMEGA ground truth");
    return false;
  }

  auto *training_capable = index->GetTrainingCapability();
  if (training_capable == nullptr) {
    LOG_ERROR("OMEGA index does not expose training capability");
    return false;
  }

  auto session = training_capable->CreateTrainingSession();
  if (session == nullptr) {
    LOG_ERROR("Failed to create OMEGA training session");
    return false;
  }

  TrainingSessionConfig config;
  config.ground_truth = ground_truth;
  config.topk = kOmegaTrainingTopk;
  config.k_train = param.k_train;

  {
    ScopedTimer timer(timing_stats, "Step3: EnableTrainingMode");
    auto status = session->Start(config);
    if (!status.ok()) {
      LOG_ERROR("Failed to start OMEGA training session: %s",
                status.message().c_str());
      return false;
    }
  }

  AILEGO_DEFER([&]() { session->Finish(); });

  {
    ScopedTimer timer(timing_stats, "Step4: TrainingSearches");
    size_t actual_threads = std::thread::hardware_concurrency();
    actual_threads = std::max<size_t>(1, actual_threads);
    actual_threads = std::min(actual_threads, training_queries.size());
    const size_t queries_per_thread =
        (training_queries.size() + actual_threads - 1) / actual_threads;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < actual_threads; ++t) {
      const size_t start = t * queries_per_thread;
      const size_t end =
          std::min(start + queries_per_thread, training_queries.size());
      if (start >= end) {
        continue;
      }
      threads.emplace_back([&, start, end, actual_threads]() {
        for (size_t q = start; q < end; ++q) {
          if (actual_threads == 1) {
            session->BeginQuery(static_cast<int>(q));
          }

          SearchResult search_result;
          VectorData query{DenseVector{training_queries[q].data()}};
          auto query_param =
              OmegaQueryParamBuilder()
                  .with_topk(static_cast<uint32_t>(kOmegaTrainingTopk))
                  .with_fetch_vector(false)
                  .with_ef_search(param.ef_training)
                  .with_training_query_id(static_cast<int>(q))
                  .with_target_recall(0.95f)
                  .build();
          if (index->Search(query, query_param, &search_result) != 0) {
            LOG_WARN("OMEGA training search failed for query %zu", q);
            continue;
          }

          QueryTrainingArtifacts artifacts;
          artifacts.records = std::move(search_result.training_records_);
          artifacts.gt_cmps_per_rank =
              std::move(search_result.gt_cmps_per_rank_);
          artifacts.total_cmps = search_result.total_cmps_;
          artifacts.training_query_id = search_result.training_query_id_;
          session->CollectQueryArtifacts(std::move(artifacts));
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }

  {
    ScopedTimer timer(timing_stats, "Step5: CollectTrainingRecords");
    TrainingArtifacts artifacts = session->ConsumeArtifacts();
    result->records = std::move(artifacts.records);
    result->gt_cmps_data = std::move(artifacts.gt_cmps_data);
  }
  result->training_queries = training_queries;
  result->query_doc_ids = query_doc_ids;
  return true;
}

bool SampleHeldOutQueries(const OmegaIndexParam &param, Index *index,
                          std::vector<std::vector<float>> *training_queries,
                          std::vector<uint64_t> *query_doc_ids) {
  const uint32_t doc_count = index->GetDocCount();
  if (doc_count == 0) {
    LOG_WARN("OMEGA training skipped because index is empty");
    return false;
  }

  const size_t sample_count =
      std::min<size_t>(param.num_training_queries, doc_count);
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint32_t> dist(0, doc_count - 1);

  training_queries->clear();
  query_doc_ids->clear();
  training_queries->reserve(sample_count);
  query_doc_ids->reserve(sample_count);

  for (size_t i = 0; i < sample_count; ++i) {
    const uint32_t doc_id = dist(rng);
    std::vector<float> values;
    if (!FetchDenseVector(param, index, doc_id, &values)) {
      continue;
    }
    training_queries->push_back(std::move(values));
    query_doc_ids->push_back(doc_id);
  }
  return !training_queries->empty();
}

omega::TrainingRecord ConvertTrainingRecord(const TrainingRecord &src) {
  omega::TrainingRecord dst;
  dst.query_id = src.query_id;
  dst.hops_visited = src.hops_visited;
  dst.cmps_visited = src.cmps_visited;
  dst.dist_1st = src.dist_1st;
  dst.dist_start = src.dist_start;
  dst.traversal_window_stats.assign(src.traversal_window_stats.begin(),
                                    src.traversal_window_stats.end());
  dst.label = src.label;
  return dst;
}

omega::GtCmpsData ConvertGtCmpsData(const GtCmpsData &src) {
  omega::GtCmpsData dst;
  dst.num_queries = src.num_queries;
  dst.topk = src.topk;
  dst.gt_cmps = src.gt_cmps;
  dst.total_cmps = src.total_cmps;
  return dst;
}

bool TrainModel(const OmegaTrainingArtifacts &artifacts,
                const std::string &model_output_dir) {
  if (artifacts.records.empty()) {
    LOG_WARN("No OMEGA training records collected, skipping model training");
    return true;
  }

  std::vector<omega::TrainingRecord> omega_records;
  omega_records.reserve(artifacts.records.size());
  for (const auto &record : artifacts.records) {
    omega_records.push_back(ConvertTrainingRecord(record));
  }
  std::sort(
      omega_records.begin(), omega_records.end(),
      [](const omega::TrainingRecord &lhs, const omega::TrainingRecord &rhs) {
        if (lhs.query_id != rhs.query_id) {
          return lhs.query_id < rhs.query_id;
        }
        if (lhs.cmps_visited != rhs.cmps_visited) {
          return lhs.cmps_visited < rhs.cmps_visited;
        }
        if (lhs.hops_visited != rhs.hops_visited) {
          return lhs.hops_visited < rhs.hops_visited;
        }
        return lhs.label < rhs.label;
      });

  omega::OmegaTrainerOptions trainer_options;
  trainer_options.output_dir = model_output_dir;
  trainer_options.num_iterations = 100;
  trainer_options.num_leaves = 31;
  trainer_options.learning_rate = 0.1;
  trainer_options.num_threads = DefaultOmegaTrainerThreads();
  trainer_options.seed = 42;
  trainer_options.deterministic = true;
  trainer_options.verbose = true;
  trainer_options.topk = artifacts.gt_cmps_data.topk > 0
                             ? artifacts.gt_cmps_data.topk
                             : kOmegaTrainingTopk;

  if (omega::OmegaTrainer::TrainModel(omega_records,
                                      ConvertGtCmpsData(artifacts.gt_cmps_data),
                                      trainer_options) != 0) {
    LOG_ERROR("OMEGA model training failed");
    return false;
  }

  return true;
}

}  // namespace

// OmegaIndex owns the framework-facing index lifecycle and delegates OMEGA-
// specific runtime behavior to OmegaStreamer/OmegaSearcher. It is responsible
// for creating the correct streamer and injecting OMEGA query params into the
// search context. It does not own the adaptive-search algorithm itself.
int OmegaIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const OmegaIndexParam &>(param);

  // Reuse HNSWIndex setup so the HNSW-compatible on-disk/index metadata is
  // initialized consistently before swapping in the OMEGA-aware streamer.
  int ret = HNSWIndex::CreateAndInitStreamer(param);
  if (ret != core::IndexError_Success) {
    return ret;
  }

  proxima_index_params_.insert("omega.enabled", true);
  proxima_index_params_.insert("omega.min_vector_threshold",
                               param_.min_vector_threshold);
  proxima_index_params_.insert("omega.window_size", param_.window_size);

  // OMEGA build/merge still happens through the streamer path. Keeping the
  // HNSW builder path untouched avoids changing merge semantics.
  core::IndexMeta saved_meta = proxima_index_meta_;
  ailego::Params saved_params = proxima_index_params_;

  streamer_ = core::IndexFactory::CreateStreamer("OmegaStreamer");
  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create OmegaStreamer");
    return core::IndexError_Runtime;
  }

  if (ailego_unlikely(streamer_->init(saved_meta, saved_params) != 0)) {
    LOG_ERROR("Failed to init OmegaStreamer");
    return core::IndexError_Runtime;
  }

  // Persist the OMEGA-aware searcher type so reopened indices route searches
  // through OmegaSearcher instead of the plain HNSW searcher.
  proxima_index_meta_.set_searcher("OmegaSearcher", 0, ailego::Params());

  return core::IndexError_Success;
}

int OmegaIndex::Train() {
  if (!is_open_) {
    LOG_ERROR("Index must be open before OMEGA training");
    return core::IndexError_Runtime;
  }
  if (is_sparse_) {
    LOG_ERROR("OMEGA training does not support sparse indexes");
    return core::IndexError_Unsupported;
  }
  if (is_read_only_) {
    LOG_ERROR("OMEGA training does not support read-only indexes");
    return core::IndexError_Runtime;
  }
  if (file_path_.empty()) {
    LOG_ERROR("OMEGA training requires a persistent index path");
    return core::IndexError_Runtime;
  }

  const uint32_t doc_count = GetDocCount();
  if (doc_count == 0) {
    is_trained_ = true;
    return 0;
  }

  is_training_ = true;
  AILEGO_DEFER([&]() { is_training_ = false; });

  if (!EnsureOmegaModelDir(file_path_)) {
    return core::IndexError_Runtime;
  }

  if (doc_count < param_.min_vector_threshold) {
    LOG_INFO(
        "Skipping OMEGA model training: doc_count=%u < min_vector_threshold=%u",
        doc_count, param_.min_vector_threshold);
    if (Flush() != 0) {
      return core::IndexError_Runtime;
    }
    is_trained_ = true;
    return 0;
  }

  std::vector<std::pair<std::string, int64_t>> timing_stats;
  OmegaTrainingArtifacts training_artifacts;
  std::vector<std::vector<float>> training_queries;
  std::vector<uint64_t> query_doc_ids;

  {
    ScopedTimer timer(&timing_stats, "Step1: GenerateHeldOutQueries");
    if (!LoadOmegaTrainingQueryCache(file_path_, &training_queries,
                                     &query_doc_ids)) {
      if (!SampleHeldOutQueries(param_, this, &training_queries,
                                &query_doc_ids)) {
        LOG_ERROR("Failed to generate held-out queries for OMEGA training");
        return core::IndexError_Runtime;
      }
    } else {
      LOG_INFO("Loaded %zu cached OMEGA training queries",
               training_queries.size());
    }
  }

  if (!CollectTrainingData(this, param_, training_queries, query_doc_ids,
                           &training_artifacts, &timing_stats)) {
    return core::IndexError_Runtime;
  }

  if (!SaveOmegaTrainingQueryCache(file_path_,
                                   training_artifacts.training_queries,
                                   training_artifacts.query_doc_ids)) {
    LOG_WARN("Failed to persist OMEGA training query cache");
  }

  WriteTimingStatsJson(
      (GetOmegaModelDir(file_path_) / "training_collection_timing.json")
          .string(),
      timing_stats);

  if (Flush() != 0) {
    return core::IndexError_Runtime;
  }

  if (training_artifacts.records.size() < 100) {
    LOG_WARN("Skipping OMEGA model training: only %zu records collected",
             training_artifacts.records.size());
    is_trained_ = true;
    return 0;
  }

  const std::string model_output_dir = GetOmegaModelDir(file_path_).string();
  if (!TrainModel(training_artifacts, model_output_dir)) {
    return core::IndexError_Runtime;
  }

  auto *omega_streamer = dynamic_cast<core::OmegaStreamer *>(streamer_.get());
  if (omega_streamer == nullptr ||
      !omega_streamer->ReloadModel(model_output_dir)) {
    LOG_ERROR("Failed to reload trained OMEGA model from %s",
              model_output_dir.c_str());
    return core::IndexError_Runtime;
  }

  is_trained_ = true;
  return 0;
}

ITrainingSession::Pointer OmegaIndex::CreateTrainingSession() {
  if (auto *omega_streamer =
          streamer_ ? dynamic_cast<core::OmegaStreamer *>(streamer_.get())
                    : nullptr) {
    return std::make_shared<OmegaTrainingSession>(omega_streamer);
  }
  return nullptr;
}

int OmegaIndex::_prepare_for_search(
    const VectorData &vector_data,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  // First call parent for base HNSW parameter handling (ef_search, etc.)
  int ret = HNSWIndex::_prepare_for_search(vector_data, search_param, context);
  if (ret != 0) {
    return ret;
  }

  ailego::Params params;

  // Extract OMEGA-specific parameter (target_recall)
  const auto &omega_search_param =
      std::dynamic_pointer_cast<OmegaQueryParam>(search_param);
  if (omega_search_param) {
    params.set(core::PARAM_OMEGA_SEARCHER_TARGET_RECALL,
               omega_search_param->target_recall);
    // Pass training_query_id for parallel training searches
    if (omega_search_param->training_query_id >= 0) {
      params.set(core::PARAM_OMEGA_SEARCHER_TRAINING_QUERY_ID,
                 omega_search_param->training_query_id);
    }
  }

  if (!params.empty()) {
    context->update(params);
  }

  return 0;
}

}  // namespace zvec::core_interface
