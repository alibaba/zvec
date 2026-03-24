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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "zvec/ailego/container/params.h"
#include "utility/rdtsc_timer.h"
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "db/common/file_helper.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/common/version_manager.h"
#include "omega/search_context.h"

namespace zvec {
namespace core {

namespace {

namespace fs = std::filesystem;

struct Options {
  std::string mode = "all";
  std::string index_path;
  uint32_t dimension = 768;
  uint32_t m = 15;
  uint32_t ef_construction = 500;
  uint32_t ef_search = 180;
  uint32_t topk = 100;
  uint32_t query_count = 1000;
  uint32_t iterations = 1000;
  uint32_t warmup = 100;
  uint32_t seed = 12345;
  int window_size = 100;
  float target_recall = 0.90f;
};

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --index-path <path> [options]\n"
      << "Options:\n"
      << "  --mode <name>        all|fast|empty|omega\n"
      << "  --dimension <n>      Vector dimension\n"
      << "  --m <n>              HNSW max neighbors\n"
      << "  --ef-construction <n> HNSW ef_construction\n"
      << "  --ef-search <n>      HNSW ef_search\n"
      << "  --topk <n>           Search topk\n"
      << "  --query-count <n>    Number of sampled queries\n"
      << "  --iterations <n>     Number of measured iterations\n"
      << "  --warmup <n>         Number of warmup iterations\n"
      << "  --seed <n>           RNG seed for sampled queries\n"
      << "  --window-size <n>    OMEGA window size for hooks-only mode\n"
      << "  --target-recall <f>  OMEGA target recall for hooks-only mode\n"
      << "\n"
      << "--index-path accepts either the benchmark index directory\n"
      << "(for example .../cohere_1m_hnsw) or a file under its segment\n"
      << "subdirectory such as .../0/dense.qindex.5.proxima.\n";
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--index-path") == 0 && i + 1 < argc) {
      opts->index_path = argv[++i];
    } else if (std::strcmp(arg, "--mode") == 0 && i + 1 < argc) {
      opts->mode = argv[++i];
    } else if (std::strcmp(arg, "--dimension") == 0 && i + 1 < argc) {
      opts->dimension =
          static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--m") == 0 && i + 1 < argc) {
      opts->m = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--ef-construction") == 0 && i + 1 < argc) {
      opts->ef_construction =
          static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--ef-search") == 0 && i + 1 < argc) {
      opts->ef_search = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--topk") == 0 && i + 1 < argc) {
      opts->topk = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--query-count") == 0 && i + 1 < argc) {
      opts->query_count =
          static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--iterations") == 0 && i + 1 < argc) {
      opts->iterations =
          static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
      opts->warmup =
          static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
      opts->seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--window-size") == 0 && i + 1 < argc) {
      opts->window_size = std::atoi(argv[++i]);
    } else if (std::strcmp(arg, "--target-recall") == 0 && i + 1 < argc) {
      opts->target_recall = std::strtof(argv[++i], nullptr);
    } else if (std::strcmp(arg, "--help") == 0 ||
               std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }

  if (opts->index_path.empty()) {
    PrintUsage(argv[0]);
    return false;
  }
  opts->query_count = std::max(1u, opts->query_count);
  opts->iterations = std::max(1u, opts->iterations);
  opts->topk = std::max(1u, opts->topk);
  opts->window_size = std::max(1, opts->window_size);
  if (opts->mode != "all" && opts->mode != "fast" &&
      opts->mode != "empty" && opts->mode != "omega") {
    std::cerr << "Invalid --mode: " << opts->mode << "\n";
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

struct OmegaHookState {
  omega::SearchContext* search_ctx{nullptr};
  bool enable_early_stopping{false};
};

void OnOmegaLevel0Entry(node_id_t id, dist_t dist, bool /*inserted_to_topk*/,
                        void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
  state.search_ctx->SetDistStart(dist);
  state.search_ctx->ReportVisitCandidate(id, dist, true);
}

void OnOmegaHop(void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
  state.search_ctx->ReportHop();
}

bool OnOmegaVisitCandidate(node_id_t id, dist_t dist,
                           bool should_consider_candidate, void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
  state.search_ctx->ReportVisitCandidate(id, dist, should_consider_candidate);
  if (!state.enable_early_stopping) {
    return false;
  }
  return state.search_ctx->ShouldPredict() &&
         state.search_ctx->ShouldStopEarly();
}

struct BenchStats {
  double avg_ns{0.0};
  double avg_cmps{0.0};
  double ns_per_cmp{0.0};
  double checksum{0.0};
};

std::string NormalizeIndexPath(const std::string& input_path) {
  if (input_path.empty()) {
    return input_path;
  }

  fs::path path(input_path);
  std::error_code ec;
  if (!fs::exists(path, ec)) {
    return input_path;
  }

  if (fs::is_regular_file(path, ec)) {
    auto segment_dir = path.parent_path();
    auto index_root = segment_dir.parent_path();
    if (!segment_dir.empty() && segment_dir.filename() == "0" &&
        !index_root.empty()) {
      return index_root.string();
    }
    return segment_dir.string();
  }

  if (fs::is_directory(path, ec) && path.filename() == "0") {
    auto index_root = path.parent_path();
    if (!index_root.empty()) {
      return index_root.string();
    }
  }

  return path.string();
}

std::vector<const void*> SampleIndexVectors(const IndexProvider::Pointer& provider,
                                            const IndexMeta& meta,
                                            uint32_t count, uint32_t seed) {
  const uint32_t doc_cnt = static_cast<uint32_t>(provider->count());
  std::vector<const void*> all_queries;
  all_queries.reserve(doc_cnt);

  auto it = provider->create_iterator();
  for (; it && it->is_valid(); it->next()) {
    all_queries.push_back(it->data());
  }

  std::vector<uint32_t> ids(all_queries.size());
  std::iota(ids.begin(), ids.end(), 0u);
  std::mt19937 rng(seed);
  std::shuffle(ids.begin(), ids.end(), rng);

  count = std::min<uint32_t>(count, static_cast<uint32_t>(all_queries.size()));
  std::vector<const void*> queries;
  queries.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    queries.push_back(all_queries[ids[i]]);
  }

  std::cout << "Using " << queries.size() << " sampled in-index queries"
            << " element_size=" << meta.element_size()
            << " doc_cnt=" << doc_cnt
            << " valid_vectors=" << all_queries.size() << "\n";
  return queries;
}

bool OpenBenchmarkVectorIndexer(const std::string& benchmark_root,
                                VectorColumnIndexer::Ptr* out,
                                std::string* error) {
  auto version_manager_result = VersionManager::Recovery(benchmark_root);
  if (!version_manager_result.has_value()) {
    *error = version_manager_result.error().message();
    return false;
  }

  auto version_manager = version_manager_result.value();
  const Version version = version_manager->get_current_version();
  const auto& schema = version.schema();
  const auto vector_fields = schema.vector_fields();
  if (vector_fields.empty()) {
    *error = "No vector field found in benchmark root";
    return false;
  }

  const FieldSchema* field = vector_fields.front().get();
  if (field == nullptr) {
    *error = "Invalid vector field in benchmark root";
    return false;
  }

  std::string index_file_path;
  uint32_t best_doc_count = 0;
  bool found_quantized = false;

  for (const auto& segment_meta : version.persisted_segment_metas()) {
    for (const auto& block : segment_meta->persisted_blocks()) {
      const bool match_field = block.contain_column(field->name());
      const bool is_quantized = block.type() == BlockType::VECTOR_INDEX_QUANTIZE;
      const bool is_plain = block.type() == BlockType::VECTOR_INDEX;
      if (!match_field || (!is_quantized && !is_plain)) {
        continue;
      }

      if (is_quantized && (!found_quantized || block.doc_count() > best_doc_count)) {
        index_file_path = FileHelper::MakeQuantizeVectorIndexPath(
            benchmark_root, field->name(), segment_meta->id(), block.id());
        best_doc_count = block.doc_count();
        found_quantized = true;
        continue;
      }

      if (!found_quantized && block.doc_count() > best_doc_count) {
        index_file_path = FileHelper::MakeVectorIndexPath(
            benchmark_root, field->name(), segment_meta->id(), block.id());
        best_doc_count = block.doc_count();
      }
    }
  }

  if (index_file_path.empty()) {
    *error = "No HNSW vector index file found under benchmark root";
    return false;
  }

  auto indexer = std::make_shared<VectorColumnIndexer>(index_file_path, *field);
  auto status = indexer->Open({true, false, true});
  if (!status.ok()) {
    *error = status.message();
    return false;
  }

  std::cout << "Opened benchmark root: " << benchmark_root << "\n"
            << "Selected vector index file: " << index_file_path << "\n"
            << "Vector field: " << field->name() << "\n";
  *out = std::move(indexer);
  return true;
}

template <typename Fn>
BenchStats RunBench(const std::string& name, HnswContext* ctx,
                    const std::vector<const void*>& queries, uint32_t warmup,
                    uint32_t iterations, Fn&& fn) {
  for (uint32_t i = 0; i < warmup; ++i) {
    const void* query = queries[i % queries.size()];
    ctx->clear();
    ctx->resize_results(1);
    ctx->reset_query(query);
    fn();
  }

  uint64_t total_ns = 0;
  uint64_t total_cmps = 0;
  double checksum = 0.0;

  for (uint32_t i = 0; i < iterations; ++i) {
    const void* query = queries[i % queries.size()];
    ctx->clear();
    ctx->resize_results(1);
    ctx->reset_query(query);
    const auto start = RdtscTimer::Now();
    fn();
    const auto end = RdtscTimer::Now();
    total_ns += RdtscTimer::ElapsedNs(start, end);
    total_cmps += ctx->get_pairwise_dist_num();
    if (!ctx->topk_heap().empty()) {
      checksum += ctx->topk_heap()[0].second;
    }
  }

  BenchStats stats;
  stats.avg_ns = static_cast<double>(total_ns) / iterations;
  stats.avg_cmps = static_cast<double>(total_cmps) / iterations;
  stats.ns_per_cmp =
      total_cmps == 0 ? 0.0 : static_cast<double>(total_ns) / total_cmps;
  stats.checksum = checksum;

  std::cout << std::fixed << std::setprecision(3)
            << name << ": avg_ns=" << stats.avg_ns
            << " avg_cmps=" << stats.avg_cmps
            << " ns_per_cmp=" << stats.ns_per_cmp
            << " checksum=" << stats.checksum << "\n";
  return stats;
}

}  // namespace

}  // namespace core
}  // namespace zvec

int main(int argc, char** argv) {
  using namespace zvec::core;
  using namespace zvec;

  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    return 1;
  }
  opts.index_path = NormalizeIndexPath(opts.index_path);

  VectorColumnIndexer::Ptr indexer;
  std::string open_error;
  if (!OpenBenchmarkVectorIndexer(opts.index_path, &indexer, &open_error)) {
    std::cerr << "Failed to open benchmark index: " << open_error << "\n";
    return 2;
  }
  auto index = indexer->core_index();
  if (!index) {
    std::cerr << "Opened benchmark indexer without underlying core index\n";
    return 3;
  }

  auto streamer_base = index->index_searcher();
  auto* streamer = dynamic_cast<HnswStreamer*>(streamer_base.get());
  if (streamer == nullptr) {
    std::cerr << "Failed to get HnswStreamer from opened index\n";
    return 4;
  }

  auto context = streamer_base->create_context();
  auto* ctx = dynamic_cast<HnswContext*>(context.get());
  if (ctx == nullptr) {
    std::cerr << "Failed to create HNSW context\n";
    return 5;
  }
  zvec::ailego::Params query_params;
  query_params.set(PARAM_HNSW_STREAMER_EF, opts.ef_search);
  if (ctx->update(query_params) != 0) {
    std::cerr << "Failed to update HNSW query params\n";
    return 6;
  }
  ctx->set_topk(opts.topk);

  auto provider = streamer_base->create_provider();
  if (!provider) {
    std::cerr << "Failed to create HNSW provider\n";
    return 7;
  }

  auto queries = SampleIndexVectors(provider, streamer_base->meta(),
                                    opts.query_count, opts.seed);
  if (queries.empty()) {
    std::cerr << "No queries sampled from index\n";
    return 8;
  }

  HnswAlgorithm::SearchHooks empty_hooks;

  omega::SearchContext omega_search_ctx(
      nullptr, nullptr, opts.target_recall, static_cast<int>(opts.topk),
      opts.window_size);
  OmegaHookState omega_hook_state;
  omega_hook_state.search_ctx = &omega_search_ctx;
  omega_hook_state.enable_early_stopping = false;
  HnswAlgorithm::SearchHooks omega_hooks;
  omega_hooks.user_data = &omega_hook_state;
  omega_hooks.on_level0_entry = OnOmegaLevel0Entry;
  omega_hooks.on_hop = OnOmegaHop;
  omega_hooks.on_visit_candidate = OnOmegaVisitCandidate;

  if (opts.mode == "all" || opts.mode == "fast") {
    RunBench("alg_fast_search", ctx, queries, opts.warmup, opts.iterations,
             [&]() { return streamer->FastSearch(ctx); });
  }

  if (opts.mode == "all" || opts.mode == "empty") {
    RunBench("alg_fast_search_with_empty_hooks", ctx, queries, opts.warmup,
             opts.iterations, [&]() {
               bool stopped_early = false;
               return streamer->FastSearchWithHooks(ctx, &empty_hooks,
                                                    &stopped_early);
             });
  }

  if (opts.mode == "all" || opts.mode == "omega") {
    RunBench("alg_fast_search_with_omega_hooks_only", ctx, queries, opts.warmup,
             opts.iterations, [&]() {
               omega_search_ctx.Reset();
               bool stopped_early = false;
               return streamer->FastSearchWithHooks(ctx, &omega_hooks,
                                                    &stopped_early);
             });
  }

  return 0;
}
