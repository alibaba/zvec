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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "zvec/core/framework/index_factory.h"
#include "utility/rdtsc_timer.h"
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_searcher.h"
#include "omega/search_context.h"

namespace zvec {
namespace core {

namespace {

struct Options {
  std::string index_path;
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
      << "  --ef-search <n>      HNSW ef_search\n"
      << "  --topk <n>           Search topk\n"
      << "  --query-count <n>    Number of sampled queries\n"
      << "  --iterations <n>     Number of measured iterations\n"
      << "  --warmup <n>         Number of warmup iterations\n"
      << "  --seed <n>           RNG seed for sampled queries\n"
      << "  --window-size <n>    OMEGA window size for hooks-only mode\n"
      << "  --target-recall <f>  OMEGA target recall for hooks-only mode\n";
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--index-path") == 0 && i + 1 < argc) {
      opts->index_path = argv[++i];
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
  return true;
}

class ExposedHnswSearcher : public HnswSearcher {
 public:
  int Init(const ailego::Params& params) { return HnswSearcher::init(params); }

  int Load(IndexStorage::Pointer container) {
    return HnswSearcher::load(std::move(container), nullptr);
  }

  ContextPointer CreateContext() const { return HnswSearcher::create_context(); }

  int FastSearch(HnswContext* ctx) const { return fast_search(ctx); }

  int FastSearchWithHooks(HnswContext* ctx,
                          const HnswAlgorithm::SearchHooks* hooks,
                          bool* stopped_early) const {
    return fast_search_with_hooks(ctx, hooks, stopped_early);
  }

  const IndexMeta& MetaPublic() const { return meta(); }
};

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

std::vector<const void*> SampleIndexVectors(HnswContext* ctx, const IndexMeta& meta,
                                            uint32_t count, uint32_t seed) {
  const auto& entity = ctx->get_entity();
  const uint32_t doc_cnt = static_cast<uint32_t>(entity.doc_cnt());
  std::vector<uint32_t> ids(doc_cnt);
  std::iota(ids.begin(), ids.end(), 0u);
  std::mt19937 rng(seed);
  std::shuffle(ids.begin(), ids.end(), rng);

  count = std::min(count, doc_cnt);
  std::vector<const void*> queries;
  queries.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    queries.push_back(entity.get_vector(ids[i]));
  }

  std::cout << "Using " << queries.size() << " sampled in-index queries"
            << " element_size=" << meta.element_size()
            << " doc_cnt=" << doc_cnt << "\n";
  return queries;
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

  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    return 1;
  }

  ailego::Params params;
  params.set(PARAM_HNSW_SEARCHER_EF, opts.ef_search);

  ExposedHnswSearcher searcher;
  if (searcher.Init(params) != 0) {
    std::cerr << "Failed to init HNSW searcher\n";
    return 2;
  }

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  if (!storage || storage->open(opts.index_path, false) != 0) {
    std::cerr << "Failed to open index storage: " << opts.index_path << "\n";
    return 3;
  }

  if (searcher.Load(storage) != 0) {
    std::cerr << "Failed to load HNSW searcher\n";
    return 4;
  }

  auto context = searcher.CreateContext();
  auto* ctx = dynamic_cast<HnswContext*>(context.get());
  if (ctx == nullptr) {
    std::cerr << "Failed to create HNSW context\n";
    return 5;
  }
  ctx->set_topk(opts.topk);

  auto queries = SampleIndexVectors(ctx, searcher.MetaPublic(), opts.query_count,
                                    opts.seed);
  if (queries.empty()) {
    std::cerr << "No queries sampled from index\n";
    return 6;
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

  RunBench("alg_fast_search", ctx, queries, opts.warmup, opts.iterations, [&]() {
    return searcher.FastSearch(ctx);
  });

  RunBench("alg_fast_search_with_empty_hooks", ctx, queries, opts.warmup,
           opts.iterations, [&]() {
             bool stopped_early = false;
             return searcher.FastSearchWithHooks(ctx, &empty_hooks,
                                                &stopped_early);
           });

  RunBench("alg_fast_search_with_omega_hooks_only", ctx, queries, opts.warmup,
           opts.iterations, [&]() {
             omega_search_ctx.Reset();
             bool stopped_early = false;
             return searcher.FastSearchWithHooks(ctx, &omega_hooks,
                                                &stopped_early);
           });

  return 0;
}
