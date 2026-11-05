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

#include "hnsw_bulk_builder.h"
#include <atomic>
#include <chrono>
#include <limits>
#include <zvec/core/framework/index_framework.h>
#include "utility/stable_vector_holder.h"
#include "hnsw_build_dumper.h"
#include "hnsw_streamer.h"

namespace zvec::core {

HnswBulkBuilder::HnswBulkBuilder(const IndexMeta &meta,
                                 const ailego::Params &params,
                                 std::shared_ptr<turbo::Quantizer> quantizer,
                                 const IndexQueryMeta &vector_meta)
    : meta_(meta),
      params_(params),
      quantizer_(std::move(quantizer)),
      vector_meta_(vector_meta) {}

HnswBulkBuilder::~HnswBulkBuilder() {
  if (!packed_path_.empty()) ailego::File::Delete(packed_path_);
}

int HnswBulkBuilder::add_source(IndexProvider::Pointer provider) {
  // Establish capabilities before evaluating any filters. Each lifetime keeps
  // iterator addresses valid even after its provider has been released.
  auto *stable = dynamic_cast<StableVectorHolder *>(provider.get());
  if (!stable) return IndexError_NotImplemented;
  std::shared_ptr<const void> lifetime;
  int ret = stable->retain_stable_data(&lifetime);
  if (ret != 0) return ret;
  if (!lifetime) return IndexError_Runtime;
  lifetimes_.emplace_back(std::move(lifetime));
  providers_.emplace_back(std::move(provider));
  return 0;
}

int HnswBulkBuilder::prepare(const IndexFilter &filter) {
  // Cache decisions once, using reduce()'s source-key offsets and dense IDs.
  uint64_t base = 0;
  for (const auto &provider : providers_) {
    if (provider->count() > std::numeric_limits<uint32_t>::max() - base) {
      return IndexError_Overflow;
    }
    auto iter = provider->create_iterator();
    if (!iter) return IndexError_Runtime;
    for (; iter->is_valid(); iter->next()) {
      const uint64_t key = iter->key();
      if (iter->status() != 0) return iter->status();
      if (key > std::numeric_limits<uint32_t>::max() - base) {
        return IndexError_Overflow;
      }
      if (filter.is_valid() && filter(base + key)) continue;
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      if (vectors_.size() >= std::numeric_limits<uint32_t>::max()) {
        return IndexError_Overflow;
      }
      vectors_.emplace_back(data);
    }
    if (iter->status() != 0) return iter->status();
    base += provider->count();
  }
  return 0;
}

int HnswBulkBuilder::build(const std::string &target_path,
                           const VectorSource &source,
                           uint32_t write_concurrency,
                           ailego::ThreadPool *pool) {
  // A new attempt must not be blocked by scratch left by a crashed optimize.
  const std::string suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string graph_path = target_path + ".hnsw-build." + suffix;
  const std::string packed_path = target_path + ".hnsw-pack." + suffix;
  if (ailego::File::IsExist(graph_path) || ailego::File::IsExist(packed_path)) {
    return IndexError_Duplicate;
  }
  packed_path_ = packed_path;
  auto graph_storage = IndexFactory::CreateStorage("MMapFileStorage");
  auto graph_owner = IndexFactory::CreateStreamer("HnswStreamer");
  IndexStreamer *build = graph_owner.get();
  auto *graph = dynamic_cast<HnswStreamer *>(build);
  if (!graph_storage || !graph) return IndexError_NoMemory;
  // Release mappings and file handles before removing scratch files, including
  // on failed builds and writes. The packed file stays owned by this builder.
  AILEGO_DEFER([&]() {
    build->cleanup();
    graph_storage->close();
    ailego::File::Delete(graph_path);
  });
  int ret = graph_storage->init(ailego::Params{});
  if (ret != 0) return ret;
  ret = graph_storage->open(graph_path, true);
  if (ret != 0) return ret;
  ret = build->init(meta_, params_, quantizer_);
  if (ret != 0) return ret;
  ret = graph->open_with_borrowed_vectors(graph_storage, &source);
  if (ret != 0) return ret;

  std::unique_ptr<ailego::ThreadPool> local_pool;
  if (!pool) {
    local_pool = std::make_unique<ailego::ThreadPool>(write_concurrency, false);
    pool = local_pool.get();
  }
  const uint32_t workers = std::min<uint32_t>(write_concurrency, pool->count());
  // clone() traverses the main entity's chunk containers. Finish every clone
  // before the first insertion can grow those containers on a worker thread.
  std::vector<IndexStreamer::Context::Pointer> contexts(workers);
  for (auto &context : contexts) {
    context = build->create_context();
    if (!context) return IndexError_NoMemory;
  }
  std::atomic<size_t> next{0};
  std::atomic<int> status{0};
  auto group = pool->make_group();
  for (uint32_t worker = 0; worker < workers; ++worker) {
    group->submit(ailego::Closure::New([&, worker]() {
      auto &context = contexts[worker];
      while (status.load() == 0) {
        const size_t id = next.fetch_add(1);
        if (id >= vectors_.size()) break;
        const int result =
            build->add_with_id_impl(id, vectors_[id], vector_meta_, context);
        if (result != 0) status.store(result);
      }
    }));
  }
  group->wait_finish();
  if (status.load() != 0) return status.load();

  auto dumper =
      std::make_shared<HnswBuildDumper>(graph->packed_segment_count());
  ret = dumper->init(ailego::Params{});
  if (ret != 0) return ret;
  ret = dumper->create(packed_path_);
  if (ret != 0) return ret;
  ret = graph->dump_with_vectors(dumper);
  return ret != 0 ? ret : dumper->close();
}

}  // namespace zvec::core
