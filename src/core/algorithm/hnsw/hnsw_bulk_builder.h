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
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_streamer.h>
#include <zvec/core/framework/vector_source.h>

namespace zvec::core {

// Internal fresh-build path. Source records must already match the target's
// stored encoding and remain immutable until the builder is destroyed.
class HnswBulkBuilder final : private VectorSource {
 public:
  HnswBulkBuilder(const IndexMeta &meta, const ailego::Params &params,
                  std::shared_ptr<turbo::Quantizer> quantizer,
                  const IndexQueryMeta &vector_meta);
  ~HnswBulkBuilder() override;

  HnswBulkBuilder(const HnswBulkBuilder &) = delete;
  HnswBulkBuilder &operator=(const HnswBulkBuilder &) = delete;

  // NotImplemented permits fallback before any filters have been evaluated.
  int add_source(IndexProvider::Pointer provider);
  int prepare(const IndexFilter &filter);
  bool empty() const {
    return vectors_.empty();
  }

  // Owns scratch files, including the packed output until its caller renames
  // it over the unpublished target. Does not open or replace the target.
  int build(const std::string &target_path, uint32_t write_concurrency,
            ailego::ThreadPool *pool);
  const std::string &packed_path() const {
    return packed_path_;
  }

 private:
  const void *get_vector(uint32_t id) const override {
    return id < vectors_.size() ? vectors_[id] : nullptr;
  }

  IndexMeta meta_;
  ailego::Params params_;
  std::shared_ptr<turbo::Quantizer> quantizer_;
  IndexQueryMeta vector_meta_;
  std::vector<std::shared_ptr<const void>> lifetimes_;
  std::vector<IndexProvider::Pointer> providers_;
  std::vector<const void *> vectors_;
  std::string packed_path_;
};

}  // namespace zvec::core
