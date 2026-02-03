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

#include <functional>
#include <memory>
#include <variant>
#include <vector>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/db/query_params.h>
#include <zvec/db/type.h>
#include "db/index/common/index_filter.h"

namespace zvec {
class VectorColumnIndexer;

namespace vector_column_params {
struct DenseVector {
  const void *data;
};

struct SparseVector {
  uint32_t count;
  const void *indices;  // uint32
  const void *values;   // FP16/FP32
};

struct VectorData {
  std::variant<DenseVector, SparseVector> vector;
};


// VectorData with memory management
struct DenseVectorBuffer {
  std::string data;  // use string to manage memory
};

struct SparseVectorBuffer {
  std::string indices;  // uint32_t
  std::string values;

  using IndexType = uint32_t;
  uint32_t count() const {
    return indices.size() / sizeof(IndexType);
  }
};

struct VectorDataBuffer {
  std::variant<DenseVectorBuffer, SparseVectorBuffer> vector_buffer;
};


struct ReadOptions {
  bool use_mmap{true};
  bool create_new{false};
  bool read_only{false};
};

struct MergeOptions {
  uint32_t write_concurrency{1};
  ailego::ThreadPool *pool{nullptr};
};

struct GroupByParams {
  GroupByParams(uint32_t group_topk, uint32_t group_count,
                std::function<std::string(uint64_t key)> group_by)
      : group_topk(group_topk),
        group_count(group_count),
        group_by(std::move(group_by)) {}

  uint32_t group_topk{0};
  uint32_t group_count{0};
  std::function<std::string(uint64_t key)> group_by{};
};

struct RefinerParam {
  float scale_factor_{10};
  std::shared_ptr<VectorColumnIndexer> reference_indexer{nullptr};
};

// This is an internal version, while QueryParams in doc.h is an interface ver
struct QueryParams {
  DataType data_type{DataType::UNDEFINED};
  uint32_t dimension{0U};
  uint32_t topk{0U};
  mutable const IndexFilter *filter{nullptr};
  bool fetch_vector{false};
  zvec::QueryParams::Ptr query_params;
  std::unique_ptr<GroupByParams> group_by;
  // TODO: 1. should be uint32? 2. if no batch mode, change to optional<vector>
  std::vector<std::vector<uint64_t>> bf_pks{};

  std::shared_ptr<RefinerParam> refiner_param{nullptr};

  ailego::Params extra_params{};
};
}  // namespace vector_column_params
}  // namespace zvec
