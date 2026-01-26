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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <magic_enum/magic_enum.hpp>
#include "core/framework/index_context.h"
#include "core/framework/index_converter.h"
#include "core/framework/index_factory.h"
#include "core/framework/index_filter.h"
#include "core/framework/index_holder.h"
#include "core/framework/index_meta.h"
#include "core/framework/index_metric.h"
#include "core/framework/index_reducer.h"
#include "core/framework/index_reformer.h"
#include "core/framework/index_searcher.h"
#include "core/framework/index_storage.h"
#include "core/mixed_reducer/mixed_reducer_params.h"
#include "index_param.h"
#include "index_param_builders.h"

namespace zvec::core_interface {

class IndexFactory;

struct DenseVector {
  const void *data;
  // core::IndexQueryMeta meta;
  // DenseVector(void *data) : data(data) {
  //   meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  // };
};

struct SparseVector {
  uint32_t count;
  const void *indices;
  const void *values;

  const uint32_t *get_indices() const {
    return reinterpret_cast<const uint32_t *>(indices);
  }

  template <typename T = void>
  const T *get_values() const {
    return reinterpret_cast<const T *>(values);
  }
};

struct VectorData {
  std::variant<DenseVector, SparseVector> vector;

  // DenseVector dense_vector;
  // SparseVector sparse_vector;
};

// Used to pass mutable vectors
struct DenseVectorBuffer {
  std::string data;  // use string to manage memory
};

struct SparseVectorBuffer {
  uint32_t count;
  std::string indices;
  std::string values;

  uint32_t *get_indices() {
    return reinterpret_cast<uint32_t *>(indices.data());
  }

  template <typename T = void>
  T *get_values() {
    return reinterpret_cast<T *>(values.data());
  }
};

struct VectorDataBuffer {
  std::variant<DenseVectorBuffer, SparseVectorBuffer> vector_buffer;
};


struct SearchResult {
  core::IndexDocumentList doc_list_;
  // use string to manage memory
  std::vector<std::string> reverted_vector_list_{};
  std::vector<std::string> reverted_sparse_values_list_{};
};

// eliminate the pre-alloc of the context pool
thread_local static std::array<core::IndexContext::Pointer,
                               (magic_enum::enum_count<IndexType>() - 1) * 2>
    _context_list;

class Index {
 public:
  typedef std::shared_ptr<Index> Pointer;
  virtual ~Index() = default;

  // static Index::Pointer Create(const BaseIndexParam &param); //IndexFactory
  virtual int Open(const std::string &file_path,
                   StorageOptions storage_options);
  int Close();
  int Flush();
  // virtual int Serialize(const std::string &file_path);
  // virtual int Deserialize(const std::string &file_path);

  // // TODO: use holder
  // virtual int Build() = 0;
  virtual int Train() {
    is_trained_ = true;
    return 0;
  }

  // virtual int Dump(const std::string &file_path) = 0;
  virtual int Merge(const std::vector<Index::Pointer> &indexes,
                    const IndexFilter &filter,
                    const MergeOptions &options = {});
  // TODO: static reduce

  virtual int Add(const VectorData &vector, uint32_t doc_id);
  virtual int Fetch(const uint32_t doc_id,
                    VectorDataBuffer *vector_data_buffer);
  virtual int Search(const VectorData &query,
                     const BaseIndexQueryParam::Pointer &search_param,
                     SearchResult *result);

  virtual BaseIndexParam::Pointer GetParam() const {
    return std::make_shared<BaseIndexParam>(param_);
  }

  virtual bool IsTrained() const {
    return is_trained_;
  }

  uint32_t GetDocCount() const {
    if (streamer_ == nullptr) {
      return -1;
    }
    if (is_sparse_) {
      return streamer_->create_sparse_provider()->count();
    } else {
      return streamer_->create_provider()->count();
    }
  }

  core::IndexStreamer::Pointer index_searcher() {
    return streamer_;
  }

 protected:
  int _sparse_fetch(const uint32_t doc_id,
                    VectorDataBuffer *vector_data_buffer);
  virtual int _dense_fetch(const uint32_t doc_id,
                           VectorDataBuffer *vector_data_buffer);

  int _sparse_add(const VectorData &vector, const uint32_t doc_id,
                  core::IndexContext::Pointer &context);
  int _dense_add(const VectorData &vector, const uint32_t doc_id,
                 core::IndexContext::Pointer &context);
  int _sparse_search(const VectorData &query,
                     const BaseIndexQueryParam::Pointer &search_param,
                     SearchResult *result,
                     core::IndexContext::Pointer &context);
  int _dense_search(const VectorData &query,
                    const BaseIndexQueryParam::Pointer &search_param,
                    SearchResult *result, core::IndexContext::Pointer &context);
  virtual int _prepare_for_search(
      const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
      core::IndexContext::Pointer &context) = 0;
  virtual int _get_coarse_search_topk(
      const BaseIndexQueryParam::Pointer &search_param);

 protected:
  friend class IndexFactory;
  Index() = default;
  int Init(const BaseIndexParam &param);


 protected:
  int ParseMetricName(const BaseIndexParam &param);
  int CreateAndInitMetric(const BaseIndexParam &param);
  int CreateAndInitConverterReformer(const QuantizerParam &param,
                                     const BaseIndexParam &index_param);
  virtual int CreateAndInitStreamer(const BaseIndexParam &param) = 0;

 protected:
  bool init_context() {
    context_index_ = (magic_enum::enum_integer(param_.index_type) - 1) * 2 +
                     static_cast<size_t>(is_sparse_);
    if (_context_list[context_index_] == nullptr) {
      if ((_context_list[context_index_] = streamer_->create_context()) ==
          nullptr) {
        LOG_ERROR("Failed to create context");
        return false;
      }
    }
    return true;
  }

  core::IndexContext::Pointer &acquire_context() {
    init_context();
    return _context_list[context_index_];
  }

  void release_context() {
    // context_list_[get_context_index()]->reset();
  }

 protected:
  bool is_trained_{false};

  BaseIndexParam param_;
  ailego::Params proxima_index_params_{};
  core::IndexMeta proxima_index_meta_{};  // IndexQueryMeta + other index config
  core::IndexQueryMeta input_vector_meta_;     // input
  core::IndexQueryMeta streamer_vector_meta_;  // after reformer.convert()

  core::IndexBuilder::Pointer builder_{};
  core::IndexStreamer::Pointer streamer_{};
  core::IndexReformer::Pointer reformer_{};
  core::IndexConverter::Pointer converter_{};  // for build()
  core::IndexMetric::Pointer metric_{};        // to do normalization

  size_t context_index_;
  core::IndexStorage::Pointer storage_{};

  bool is_open_{false};
  bool is_sparse_{false};
  bool is_huge_page_{false};
  bool is_read_only_{false};
};


class FlatIndex : public Index {
 public:
  FlatIndex() = default;
  // FlatIndex(const FlatIndexParam &param) : param_(param) {}
  // FlatIndex(FlatIndexParam &&param) : param(std::move(param)) {}


 protected:
  virtual int CreateAndInitStreamer(const BaseIndexParam &param) override;

  virtual int _prepare_for_search(
      const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
      core::IndexContext::Pointer &context) override;

 private:
  FlatIndexParam param_{};
};

class IVFIndex : public Index {
 public:
  IVFIndex() = default;

 protected:
  virtual int CreateAndInitStreamer(const BaseIndexParam &param) override;

  virtual int _prepare_for_search(
      const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
      core::IndexContext::Pointer &context) override;

  virtual int Add(const VectorData &vector, uint32_t doc_id) override;

  virtual int Train() override;

  virtual int Open(const std::string &file_path,
                   StorageOptions storage_options) override;

  virtual int _dense_fetch(const uint32_t doc_id,
                           VectorDataBuffer *vector_data_buffer) override;
  virtual int Merge(const std::vector<Index::Pointer> &indexes,
                    const IndexFilter &filter,
                    const MergeOptions &options) override;
  int GenerateHolder();

 private:
  IVFIndexParam param_{};
  std::mutex mutex_{};
  std::vector<std::pair<uint64_t, std::string>> doc_cache_;
  core::IndexHolder::Pointer holder_{};
  std::string file_path_;
};


class HNSWIndex : public Index {
 public:
  HNSWIndex() = default;

 protected:
  virtual int CreateAndInitStreamer(const BaseIndexParam &param) override;

  virtual int _prepare_for_search(
      const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
      core::IndexContext::Pointer &context) override;
  int _get_coarse_search_topk(
      const BaseIndexQueryParam::Pointer &search_param) override;


 private:
  HNSWIndexParam param_{};
};


}  // namespace zvec::core_interface
