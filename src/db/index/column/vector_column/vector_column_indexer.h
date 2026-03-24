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
#include <string>
#include <utility>
#include <variant>
#include <mutex>
#include <map>
#include <ailego/parallel/lock.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/training.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/typedef.h"
#include "db/index/column/common/index_results.h"
#include "db/index/common/meta.h"
#include "vector_column_params.h"
#include "vector_index_results.h"

namespace zvec {

class ProximaEngineHelper;

class VectorColumnIndexer {
 public:
  using Ptr = std::shared_ptr<VectorColumnIndexer>;
  PROXIMA_DISALLOW_COPY_AND_ASSIGN(VectorColumnIndexer);

  VectorColumnIndexer(const std::string &index_file_path,
                      const FieldSchema &field_schema,
                      const std::string &engine_name = "proxima")
      : field_schema_(field_schema),
        index_file_path_(index_file_path),
        engine_name_(engine_name) {
    // assert(field_schema.is_dense_vector() ||
    // field_schema.is_sparse_vector());
    is_sparse_ = field_schema.is_sparse_vector();
  }

  virtual ~VectorColumnIndexer() = default;

 public:
  Status Open(const vector_column_params::ReadOptions &read_options);

  Status Flush();

  // Close will call Flush()
  Status Close();

  // Destroy will call Close() and remove index file
  Status Destroy();


  // If HNSWIndexer.merge([FlatIndexer1, FlatIndexer2])
  // then the merged indexer is a HNSWIndexer
  Status Merge(const std::vector<VectorColumnIndexer::Ptr> &indexers,
               const IndexFilter::Ptr &filter = nullptr,
               const vector_column_params::MergeOptions &merge_options = {});
  // TODO: should we use this function? or a Reducer?
  //  TODO: sstatic reduce, optimize; iterator/scan


  //! Insert vector
  Status Insert(const vector_column_params::VectorData &vector_data,
                uint32_t doc_id);
  // TODO: batch insert

  virtual Result<IndexResults::Ptr> Search(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params);
  // Result<std::vector<IndexResults::Ptr>> BatchSearch(
  //     const VectorDataset &vector_data,
  //     const  vector_column_params::QueryParams &query_params);

  Result<vector_column_params::VectorDataBuffer> Fetch(uint32_t doc_id) const;
  // Result<VectorDataset> BatchFetch(const std::vector<uint32_t> &doc_ids)
  // const;

 public:
  // OMEGA Training Mode Support
  /**
   * @brief Check if the underlying index supports training capability.
   *
   * @return Pointer to ITrainingCapable interface if supported, nullptr otherwise
   */
  core_interface::ITrainingCapable* GetTrainingCapability() const;

  /**
   * @brief Enable or disable training mode for collecting training features.
   *
   * Propagates the training mode setting to the underlying index.
   * When enabled, searches will collect training features.
   *
   * @param enable True to enable training mode, false to disable
   * @return Status indicating success or failure
   */
  Status EnableTrainingMode(bool enable);

  /**
   * @brief Set the query ID for the next search operation.
   *
   * Must be called before Search() when training mode is enabled.
   * The query_id will be propagated to the underlying index.
   *
   * @param query_id Unique identifier for the query
   */
  void SetCurrentQueryId(int query_id);

  /**
   * @brief Get all collected training records.
   *
   * Returns a copy of all training records collected from the
   * underlying index since training mode was enabled.
   *
   * @return Vector of TrainingRecord structures
   */
  std::vector<core_interface::TrainingRecord> GetTrainingRecords() const;

  /**
   * @brief Clear all collected training records.
   *
   * Clears training records from both this layer and the underlying index.
   */
  void ClearTrainingRecords();

  /**
   * @brief Set ground truth for training queries.
   *
   * Ground truth is used for real-time label computation during training.
   * Labels are computed as: label=1 iff top k_train GT nodes are in current topk.
   *
   * @param ground_truth 2D vector: ground_truth[query_id][rank] = node_id
   * @param k_train Number of GT nodes to check for label (typically 1)
   */
  void SetTrainingGroundTruth(const std::vector<std::vector<uint64_t>>& ground_truth,
                               int k_train = 1);

  /**
   * @brief Get collected gt_cmps data for all queries.
   *
   * Returns the gt_cmps data collected during training searches.
   * The data is indexed by query_id.
   *
   * @return GtCmpsData structure with per-query gt_cmps values
   */
  core_interface::GtCmpsData GetGtCmpsData() const;

 public:
  std::string index_file_path() const {
    return index_file_path_;
  }

  core_interface::Index::Pointer core_index() const {
    return index;
  }

  size_t doc_count() const {
    if (index == nullptr) {
      return -1;
    }
    return index->GetDocCount();
  }

  MetricType metric_type() const {
    auto index_params = field_schema_.index_params();
    if (index_params) {
      auto vector_params = std::dynamic_pointer_cast<VectorIndexParams>(index_params);
      if (vector_params) {
        return vector_params->metric_type();
      }
    }
    return MetricType::IP;  // default
  }

  // for ut
 protected:
  VectorColumnIndexer() = default;

 private:
  // protected:
  //  virtual bool init_proxima_params() = 0;

  // proxima or other engine index param like VSAGE
  // build proxima index
  Status CreateProximaIndex(
      const vector_column_params::ReadOptions &read_options);

 protected:
  friend ProximaEngineHelper;
  core_interface::Index::Pointer index;
  FieldSchema field_schema_{};
  std::string index_file_path_{};


  std::string engine_name_ = "proxima";
  bool is_sparse_{false};  // TODO: eliminate the dynamic flag and make it
                           // static/template/seperate class

  // Training mode support
  bool training_mode_enabled_{false};
  int current_query_id_{0};
  mutable std::mutex training_mutex_;
  mutable std::vector<core_interface::TrainingRecord> collected_records_;
  // GT cmps data: gt_cmps_map_[query_id] = {gt_cmps_per_rank, total_cmps}
  mutable std::map<int, std::pair<std::vector<int>, int>> gt_cmps_map_;
};


}  // namespace zvec
