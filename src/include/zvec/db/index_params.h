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
#include <sstream>
#include <string>
#include <zvec/core/interface/constants.h>
#include <zvec/db/type.h>

namespace zvec {

/*
 * Column index params
 */
class IndexParams {
 public:
  using Ptr = std::shared_ptr<IndexParams>;

  IndexParams(IndexType type) : type_(type) {}

  virtual ~IndexParams() = default;

  virtual Ptr clone() const = 0;

  virtual bool operator==(const IndexParams &other) const = 0;

  virtual std::string to_string() const = 0;

  virtual bool operator!=(const IndexParams &other) const {
    return !(*this == other);
  }

  bool is_vector_index_type() const {
    return type_ == IndexType::FLAT || type_ == IndexType::HNSW ||
           type_ == IndexType::IVF || type_ == IndexType::OMEGA;
  }

  IndexType type() const {
    return type_;
  }

 protected:
  IndexType type_;
};

/*
 * Scalar: Invert index params
 */
class InvertIndexParams : public IndexParams {
 public:
  InvertIndexParams(bool enable_range_optimization = true,
                    bool enable_extended_wildcard = false)
      : IndexParams(IndexType::INVERT),
        enable_range_optimization_(enable_range_optimization),
        enable_extended_wildcard_(enable_extended_wildcard) {}

  using OPtr = std::shared_ptr<InvertIndexParams>;

  Ptr clone() const override {
    return std::make_shared<InvertIndexParams>(enable_range_optimization_,
                                               enable_extended_wildcard_);
  }

  std::string to_string() const override;

  bool operator==(const IndexParams &other) const override {
    if (type() != other.type()) {
      return false;
    }
    auto &other_invert = dynamic_cast<const InvertIndexParams &>(other);
    return enable_range_optimization_ ==
               other_invert.enable_range_optimization_ &&
           enable_extended_wildcard_ == other_invert.enable_extended_wildcard_;
  }

  bool enable_range_optimization() const {
    return enable_range_optimization_;
  }

  void set_enable_range_optimization(bool enable_range_optimization) {
    enable_range_optimization_ = enable_range_optimization;
  }

  bool enable_extended_wildcard() const {
    return enable_extended_wildcard_;
  }

  // Enables suffix and infix search.
  // Note that prefix search is always enabled regardless of this setting.
  void set_enable_extended_wildcard(bool enable_extended_wildcard) {
    enable_extended_wildcard_ = enable_extended_wildcard;
  }

 private:
  bool enable_range_optimization_{false};
  bool enable_extended_wildcard_{false};
};

/*
 * Column index params
 */
class VectorIndexParams : public IndexParams {
 public:
  VectorIndexParams(IndexType type, MetricType metric_type,
                    QuantizeType quantize_type = QuantizeType::UNDEFINED)
      : IndexParams(type),
        metric_type_(metric_type),
        quantize_type_(quantize_type) {}

  virtual ~VectorIndexParams() = default;

  std::string vector_index_params_to_string(const std::string &class_name,
                                            MetricType metric_type,
                                            QuantizeType quantize_type) const;

  MetricType metric_type() const {
    return metric_type_;
  }

  void set_metric_type(MetricType metric_type) {
    metric_type_ = metric_type;
  }

  QuantizeType quantize_type() const {
    return quantize_type_;
  }

  void set_quantize_type(QuantizeType quantize_type) {
    quantize_type_ = quantize_type;
  }

 protected:
  MetricType metric_type_;
  QuantizeType quantize_type_;
};

/*
 * Vector: Hnsw index params
 */
class HnswIndexParams : public VectorIndexParams {
 public:
  HnswIndexParams(
      MetricType metric_type, int m = core_interface::kDefaultHnswNeighborCnt,
      int ef_construction = core_interface::kDefaultHnswEfConstruction,
      QuantizeType quantize_type = QuantizeType::UNDEFINED)
      : VectorIndexParams(IndexType::HNSW, metric_type, quantize_type),
        m_(m),
        ef_construction_(ef_construction) {}

  using OPtr = std::shared_ptr<HnswIndexParams>;

 public:
  Ptr clone() const override {
    return std::make_shared<HnswIndexParams>(metric_type_, m_, ef_construction_,
                                             quantize_type_);
  }

  std::string to_string() const override {
    auto base_str = vector_index_params_to_string("HnswIndexParams",
                                                  metric_type_, quantize_type_);
    std::ostringstream oss;
    oss << base_str << ",m:" << m_ << ",ef_construction:" << ef_construction_
        << "}";
    return oss.str();
  }

  bool operator==(const IndexParams &other) const override {
    return type() == other.type() &&
           metric_type() ==
               static_cast<const HnswIndexParams &>(other).metric_type() &&
           m_ == static_cast<const HnswIndexParams &>(other).m_ &&
           ef_construction_ ==
               static_cast<const HnswIndexParams &>(other).ef_construction_ &&
           quantize_type() ==
               static_cast<const HnswIndexParams &>(other).quantize_type();
  }

  void set_m(int m) {
    m_ = m;
  }
  int m() const {
    return m_;
  }
  void set_ef_construction(int ef_construction) {
    ef_construction_ = ef_construction;
  }
  int ef_construction() const {
    return ef_construction_;
  }

 private:
  int m_;
  int ef_construction_;
};

class FlatIndexParams : public VectorIndexParams {
 public:
  FlatIndexParams(MetricType metric_type,
                  QuantizeType quantize_type = QuantizeType::UNDEFINED)
      : VectorIndexParams(IndexType::FLAT, metric_type, quantize_type) {}

  using OPtr = std::shared_ptr<FlatIndexParams>;

 public:
  Ptr clone() const override {
    return std::make_shared<FlatIndexParams>(metric_type_, quantize_type_);
  }

  std::string to_string() const override {
    auto base_str = vector_index_params_to_string("FlatIndexParams",
                                                  metric_type_, quantize_type_);
    std::ostringstream oss;
    oss << base_str << "}";
    return oss.str();
  }

  bool operator==(const IndexParams &other) const override {
    return type() == other.type() &&
           metric_type() ==
               static_cast<const VectorIndexParams &>(other).metric_type() &&
           quantize_type() ==
               static_cast<const VectorIndexParams &>(other).quantize_type();
  }
};

// define default index params
const FlatIndexParams DefaultVectorIndexParams(MetricType::IP);

inline FlatIndexParams MakeDefaultVectorIndexParams(MetricType metric_type) {
  return FlatIndexParams(metric_type);
}

inline FlatIndexParams MakeDefaultQuantVectorIndexParams(
    MetricType metric_type, QuantizeType quantize_type) {
  return FlatIndexParams(metric_type, quantize_type);
}

class IVFIndexParams : public VectorIndexParams {
 public:
  IVFIndexParams(MetricType metric_type, int n_list = 1024, int n_iters = 10,
                 bool use_soar = false,
                 QuantizeType quantize_type = QuantizeType::UNDEFINED)
      : VectorIndexParams(IndexType::IVF, metric_type, quantize_type),
        n_list_(n_list),
        n_iters_(n_iters),
        use_soar_(use_soar) {}

  using OPtr = std::shared_ptr<IVFIndexParams>;

 public:
  Ptr clone() const override {
    return std::make_shared<IVFIndexParams>(metric_type_, n_list_, n_iters_,
                                            use_soar_, quantize_type_);
  }

  std::string to_string() const override {
    auto base_str = vector_index_params_to_string("IVFIndexParams",
                                                  metric_type_, quantize_type_);
    std::ostringstream oss;
    oss << base_str << ",n_list:" << n_list_ << ",n_iters:" << n_iters_ << "}";
    return oss.str();
  }

  int n_list() const {
    return n_list_;
  }

  void set_n_list(int n_list) {
    n_list_ = n_list;
  }

  int n_iters() const {
    return n_iters_;
  }

  void set_n_iters(int n_iters) {
    n_iters_ = n_iters;
  }

  bool use_soar() const {
    return use_soar_;
  }

  void set_use_soar(bool use_soar) {
    use_soar_ = use_soar;
  }

  bool operator==(const IndexParams &other) const override {
    return type() == other.type() &&
           metric_type() ==
               static_cast<const IVFIndexParams &>(other).metric_type() &&
           n_list_ == static_cast<const IVFIndexParams &>(other).n_list_ &&
           n_iters_ == static_cast<const IVFIndexParams &>(other).n_iters_ &&
           use_soar_ == static_cast<const IVFIndexParams &>(other).use_soar_ &&
           quantize_type() ==
               static_cast<const IVFIndexParams &>(other).quantize_type();
  }

 private:
  int n_list_;
  int n_iters_;
  bool use_soar_;
};

/*
 * Vector: Omega index params
 */
class OmegaIndexParams : public VectorIndexParams {
 public:
  OmegaIndexParams(
      MetricType metric_type, int m = core_interface::kDefaultHnswNeighborCnt,
      int ef_construction = core_interface::kDefaultHnswEfConstruction,
      QuantizeType quantize_type = QuantizeType::UNDEFINED,
      uint32_t min_vector_threshold = 100000,
      size_t num_training_queries = 1000,
      int ef_training = 1000,
      int window_size = 100,
      int ef_groundtruth = 0,
      int k_train = 1)  // 0 means use brute force, >0 means use HNSW with this ef
      : VectorIndexParams(IndexType::OMEGA, metric_type, quantize_type),
        m_(m),
        ef_construction_(ef_construction),
        min_vector_threshold_(min_vector_threshold),
        num_training_queries_(num_training_queries),
        ef_training_(ef_training),
        window_size_(window_size),
        ef_groundtruth_(ef_groundtruth),
        k_train_(k_train) {}

  using OPtr = std::shared_ptr<OmegaIndexParams>;

 public:
  Ptr clone() const override {
    return std::make_shared<OmegaIndexParams>(metric_type_, m_, ef_construction_,
                                             quantize_type_, min_vector_threshold_,
                                             num_training_queries_, ef_training_,
                                             window_size_, ef_groundtruth_,
                                             k_train_);
  }

  std::string to_string() const override {
    auto base_str = vector_index_params_to_string("OmegaIndexParams",
                                                  metric_type_, quantize_type_);
    std::ostringstream oss;
    oss << base_str << ",m:" << m_ << ",ef_construction:" << ef_construction_
        << ",min_vector_threshold:" << min_vector_threshold_
        << ",num_training_queries:" << num_training_queries_
        << ",ef_training:" << ef_training_
        << ",window_size:" << window_size_
        << ",ef_groundtruth:" << ef_groundtruth_
        << ",k_train:" << k_train_ << "}";
    return oss.str();
  }

  bool operator==(const IndexParams &other) const override {
    return type() == other.type() &&
           metric_type() ==
               static_cast<const OmegaIndexParams &>(other).metric_type() &&
           m_ == static_cast<const OmegaIndexParams &>(other).m_ &&
           ef_construction_ ==
               static_cast<const OmegaIndexParams &>(other).ef_construction_ &&
           min_vector_threshold_ ==
               static_cast<const OmegaIndexParams &>(other).min_vector_threshold_ &&
           num_training_queries_ ==
               static_cast<const OmegaIndexParams &>(other).num_training_queries_ &&
           ef_training_ ==
               static_cast<const OmegaIndexParams &>(other).ef_training_ &&
           window_size_ ==
               static_cast<const OmegaIndexParams &>(other).window_size_ &&
           ef_groundtruth_ ==
               static_cast<const OmegaIndexParams &>(other).ef_groundtruth_ &&
           k_train_ ==
               static_cast<const OmegaIndexParams &>(other).k_train_ &&
           quantize_type() ==
               static_cast<const OmegaIndexParams &>(other).quantize_type();
  }

  void set_m(int m) {
    m_ = m;
  }
  int m() const {
    return m_;
  }
  void set_ef_construction(int ef_construction) {
    ef_construction_ = ef_construction;
  }
  int ef_construction() const {
    return ef_construction_;
  }
  void set_min_vector_threshold(uint32_t min_vector_threshold) {
    min_vector_threshold_ = min_vector_threshold;
  }
  uint32_t min_vector_threshold() const {
    return min_vector_threshold_;
  }
  void set_num_training_queries(size_t num_training_queries) {
    num_training_queries_ = num_training_queries;
  }
  size_t num_training_queries() const {
    return num_training_queries_;
  }
  void set_ef_training(int ef_training) {
    ef_training_ = ef_training;
  }
  int ef_training() const {
    return ef_training_;
  }
  void set_window_size(int window_size) {
    window_size_ = window_size;
  }
  int window_size() const {
    return window_size_;
  }
  void set_ef_groundtruth(int ef_groundtruth) {
    ef_groundtruth_ = ef_groundtruth;
  }
  int ef_groundtruth() const {
    return ef_groundtruth_;
  }
  void set_k_train(int k_train) {
    k_train_ = k_train;
  }
  int k_train() const {
    return k_train_;
  }

 private:
  int m_;
  int ef_construction_;
  uint32_t min_vector_threshold_;
  size_t num_training_queries_;
  int ef_training_;
  int window_size_;
  int ef_groundtruth_;  // 0 = brute force, >0 = use HNSW with this ef
  int k_train_;
};

}  // namespace zvec
