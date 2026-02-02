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

#include <zvec/ailego/container/params.h>

namespace zvec {
namespace core {

/*! Index Meta
 */
class IndexMeta {
 public:
  /*! Meta Types
   */
  enum MetaType { MT_UNDEFINED = 0, MT_DENSE = 1, MT_SPARSE = 2 };

  /*! Data Types
   */
  enum DataType {
    DT_UNDEFINED = 0,
    DT_FP16 = 1,
    DT_FP32 = 2,
    DT_FP64 = 3,
    DT_INT8 = 4,
    DT_INT16 = 5,
    DT_INT4 = 6,
    DT_BINARY32 = 7,
    DT_BINARY64 = 8,
  };

  /*! Major Orders
   */
  enum MajorOrder {
    MO_UNDEFINED = 0,
    MO_ROW = 1,
    MO_COLUMN = 2,
  };

  //! Constructor
  IndexMeta(void) {
    this->set_meta(DataType::DT_FP32, 128u);
    this->set_metric("SquaredEuclidean", 0, ailego::Params());
  }

  //! Constructor
  IndexMeta(DataType data_type, uint32_t dim) {
    meta_type_ = MT_DENSE;
    this->set_meta(data_type, dim);
    this->set_metric("SquaredEuclidean", 0, ailego::Params());
  }

  //! Constructor
  IndexMeta(MetaType meta_type, DataType data_type) {
    meta_type_ = meta_type;

    this->set_meta(data_type, 0);
    this->set_metric("SquaredEuclidean", 0, ailego::Params());
  }

  //! Constructor
  IndexMeta(const IndexMeta &rhs)
      : meta_type_{rhs.meta_type_},
        major_order_(rhs.major_order_),
        data_type_(rhs.data_type_),
        dimension_(rhs.dimension_),
        unit_size_(rhs.unit_size_),
        element_size_(rhs.element_size_),
        space_id_(rhs.space_id_),
        metric_revision_(rhs.metric_revision_),
        converter_revision_(rhs.converter_revision_),
        reformer_revision_(rhs.reformer_revision_),
        trainer_revision_(rhs.trainer_revision_),
        builder_revision_(rhs.builder_revision_),
        reducer_revision_(rhs.reducer_revision_),
        searcher_revision_(rhs.searcher_revision_),
        streamer_revision_(rhs.streamer_revision_),
        metric_name_(rhs.metric_name_),
        converter_name_(rhs.converter_name_),
        reformer_name_(rhs.reformer_name_),
        trainer_name_(rhs.trainer_name_),
        builder_name_(rhs.builder_name_),
        reducer_name_(rhs.reducer_name_),
        searcher_name_(rhs.searcher_name_),
        streamer_name_(rhs.streamer_name_),
        metric_params_(rhs.metric_params_),
        converter_params_(rhs.converter_params_),
        reformer_params_(rhs.reformer_params_),
        trainer_params_(rhs.trainer_params_),
        builder_params_(rhs.builder_params_),
        reducer_params_(rhs.reducer_params_),
        searcher_params_(rhs.searcher_params_),
        streamer_params_(rhs.streamer_params_),
        attributes_(rhs.attributes_) {}

  //! Constructor
  IndexMeta(IndexMeta &&rhs)
      : meta_type_{rhs.meta_type_},
        major_order_(rhs.major_order_),
        data_type_(rhs.data_type_),
        dimension_(rhs.dimension_),
        unit_size_(rhs.unit_size_),
        element_size_(rhs.element_size_),
        space_id_(rhs.space_id_),
        metric_revision_(rhs.metric_revision_),
        converter_revision_(rhs.converter_revision_),
        reformer_revision_(rhs.reformer_revision_),
        trainer_revision_(rhs.trainer_revision_),
        builder_revision_(rhs.builder_revision_),
        reducer_revision_(rhs.reducer_revision_),
        searcher_revision_(rhs.searcher_revision_),
        streamer_revision_(rhs.streamer_revision_),
        metric_name_(std::move(rhs.metric_name_)),
        converter_name_(std::move(rhs.converter_name_)),
        reformer_name_(std::move(rhs.reformer_name_)),
        trainer_name_(std::move(rhs.trainer_name_)),
        builder_name_(std::move(rhs.builder_name_)),
        reducer_name_(std::move(rhs.reducer_name_)),
        searcher_name_(std::move(rhs.searcher_name_)),
        streamer_name_(std::move(rhs.streamer_name_)),
        metric_params_(std::move(rhs.metric_params_)),
        converter_params_(std::move(rhs.converter_params_)),
        reformer_params_(std::move(rhs.reformer_params_)),
        trainer_params_(std::move(rhs.trainer_params_)),
        builder_params_(std::move(rhs.builder_params_)),
        reducer_params_(std::move(rhs.reducer_params_)),
        searcher_params_(std::move(rhs.searcher_params_)),
        streamer_params_(std::move(rhs.streamer_params_)),
        attributes_(std::move(rhs.attributes_)) {}

  //! Assignment
  IndexMeta &operator=(const IndexMeta &rhs) {
    meta_type_ = rhs.meta_type_;
    major_order_ = rhs.major_order_;
    data_type_ = rhs.data_type_;
    dimension_ = rhs.dimension_;
    unit_size_ = rhs.unit_size_;
    element_size_ = rhs.element_size_;
    space_id_ = rhs.space_id_;
    metric_revision_ = rhs.metric_revision_;
    converter_revision_ = rhs.converter_revision_;
    reformer_revision_ = rhs.reformer_revision_;
    trainer_revision_ = rhs.trainer_revision_;
    builder_revision_ = rhs.builder_revision_;
    reducer_revision_ = rhs.reducer_revision_;
    searcher_revision_ = rhs.searcher_revision_;
    streamer_revision_ = rhs.streamer_revision_;
    metric_name_ = std::move(rhs.metric_name_);
    converter_name_ = std::move(rhs.converter_name_);
    reformer_name_ = std::move(rhs.reformer_name_);
    trainer_name_ = std::move(rhs.trainer_name_);
    builder_name_ = std::move(rhs.builder_name_);
    reducer_name_ = std::move(rhs.reducer_name_);
    searcher_name_ = std::move(rhs.searcher_name_);
    streamer_name_ = std::move(rhs.streamer_name_);
    metric_params_ = std::move(rhs.metric_params_);
    converter_params_ = std::move(rhs.converter_params_);
    reformer_params_ = std::move(rhs.reformer_params_);
    trainer_params_ = std::move(rhs.trainer_params_);
    builder_params_ = std::move(rhs.builder_params_);
    reducer_params_ = std::move(rhs.reducer_params_);
    searcher_params_ = std::move(rhs.searcher_params_);
    streamer_params_ = std::move(rhs.streamer_params_);
    attributes_ = std::move(rhs.attributes_);

    return *this;
  }

  //! Assignment
  IndexMeta &operator=(IndexMeta &&rhs) {
    meta_type_ = rhs.meta_type_;
    major_order_ = rhs.major_order_;
    data_type_ = rhs.data_type_;
    dimension_ = rhs.dimension_;
    unit_size_ = rhs.unit_size_;
    element_size_ = rhs.element_size_;
    space_id_ = rhs.space_id_;
    metric_revision_ = rhs.metric_revision_;
    converter_revision_ = rhs.converter_revision_;
    reformer_revision_ = rhs.reformer_revision_;
    trainer_revision_ = rhs.trainer_revision_;
    builder_revision_ = rhs.builder_revision_;
    reducer_revision_ = rhs.reducer_revision_;
    searcher_revision_ = rhs.searcher_revision_;
    streamer_revision_ = rhs.streamer_revision_;
    metric_name_ = std::move(rhs.metric_name_);
    converter_name_ = std::move(rhs.converter_name_);
    reformer_name_ = std::move(rhs.reformer_name_);
    trainer_name_ = std::move(rhs.trainer_name_);
    builder_name_ = std::move(rhs.builder_name_);
    reducer_name_ = std::move(rhs.reducer_name_);
    searcher_name_ = std::move(rhs.searcher_name_);
    streamer_name_ = std::move(rhs.streamer_name_);
    metric_params_ = std::move(rhs.metric_params_);
    converter_params_ = std::move(rhs.converter_params_);
    reformer_params_ = std::move(rhs.reformer_params_);
    trainer_params_ = std::move(rhs.trainer_params_);
    builder_params_ = std::move(rhs.builder_params_);
    reducer_params_ = std::move(rhs.reducer_params_);
    searcher_params_ = std::move(rhs.searcher_params_);
    streamer_params_ = std::move(rhs.streamer_params_);
    attributes_ = std::move(rhs.attributes_);

    return *this;
  }

  //! Reset the meta
  void clear(void) {
    meta_type_ = MetaType::MT_DENSE;
    major_order_ = MajorOrder::MO_UNDEFINED;
    data_type_ = DataType::DT_UNDEFINED;
    dimension_ = 0;
    unit_size_ = 0;
    element_size_ = 0;
    space_id_ = 0;
    metric_revision_ = 0;
    converter_revision_ = 0;
    reformer_revision_ = 0;
    trainer_revision_ = 0;
    builder_revision_ = 0;
    reducer_revision_ = 0;
    searcher_revision_ = 0;
    streamer_revision_ = 0;
    metric_name_.clear();
    converter_name_.clear();
    reformer_name_.clear();
    trainer_name_.clear();
    builder_name_.clear();
    reducer_name_.clear();
    searcher_name_.clear();
    streamer_name_.clear();
    metric_params_.clear();
    converter_params_.clear();
    reformer_params_.clear();
    trainer_params_.clear();
    builder_params_.clear();
    reducer_params_.clear();
    searcher_params_.clear();
    streamer_params_.clear();
    attributes_.clear();
  }

  //! Retrieve major order information
  MetaType meta_type(void) const {
    return meta_type_;
  }

  //! Retrieve major order information
  MajorOrder major_order(void) const {
    return major_order_;
  }

  //! Retrieve type information
  DataType data_type(void) const {
    return data_type_;
  }

  //! Retrieve dimension
  uint32_t dimension(void) const {
    return dimension_;
  }

  //! Retrieve unit size in bytes
  uint32_t unit_size(void) const {
    return unit_size_;
  }

  //! Retrieve element size in bytes
  uint32_t element_size(void) const {
    return element_size_;
  }

  //! Retrieve space id
  uint64_t space_id(void) const {
    return space_id_;
  }

  //! Retrieve revision of metric
  uint32_t metric_revision(void) const {
    return metric_revision_;
  }

  //! Retrieve revision of converter
  uint32_t converter_revision(void) const {
    return converter_revision_;
  }

  //! Retrieve revision of reformer
  uint32_t reformer_revision(void) const {
    return reformer_revision_;
  }

  //! Retrieve revision of trainer
  uint32_t trainer_revision(void) const {
    return trainer_revision_;
  }

  //! Retrieve revision of builder
  uint32_t builder_revision(void) const {
    return builder_revision_;
  }

  //! Retrieve revision of searcher
  uint32_t searcher_revision(void) const {
    return searcher_revision_;
  }

  //! Retrieve revision of reducer
  uint32_t reducer_revision(void) const {
    return reducer_revision_;
  }

  //! Retrieve revision of streamer
  uint32_t streamer_revision(void) const {
    return streamer_revision_;
  }

  //! Retrieve name of metric
  const std::string &metric_name(void) const {
    return metric_name_;
  }

  //! Retrieve name of converter
  const std::string &converter_name(void) const {
    return converter_name_;
  }

  //! Retrieve name of reformer
  const std::string &reformer_name(void) const {
    return reformer_name_;
  }

  //! Retrieve name of trainer
  const std::string &trainer_name(void) const {
    return trainer_name_;
  }

  //! Retrieve name of builder
  const std::string &builder_name(void) const {
    return builder_name_;
  }

  //! Retrieve name of reducer
  const std::string &reducer_name(void) const {
    return reducer_name_;
  }

  //! Retrieve name of searcher
  const std::string &searcher_name(void) const {
    return searcher_name_;
  }

  //! Retrieve name of streamer
  const std::string &streamer_name(void) const {
    return streamer_name_;
  }

  //! Retrieve metric params
  const ailego::Params &metric_params(void) const {
    return metric_params_;
  }

  //! Retrieve converter params
  const ailego::Params &converter_params(void) const {
    return converter_params_;
  }

  //! Retrieve reformer params
  const ailego::Params &reformer_params(void) const {
    return reformer_params_;
  }

  //! Retrieve trainer params
  const ailego::Params &trainer_params(void) const {
    return trainer_params_;
  }

  //! Retrieve builder params
  const ailego::Params &builder_params(void) const {
    return builder_params_;
  }

  //! Retrieve reducer params
  const ailego::Params &reducer_params(void) const {
    return reducer_params_;
  }

  //! Retrieve searcher params
  const ailego::Params &searcher_params(void) const {
    return searcher_params_;
  }

  //! Retrieve streamer params
  const ailego::Params &streamer_params(void) const {
    return streamer_params_;
  }

  //! Retrieve attributes
  const ailego::Params &attributes(void) const {
    return attributes_;
  }

  //! Retrieve mutable attributes
  ailego::Params *mutable_attributes(void) {
    return &attributes_;
  }

  //! Set meta type
  void set_meta_type(MetaType meta_type) {
    meta_type_ = meta_type;
  }

  //! Set major order of features
  void set_major_order(MajorOrder major_order) {
    major_order_ = major_order;
  }

  //! Set dimension of feature
  void set_dimension(uint32_t dim) {
    dimension_ = dim;
    element_size_ = IndexMeta::ElementSizeof(data_type_, unit_size_, dim);
  }

  //! Set meta information of feature
  void set_data_type(DataType data_type) {
    data_type_ = data_type;
    unit_size_ = UnitSizeof(data_type);
  }

  //! Set meta information of feature
  void set_meta(DataType data_type, uint32_t unit, uint32_t dim) {
    data_type_ = data_type;
    dimension_ = dim;
    unit_size_ = unit;
    element_size_ = ElementSizeof(data_type, unit, dim);
  }

  //! Set meta information of feature
  void set_meta(DataType data_type, uint32_t dim) {
    this->set_meta(data_type, UnitSizeof(data_type), dim);
  }

  //! Set information of metric
  template <typename TName, typename TParams>
  void set_metric(TName &&name, uint32_t rev, TParams &&params) {
    metric_name_ = std::forward<TName>(name);
    metric_revision_ = rev;
    metric_params_ = std::forward<TParams>(params);
  }

  //! Set information of converter
  template <typename TName, typename TParams>
  void set_converter(TName &&name, uint32_t rev, TParams &&params) {
    converter_name_ = std::forward<TName>(name);
    converter_revision_ = rev;
    converter_params_ = std::forward<TParams>(params);
  }

  //! Set information of reformer
  template <typename TName, typename TParams>
  void set_reformer(TName &&name, uint32_t rev, TParams &&params) {
    reformer_name_ = std::forward<TName>(name);
    reformer_revision_ = rev;
    reformer_params_ = std::forward<TParams>(params);
  }

  //! Set information of trainer
  template <typename TName, typename TParams>
  void set_trainer(TName &&name, uint32_t rev, TParams &&params) {
    trainer_name_ = std::forward<TName>(name);
    trainer_revision_ = rev;
    trainer_params_ = std::forward<TParams>(params);
  }

  //! Set information of builder
  template <typename TName, typename TParams>
  void set_builder(TName &&name, uint32_t rev, TParams &&params) {
    builder_name_ = std::forward<TName>(name);
    builder_revision_ = rev;
    builder_params_ = std::forward<TParams>(params);
  }

  //! Set information of reducer
  template <typename TName, typename TParams>
  void set_reducer(TName &&name, uint32_t rev, TParams &&params) {
    reducer_name_ = std::forward<TName>(name);
    reducer_revision_ = rev;
    reducer_params_ = std::forward<TParams>(params);
  }

  //! Set information of searcher
  template <typename TName, typename TParams>
  void set_searcher(TName &&name, uint32_t rev, TParams &&params) {
    searcher_name_ = std::forward<TName>(name);
    searcher_revision_ = rev;
    searcher_params_ = std::forward<TParams>(params);
  }

  //! Set information of streamer
  template <typename TName, typename TParams>
  void set_streamer(TName &&name, uint32_t rev, TParams &&params) {
    streamer_name_ = std::forward<TName>(name);
    streamer_revision_ = rev;
    streamer_params_ = std::forward<TParams>(params);
  }

  //! Serialize meta information into buffer
  void serialize(std::string *out) const;

  //! Derialize meta information from buffer
  bool deserialize(const void *data, size_t len);

  //! Calculate unit size of feature
  static uint32_t UnitSizeof(DataType data_type) {
    static const uint32_t unit_size_table[] = {
        0u,                // DT_UNDEFINED
        sizeof(uint16_t),  // DT_FP16
        sizeof(float),     // DT_FP32
        sizeof(double),    // DT_FP64
        sizeof(int8_t),    // DT_INT8
        sizeof(int16_t),   // DT_INT16
        sizeof(uint8_t),   // DT_INT4
        sizeof(uint32_t),  // DT_BINARY32
        sizeof(uint64_t)   // DT_BINARY64
    };
    return unit_size_table[data_type];
  }

  //! Calculate align size of feature
  static uint32_t AlignSizeof(DataType ft) {
    static const uint32_t align_size_table[] = {
        0u,                   // DT_UNDEFINED
        sizeof(uint16_t),     // DT_FP16
        sizeof(float),        // DT_FP32
        sizeof(double),       // DT_FP64
        sizeof(int8_t) * 4,   // DT_INT8
        sizeof(int16_t),      // DT_INT16
        sizeof(uint8_t) * 4,  // DT_INT4
        sizeof(uint32_t),     // DT_BINARY32
        sizeof(uint64_t)      // DT_BINARY64
    };
    return align_size_table[ft];
  }

  //! Calculate element size of feature
  static uint32_t ElementSizeof(DataType data_type, uint32_t unit,
                                uint32_t dim) {
    switch (data_type) {
      case DataType::DT_UNDEFINED:
        return 0;
      case DataType::DT_FP16:
      case DataType::DT_FP32:
      case DataType::DT_FP64:
      case DataType::DT_INT8:
      case DataType::DT_INT16:
        return (dim * unit);
      case DataType::DT_INT4:
        return (dim + unit * 2 - 1) / (unit * 2) * unit;
      case DataType::DT_BINARY32:
      case DataType::DT_BINARY64:
        return (dim + unit * 8 - 1) / (unit * 8) * unit;
    }
    return 0;
  }

  //! Calculate element size of vector
  static uint32_t ElementSizeof(DataType data_type, uint32_t dim) {
    return ElementSizeof(data_type, UnitSizeof(data_type), dim);
  }

 private:
  MetaType meta_type_{MetaType::MT_DENSE};
  MajorOrder major_order_{MajorOrder::MO_UNDEFINED};
  DataType data_type_{DataType::DT_UNDEFINED};
  uint32_t dimension_{0};
  uint32_t unit_size_{0};
  uint32_t element_size_{0};
  uint64_t space_id_{0};
  uint32_t metric_revision_{0};
  uint32_t converter_revision_{0};
  uint32_t reformer_revision_{0};
  uint32_t trainer_revision_{0};
  uint32_t builder_revision_{0};
  uint32_t reducer_revision_{0};
  uint32_t searcher_revision_{0};
  uint32_t streamer_revision_{0};

  std::string metric_name_{};
  std::string converter_name_{};
  std::string reformer_name_{};
  std::string trainer_name_{};
  std::string builder_name_{};
  std::string reducer_name_{};
  std::string searcher_name_{};
  std::string streamer_name_{};

  ailego::Params metric_params_{};
  ailego::Params converter_params_{};
  ailego::Params reformer_params_{};
  ailego::Params trainer_params_{};
  ailego::Params builder_params_{};
  ailego::Params reducer_params_{};
  ailego::Params searcher_params_{};
  ailego::Params streamer_params_{};
  ailego::Params attributes_{};
};

/*! Index Query Meta
 */
class IndexQueryMeta {
 public:
  //! Constructor
  IndexQueryMeta(void) {}

  //! Constructor
  IndexQueryMeta(IndexMeta::MetaType meta_type, IndexMeta::DataType data_type,
                 uint32_t unit, uint32_t dim)
      : meta_type_(meta_type),
        data_type_(data_type),
        dimension_(dim),
        unit_size_(unit),
        element_size_(IndexMeta::ElementSizeof(data_type, unit, dim)) {}

  //! Constructor
  IndexQueryMeta(IndexMeta::DataType data_type, uint32_t dim)
      : IndexQueryMeta{IndexMeta::MetaType::MT_DENSE, data_type,
                       IndexMeta::UnitSizeof(data_type), dim} {}

  //! Constructor
  IndexQueryMeta(IndexMeta::DataType data_type)
      : IndexQueryMeta{IndexMeta::MetaType::MT_SPARSE, data_type,
                       IndexMeta::UnitSizeof(data_type), 0} {}

  //! Constructor
  IndexQueryMeta(IndexMeta::MetaType meta_type, IndexMeta::DataType data_type,
                 uint32_t dim = 0)
      : IndexQueryMeta{meta_type, data_type, IndexMeta::UnitSizeof(data_type),
                       dim} {}

  //! Retrieve meta type
  IndexMeta::MetaType meta_type(void) const {
    return meta_type_;
  }

  //! Retrieve data
  IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  //! Retrieve dimension of features
  uint32_t dimension(void) const {
    return dimension_;
  }

  //! Retrieve unit size of feature
  uint32_t unit_size(void) const {
    return unit_size_;
  }

  //! Retrieve element size of feature
  uint32_t element_size(void) const {
    return element_size_;
  }

  //! Set dimension of feature
  void set_dimension(uint32_t dim) {
    dimension_ = dim;
    element_size_ = IndexMeta::ElementSizeof(data_type_, unit_size_, dim);
  }

  //! Set meta type
  void set_meta_type(IndexMeta::MetaType meta_type) {
    meta_type_ = meta_type;
  }

  //! Set data type
  void set_data_type(IndexMeta::DataType data_type) {
    data_type_ = data_type;
  }

  //! Set meta information of feature
  void set_meta(IndexMeta::DataType data_type, uint32_t unit, uint32_t dim) {
    data_type_ = data_type;
    dimension_ = dim;
    unit_size_ = unit;
    element_size_ = IndexMeta::ElementSizeof(data_type, unit, dim);
  }

  //! Set meta information of feature
  void set_meta(IndexMeta::DataType data_type, uint32_t dim) {
    this->set_meta(data_type, IndexMeta::UnitSizeof(data_type), dim);
  }

 private:
  IndexMeta::MetaType meta_type_{IndexMeta::MetaType::MT_DENSE};
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_UNDEFINED};
  uint32_t dimension_{0};
  uint32_t unit_size_{0};
  uint32_t element_size_{0};
};

}  // namespace core
}  // namespace zvec
