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
#include "ivf_builder.h"
#include <ailego/pattern/defer.h>
#include <zvec/ailego/utility/string_helper.h>
#include "algorithm/cluster/cluster_params.h"
#include "ivf_dumper.h"

namespace zvec {
namespace core {

/*! IndexHolder support filtered by vector labels
 */
class LabelFilteredIndexHolder : public IndexHolder {
 public:
  /*! Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const IVFBuilder::RandomAccessIndexHolder::Pointer &holder,
             const std::vector<uint32_t> *elems)
        : holder_(holder), elems_(elems) {}

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    virtual const void *data(void) const override {
      return holder_->element((*elems_)[index_]);
    }

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return index_ < elems_->size();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return (*elems_)[index_];
    }

    //! Next iterator
    virtual void next(void) override {
      ++index_;
    }

   private:
    //! Members
    const IVFBuilder::RandomAccessIndexHolder::Pointer holder_{nullptr};
    const std::vector<uint32_t> *elems_{nullptr};
    size_t index_{0};
  };

  //! Constructor
  LabelFilteredIndexHolder(
      const IVFBuilder::RandomAccessIndexHolder::Pointer &holder,
      const std::vector<uint32_t> &items)
      : holder_(holder), elems_(&items) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  virtual size_t count(void) const override {
    return elems_->size();
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const override {
    return holder_->dimension();
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const override {
    return holder_->data_type();
  }

  //! Retrieve element size in bytes
  virtual size_t element_size(void) const override {
    return holder_->element_size();
  }

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  virtual IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new LabelFilteredIndexHolder::Iterator(holder_, elems_));
  }

 private:
  //! Members
  const IVFBuilder::RandomAccessIndexHolder::Pointer holder_{};
  const std::vector<uint32_t> *elems_{};
};

IVFBuilder::IVFBuilder() {}

IVFBuilder::~IVFBuilder() {
  this->cleanup();
}

int IVFBuilder::init(const IndexMeta &meta, const ailego::Params &params) {
  LOG_INFO("Begin IVFBuilder::init!");

  if (state_ != INIT) {
    LOG_ERROR("IVFBuilder state wrong. state=%d", state_);
    return IndexError_Logic;
  }

  meta_ = meta;
  converted_meta_ = meta;
  quantized_meta_ = meta;
  // Clear the converter/reformer params for external transforms
  converted_meta_.set_reformer(std::string(), 0, ailego::Params());
  converted_meta_.set_converter(std::string(), 0, ailego::Params());
  quantized_meta_.set_reformer(std::string(), 0, ailego::Params());
  quantized_meta_.set_converter(std::string(), 0, ailego::Params());
  params_ = params;

  if (!IndexFactory::HasMetric(meta_.metric_name())) {
    LOG_ERROR("Metric %s not exist", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }

  int ret = parse_centroids_num(params);
  ivf_check_with_msg(ret, "Failed to parse centroids, ret=%d", ret);

  ret = parse_clustering_params(params);
  ivf_check_with_msg(ret, "Failed to parse clustering params, ret=%d", ret);

  ret = parse_general_params(params);
  ivf_check_with_msg(ret, "Failed to parse general params, ret=%d", ret);

  LOG_INFO("End IVFBuilder::init!");

  LOG_DEBUG(
      "Converter=%s Quantizer=%s Optimizer=%s "
      "OptimizerQuantizer=%s QuantizeByCentroid=%u StoreFeatures=%u "
      "ClusterClass=%s TrainSamplesCount=%u TrainSampleRatio=%f "
      "BlockVectorCount=%u",
      params.get_as_string(PARAM_IVF_BUILDER_CONVERTER_CLASS).c_str(),
      params.get_as_string(PARAM_IVF_BUILDER_QUANTIZER_CLASS).c_str(),
      params.get_as_string(PARAM_IVF_BUILDER_OPTIMIZER_CLASS).c_str(),
      params.get_as_string(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS).c_str(),
      params.get_as_bool(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID),
      params.get_as_bool(PARAM_IVF_BUILDER_STORE_ORIGINAL_FEATURES),
      params.get_as_string(PARAM_IVF_BUILDER_CLUSTER_CLASS).c_str(),
      params.get_as_uint32(PARAM_IVF_BUILDER_TRAIN_SAMPLE_COUNT),
      params.get_as_float(PARAM_IVF_BUILDER_TRAIN_SAMPLE_RATIO),
      block_vector_count_);

  state_ = INITED;
  return 0;
}

int IVFBuilder::cleanup(void) {
  LOG_INFO("Begin IVFBuilder::cleanup");

  state_ = INIT;
  stats_.clear_attributes();
  stats_.set_built_costtime(0u);
  stats_.set_built_count(0u);
  stats_.set_discarded_count(0u);
  stats_.set_dumped_costtime(0u);
  stats_.set_dumped_count(0u);
  stats_.set_trained_costtime(0u);
  stats_.set_trained_count(0u);

  centroid_num_vec_.clear();
  cluster_class_.clear();
  converter_class_.clear();
  cluster_params_.clear();

  labels_.clear();
  centroid_index_.reset();
  holder_.reset();
  converted_meta_ = meta_;
  converter_.reset();
  quantized_meta_ = meta_;
  quantizers_.clear();

  error_ = false;
  err_code_ = 0;

  thread_count_ = 0;
  sample_count_ = 0;
  cluster_auto_tuning_ = false;
  store_original_features_ = false;
  quantize_by_centroid_ = false;

  LOG_INFO("End IVFBuilder::cleanup");

  return 0;
}

int IVFBuilder::train(IndexThreads::Pointer threads,
                      IndexHolder::Pointer holder) {
  LOG_INFO("Begin IVFBuilder::train with holder");
  if (state_ != INITED) {
    LOG_ERROR("IVFBuilder train failed, wrong state=%d", state_);
    return IndexError_Runtime;
  }

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  ailego::ElapsedTime timer;
  if (!holder || holder->count() == 0) {
    LOG_ERROR("Input holder is nullptr or empty while train index");
    return IndexError_InvalidArgument;
  }
  if (!holder->is_matched(meta_)) {
    LOG_ERROR("Input holder doesn't match index meta while train index");
    return IndexError_Mismatch;
  }

  if (converter_) {
    int ret = IndexConverter::TrainAndTransform(converter_, std::move(holder));
    ivf_check_with_msg(ret, "Failed to train or transform by converter %s",
                       converter_->name().c_str());
    converted_meta_ = converter_->meta();
    holder = converter_->result();
  }

  ailego::Params train_params;
  int ret = prepare_trainer_params(train_params);
  ivf_check_with_msg(ret, "Failed to prepare trainer params, ret=%d", ret);

  IndexTrainer::Pointer trainer =
      IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ivf_assert_with_msg(trainer, IndexError_NoExist, "Failed to create trainer");

  ret = trainer->init(converted_meta_, train_params);
  ivf_check_with_msg(ret, "Trainer init failed with ret %d", ret);

  ret = trainer->train(std::move(threads), std::move(holder));
  ivf_check_with_msg(ret, "Trainer train failed with ret %d", ret);

  ret = this->train(trainer);
  ivf_check_error_code(ret);

  stats_.set_trained_costtime(timer.milli_seconds());

  LOG_INFO("End IVFBuilder::train with holder");

  state_ = TRAINED;
  return 0;
}

int IVFBuilder::train(const IndexTrainer::Pointer &trainer) {
  LOG_DEBUG("Begin IVFBuilder::train by trainer");
  ailego::ElapsedTime timer;

  if (state_ != INITED) {
    LOG_ERROR("IVFBuilder train failed, wrong state=%d", state_);
    return IndexError_Runtime;
  }

  if (!trainer) {
    LOG_ERROR("Input trainer is nullptr while train index");
    return IndexError_InvalidArgument;
  }

  IndexCluster::CentroidList centroid_list;
  IndexBundle::Pointer boundle = trainer->indexes();
  int ret = IndexCluster::Deserialize(trainer->meta(), boundle, &centroid_list);
  ivf_check_with_msg(ret, "Failed to deserialize index");

  const IndexMeta &meta = trainer->meta();
  if (meta.data_type() != converted_meta_.data_type() ||
      meta.metric_name().compare(converted_meta_.metric_name()) != 0 ||
      meta.element_size() != converted_meta_.element_size()) {
    if (meta.converter_name() != converter_class_) {
      LOG_ERROR("Input trainer doesn't match index meta while train index");
      return IndexError_Mismatch;
    }
    //! Create converter from trainer params
    LOG_INFO("Train IVFBuilder by trainer with converter");
    converter_ = CreateAndInitConverter(meta_, meta.converter_name(),
                                        meta.converter_params());
    ivf_assert(converter_, IndexError_Runtime);
    converted_meta_ = meta;
  }

  centroid_index_ = std::make_shared<IVFCentroidIndex>();
  if (!centroid_index_) {
    return IndexError_NoMemory;
  }
  ret = centroid_index_->init(converted_meta_, params_);
  ivf_check_error_code(ret);

  ret = centroid_index_->build(centroid_list);
  ivf_check_with_msg(ret, "Failed to build centroid index");

  if (params_.has(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS)) {
    //! Quantize the centroids for searcher
    searcher_centroid_index_ = std::make_shared<IVFCentroidIndex>();
    if (!searcher_centroid_index_) {
      return IndexError_NoMemory;
    }
    ailego::Params params;
    params_.get(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_PARAMS, &params);
    searcher_centroid_index_->set_quantizer(
        params_.get_as_string(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS),
        params);
    ret = searcher_centroid_index_->init(converted_meta_, params_);
    ivf_check_error_code(ret);

    ret = searcher_centroid_index_->build(centroid_list);
    ivf_check_with_msg(ret, "Failed to build centroid index");
  }

  stats_.set_trained_costtime(timer.milli_seconds());

  LOG_DEBUG("End IVFBuilder::train by trainer");

  state_ = TRAINED;
  return 0;
}

int IVFBuilder::build(IndexThreads::Pointer threads,
                      IndexHolder::Pointer holder) {
  LOG_INFO("Begin IVFBuilder::build!");

  if (state_ != TRAINED) {
    LOG_ERROR("Train the index first before build");
    return IndexError_Runtime;
  }

  ailego::ElapsedTime timer;
  if (!holder || holder->count() == 0) {
    LOG_ERROR("Input holder is nullptr or empty while building index");
    return IndexError_InvalidArgument;
  }

  if (!holder->is_matched(meta_)) {
    LOG_ERROR("Input holder doesn't match index meta while building index");
    return IndexError_Mismatch;
  }
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }

  holder_ = std::make_shared<RandomAccessIndexHolder>(meta_);
  if (!holder_) {
    return IndexError_NoMemory;
  }
  if (holder->count() > 0) {
    holder_->reserve(holder->count());
  }
  for (auto iter = holder->create_iterator(); iter && iter->is_valid();
       iter->next()) {
    holder_->emplace(iter->key(), iter->data());
  }

  // Holder is not needed, cleanup it.
  holder.reset();

  IndexHolder::Pointer converted_holder = holder_;
  if (converter_) {
    int ret = converter_->transform(holder_);
    ivf_check_with_msg(ret, "Failed to transform by converter %s",
                       converter_->name().c_str());
    converted_holder = converter_->result();
  }

  labels_.resize(centroid_index_->centroids_count());
  int ret = this->build_label_index(threads.get(), converted_holder);
  ivf_check_with_msg(ret, "Failed to build index for %s",
                     IndexError::What(ret));

  ret = this->prepare_quantizer(threads.get());
  ivf_check_error_code(ret);

  stats_.set_built_costtime(timer.milli_seconds());

  LOG_INFO("End IVFBuilder::build");

  state_ = BUILT;
  return 0;
}

int IVFBuilder::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Begin IVFBuilder::dump");

  if (state_ != BUILT) {
    LOG_ERROR("Build the index before dump QC Index");
    return IndexError_Runtime;
  }

  ailego::ElapsedTime timer;
  int ret = this->dump_index(dumper);
  ivf_check_with_msg(ret, "Failed to dump index with ret=%d", ret);

  // the fitting function for the follow points: 1000000(0.02) 10000000(0.01)
  // 50000000(0.005) 100000000(0.001)
  float scan_ratio = -0.004 * std::log(holder_->count()) + 0.0751;
  scan_ratio = std::max(scan_ratio, 0.0001f);

  // Set Searcher Params
  ailego::Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, scan_ratio);
  meta_.set_searcher("IVFSearcher", 0, std::move(params));
  meta_.set_builder("IVFBuilder", 0, std::move(params_));

  ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  stats_.set_discarded_count(stats_.built_count() - stats_.dumped_count());
  stats_.set_dumped_costtime(timer.milli_seconds());

  LOG_INFO("End IVFBuilder::dump");

  return 0;
}

int IVFBuilder::CheckAndUpdateMajorOrder(IndexMeta &meta) {
  const std::string &metric_name = meta.metric_name();
  auto metric = IndexFactory::CreateMetric(metric_name);
  if (!metric) {
    LOG_ERROR("CreateMetric %s failed", metric_name.c_str());
    return IndexError_InvalidArgument;
  }
  int ret = metric->init(meta, meta.metric_params());
  ivf_check_with_msg(ret, "IndexMetric %s init failed", metric_name.c_str());

  bool support_column_major = true;
  for (size_t m = 32; m != 0; m /= 2) {
    for (size_t n = m; n != 0; n /= 2) {
      if (metric->distance_matrix(m, n) == nullptr) {
        support_column_major = false;
        break;
      }
    }
    if (!support_column_major) {
      break;
    }
  }
  support_column_major &=
      meta.element_size() % IndexMeta::AlignSizeof(meta.data_type()) == 0;

  if (meta.major_order() == IndexMeta::MO_UNDEFINED) {
    if (support_column_major && meta.dimension() <= 512) {
      meta.set_major_order(IndexMeta::MO_COLUMN);
    } else {
      meta.set_major_order(IndexMeta::MO_ROW);
    }
  } else {
    if (!support_column_major && meta.major_order() == IndexMeta::MO_COLUMN) {
      LOG_WARN(
          "Index Metric %s Unsupported "
          "Column Major Order",
          metric_name.c_str());
      return IndexError_Unsupported;
    }
  }

  if (block_vector_count_ * quantized_meta_.element_size() % 32 != 0) {
    LOG_ERROR(
        "block_vector_count * quantized_element_size not align with 32 bytes.");
    return IndexError_InvalidArgument;
  }

  return 0;
}

int IVFBuilder::parse_centroids_num(const ailego::Params &params) {
  std::string centroids_num =
      params.get_as_string(PARAM_IVF_BUILDER_CENTROID_COUNT);
  if (centroids_num.empty()) {
    LOG_ERROR("Param %s is required", PARAM_IVF_BUILDER_CENTROID_COUNT.c_str());
    return IndexError_InvalidArgument;
  }

  std::vector<std::string> centroid_str_vec;
  ailego::StringHelper::Split(centroids_num, CENTROID_SEPERATOR,
                              &centroid_str_vec);
  size_t level_cnt = centroid_str_vec.size();
  if ((level_cnt <= 0) || (level_cnt > 2)) {
    LOG_ERROR("Centroids level count must be [1,2]");
    return IndexError_InvalidArgument;
  }

  for (size_t idx = 0; idx < level_cnt; ++idx) {
    uint32_t centroid_cnt = 0;
    if (!ailego::StringHelper::ToUint32(centroid_str_vec[idx], &centroid_cnt)) {
      LOG_ERROR("Invalid centroids count %s", centroid_str_vec[idx].c_str());
      return IndexError_InvalidArgument;
    }
    centroid_num_vec_.push_back(centroid_cnt);
  }

  return 0;
}

int IVFBuilder::parse_clustering_params(const ailego::Params &params) {
  params.get(PARAM_IVF_BUILDER_CLUSTER_AUTO_TUNING, &cluster_auto_tuning_);

  cluster_class_ = params.get_as_string(PARAM_IVF_BUILDER_CLUSTER_CLASS);
  if (cluster_class_.empty()) {
    // OptKmeansCluster does not support custom metric
    cluster_class_ = meta_.metric_name() == kMipsMetricName
                         ? "KmeansCluster"
                         : "OptKmeansCluster";
    LOG_INFO("Using [%s] as default cluster class", cluster_class_.c_str());
  }
  for (size_t i = 1; i <= centroid_num_vec_.size(); ++i) {
    std::string level_params_key =
        PARAM_IVF_BUILDER_CLUSTER_PARAMS_IN_LEVEL_PREFIX + std::to_string(i);
    ailego::Params level_params;
    params.get<ailego::Params>(level_params_key, &level_params);
    cluster_params_.push_back(level_params);
  }

  return 0;
}

int IVFBuilder::parse_general_params(const ailego::Params &params) {
  thread_count_ = params.get_as_uint32(PARAM_IVF_BUILDER_THREAD_COUNT);
  sample_count_ = params.get_as_uint32(PARAM_IVF_BUILDER_TRAIN_SAMPLE_COUNT);
  sample_ratio_ = params.get_as_float(PARAM_IVF_BUILDER_TRAIN_SAMPLE_RATIO);

  params.get(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, &quantize_by_centroid_);
  params.get(PARAM_IVF_BUILDER_STORE_ORIGINAL_FEATURES,
             &store_original_features_);

  //! Prepare Converter for training
  if (meta_.metric_name() == kIPMetricName) {
    converter_class_ = kMipsConverterName;
  }
  params.get(PARAM_IVF_BUILDER_CONVERTER_CLASS, &converter_class_);
  if (!converter_class_.empty()) {
    ailego::Params converter_params;
    params_.get(PARAM_IVF_BUILDER_CONVERTER_PARAMS, &converter_params);
    converter_ =
        CreateAndInitConverter(meta_, converter_class_, converter_params);
    ivf_assert(converter_, IndexError_NoExist);
  }

  params_.get(PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT, &block_vector_count_);
  if (block_vector_count_ == 0) {
    block_vector_count_ = kDefaultBlockCount;
  }
  if (block_vector_count_ > kDefaultBlockCount ||
      block_vector_count_ & (block_vector_count_ - 1)) {
    LOG_ERROR("block_vector_count only can be [1|2|4|8|16|32].");
    return IndexError_InvalidArgument;
  }
  if (block_vector_count_ * meta_.element_size() % 32 != 0) {
    LOG_ERROR("block_vector_count * element_size not align with 32 bytes.");
    return IndexError_InvalidArgument;
  }
  return 0;
}

int IVFBuilder::prepare_trainer_params(ailego::Params &params) {
  params.set(STRATIFIED_TRAINER_SAMPLE_COUNT, sample_count_);
  params.set(STRATIFIED_TRAINER_SAMPLE_RATIO, sample_ratio_);
  params.set(STRATIFIED_TRAINER_THREAD_COUNT, thread_count_);
  params.set(STRATIFIED_TRAINER_AUTOAUNE, cluster_auto_tuning_);
  if (centroid_num_vec_.empty()) {
    LOG_ERROR("Centroids no specified.");
    return IndexError_InvalidArgument;
  }
  std::string cluster_count = std::to_string(centroid_num_vec_[0]);
  if (centroid_num_vec_.size() > 1) {
    cluster_count +=
        (CENTROID_SEPERATOR + std::to_string(centroid_num_vec_[1]));
  }
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, cluster_count);

  for (size_t i = 1; i <= cluster_params_.size(); ++i) {
    std::string level_params_key =
        STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + std::to_string(i);
    params.set(level_params_key, cluster_params_[i - 1]);
  }
  params.set(STRATIFIED_TRAINER_CLASS_NAME, cluster_class_);

  return 0;
}

int IVFBuilder::build_label_index(IndexThreads *threads,
                                  const IndexHolder::Pointer &holder) {
  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  size_t id = 0UL;
  AILEGO_DEFER([&]() {
    task_group->wait_finish();
    stats_.set_built_count(id);
    LOG_INFO("Finished building, total=%zu", id);
  });

  size_t elem_size = holder->element_size();
  std::shared_ptr<VectorList> vectors = std::make_shared<VectorList>();
  ivf_assert(vectors, IndexError_NoMemory);
  for (; iter && iter->is_valid(); iter->next()) {
    ivf_assert(!error_, err_code_);
    vectors->emplace_back(iter->data(), elem_size, id);
    id++;
    if (vectors->size() == kBatchSize || id == holder_->count()) {
      auto task = ailego::Closure ::New(const_cast<IVFBuilder *>(this),
                                        &IVFBuilder::label, vectors);
      task_group->submit(std::move(task));
      vectors = std::make_shared<VectorList>();
      ivf_assert(vectors, IndexError_NoMemory);
      vectors->reserve(kBatchSize);
    }
    if (!(id & 0xFFFFF)) {
      LOG_INFO("Current built count:%zu", id);
    }
  }
  ailego_assert_with(vectors->size() == 0, "invalid size");

  return err_code_;
}

int IVFBuilder::dump_index(const IndexDumper::Pointer &dumper) {
  int ret = CheckAndUpdateMajorOrder(quantized_meta_);
  ivf_check_error_code(ret);

  IVFDumper::Pointer ivf_dumper = std::make_shared<IVFDumper>(
      quantized_meta_, dumper, centroid_index_->centroids_count(),
      block_vector_count_);
  if (!ivf_dumper) {
    LOG_ERROR("Alloc IVFDumper failed");
    return IndexError_NoMemory;
  }

  //! Dump inverted vectors
  std::vector<uint32_t> dumped_ids;
  std::function<void(uint32_t)> record_dumped_id = [&](uint32_t) {};
  if (store_original_features_) {
    dumped_ids.reserve(holder_->count());
    record_dumped_id = [&](uint32_t id) { dumped_ids.emplace_back(id); };
  }
  if (quantizers_.size() == 0) {
    //! No quantizer for inverted vectors
    for (size_t i = 0; i < centroid_index_->centroids_count(); ++i) {
      ailego_assert_with(i < labels_.size(), "Index Overflow");
      for (size_t j = 0; j < labels_[i].size(); ++j) {
        auto id = labels_[i][j];
        record_dumped_id(id);
        ret = ivf_dumper->dump_inverted_vector(i, holder_->key(id),
                                               holder_->element(id));
        ivf_check_error_code(ret);
      }
    }
  } else {
    for (size_t i = 0; i < centroid_index_->centroids_count(); ++i) {
      ailego_assert_with(i < labels_.size(), "Index Overflow");
      auto holder =
          std::make_shared<LabelFilteredIndexHolder>(holder_, labels_[i]);
      if (!holder) {
        return IndexError_NoMemory;
      }
      auto quantizer = quantize_by_centroid_ ? quantizers_[i] : quantizers_[0];
      ret = quantizer->transform(holder);
      ivf_check_error_code(ret);

      auto iter = quantizer->result()->create_iterator();
      for (; iter->is_valid(); iter->next()) {
        uint32_t id = iter->key();
        record_dumped_id(id);
        ret =
            ivf_dumper->dump_inverted_vector(i, holder_->key(id), iter->data());
        ivf_check_error_code(ret);
      }
    }
  }

  ret = ivf_dumper->dump_inverted_vector_finished();
  ivf_check_error_code(ret);

  ret = ivf_dumper->dump_quantizer_params(quantizers_);
  ivf_check_error_code(ret);

  auto centroid_index =
      searcher_centroid_index_ ? searcher_centroid_index_ : centroid_index_;
  ret = ivf_dumper->dump_centroid_index(centroid_index->data(),
                                        centroid_index->size());
  ivf_check_with_msg(ret, "Failed to dump CentroidIndex");

  if (store_original_features_) {
    for (size_t i = 0; i < dumped_ids.size(); ++i) {
      ret = ivf_dumper->dump_original_vector(holder_->element(dumped_ids[i]),
                                             holder_->element_size());
      ivf_check_error_code(ret);
    }
  }

  stats_.set_dumped_count(stats_.dumped_count() + ivf_dumper->dumped_count());

  return 0;
}

int IVFBuilder::prepare_quantizer(IndexThreads *threads) {
  std::string quantizer_name;
  params_.get(PARAM_IVF_BUILDER_QUANTIZER_CLASS, &quantizer_name);
  if (quantizer_name.empty()) {
    return 0;
  }

  //! Prepare Quantizers for inverted index
  ailego::Params quantizer_params;
  params_.get(PARAM_IVF_BUILDER_QUANTIZER_PARAMS, &quantizer_params);
  if (((quantizer_name != kInt8QuantizerName &&
        quantizer_name != kInt4QuantizerName) ||
       meta_.metric_name() != kIPMetricName) &&
      quantize_by_centroid_) {
    LOG_WARN("%s is supported in InnerProduct only",
             PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID.c_str());
    quantize_by_centroid_ = false;
  }
  if (quantizer_name == kInt4QuantizerName && meta_.dimension() & 0x1) {
    LOG_ERROR("Unsupport quantizer=%s for dim=%u", kInt4QuantizerName,
              meta_.dimension());
    return IndexError_Unsupported;
  }

  int ret = 0;
  auto create_and_init_quantizer = [&]() {
    auto quantizer = IndexFactory::CreateConverter(quantizer_name);
    if (!quantizer) {
      LOG_ERROR("Failed to create converter %s", quantizer_name.c_str());
      ret = IndexError_NoExist;
      return IndexConverter::Pointer();
    }
    ret = quantizer->init(meta_, quantizer_params);
    if (ret != 0) {
      LOG_ERROR("Failed to initialize converter %s for %s",
                quantizer_name.c_str(), IndexError::What(ret));
      return IndexConverter::Pointer();
    }
    return quantizer;
  };
  for (size_t i = 0; i < centroid_index_->centroids_count(); ++i) {
    quantizers_.emplace_back(create_and_init_quantizer());
    ivf_check_error_code(ret);
    if (!quantize_by_centroid_) {
      break;
    }
  }

  //! Train the quantizers
  auto train_data = [&](size_t i) {
    IndexHolder::Pointer holder = holder_;
    size_t idx = 0;
    if (quantize_by_centroid_) {
      holder = std::make_shared<LabelFilteredIndexHolder>(holder_, labels_[i]);
      if (!holder && !error_.exchange(true)) {
        err_code_ = IndexError_NoMemory;
        return;
      }
      idx = i;
    }
    if (holder->count() == 0) {
      return;
    }
    ret = quantizers_[idx]->train(holder);
    if (ret != 0) {
      LOG_ERROR("Failed to train converter %s for %s", quantizer_name.c_str(),
                IndexError::What(ret));
      if (!error_.exchange(true)) {
        err_code_ = IndexError_Runtime;
      }
    }
  };

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  for (size_t i = 0; i < quantizers_.size(); ++i) {
    if (error_) {
      task_group->wait_finish();
      return err_code_;
    }
    task_group->submit(ailego::Closure ::New(train_data, i));
  }

  task_group->wait_finish();
  if (quantizers_.size() > 0) {
    quantized_meta_ = quantizers_[0]->meta();
  }

  return 0;
}

INDEX_FACTORY_REGISTER_BUILDER(IVFBuilder);

}  // namespace core
}  // namespace zvec
