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
#include "ivf_centroid_index.h"
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_framework.h>
#include "metric/metric_params.h"

namespace zvec {
namespace core {

/*! Fake Trainer to supply centroids in bundle
 */
class FakeClusterTrainer : public IndexTrainer {
 public:
  //! Constructor
  FakeClusterTrainer(const IndexMeta &imeta, const IndexBundle::Pointer &bundle)
      : meta_(imeta), bundle_(bundle) {}

  //! Destructor
  ~FakeClusterTrainer(void) {}

 protected:
  //! Initialize Trainer
  virtual int init(const IndexMeta &, const ailego::Params &) override {
    return 0;
  }

  //! Cleanup Trainer
  virtual int cleanup(void) override {
    return 0;
  }

  //! Train the data
  virtual int train(IndexHolder::Pointer) override {
    return 0;
  }

  //! Train the data
  virtual int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    return 0;
  }

  //! Load index from file path or dir
  virtual int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Dump index into file path or dir
  virtual int dump(const IndexDumper::Pointer &) override {
    return 0;
  }

  //! Retrieve Index Meta
  virtual const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve statistics
  virtual const IndexTrainer::Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve the output indexes
  virtual IndexBundle::Pointer indexes(void) const override {
    return bundle_;
  }

 private:
  //! Members
  IndexMeta meta_{};
  Stats stats_{};
  IndexBundle::Pointer bundle_{};
};

/*! Int8QuantizerReformer for InnerProduct Measure
 */
class Int8QuantizerReformer4IP : public IndexReformer {
 public:
  //! Initialize Reformer
  virtual int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup Reformer
  virtual int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  virtual int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  virtual int unload(void) override {
    return 0;
  }

  //! Transform query
  virtual int transform(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/, std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const override {
#if 0
        size_t dim = qmeta.dimension();
        out->resize(IndexMeta::ElementSizeof(
            IndexMeta::DataType::DT_INT8, dim));
        ometa->set_meta(IndexMeta::DataType::DT_INT8, dim);
        const float *ivec = reinterpret_cast<const float *>(query);
        int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
        float abs_max = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            auto abs = std::abs(ivec[i]);
            if (abs > abs_max) {
                abs_max = abs;
            }
        }
        if (abs_max > 0.0f) {
            float scale = 127 / abs_max;
            for (size_t i = 0; i < dim; ++i) {
                ovec[i] = static_cast<int8_t>(std::round(ivec[i] * scale));
            }
        } else {
            std::fill(ovec, ovec + dim, static_cast<int8_t>(1));
        }
        return 0;
#else
    return IndexError_NotImplemented;
#endif
  }

  //! Transform queries
  virtual int transform(const void *query, const IndexQueryMeta &qmeta,
                        uint32_t count, std::string *oquery,
                        IndexQueryMeta *ometa) const override {
    size_t dim = qmeta.dimension();
    oquery->resize(count *
                   IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8, dim));
    ometa->set_meta(IndexMeta::DataType::DT_INT8, dim);
    const float *ivec = reinterpret_cast<const float *>(query);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*oquery)[0]);
    for (size_t q = 0; q < count; ++q) {
      float abs_max = 0.0f;
      const float *in = &ivec[q * dim];
      int8_t *out = &ovec[q * dim];
      for (size_t i = 0; i < dim; ++i) {
        auto abs = std::abs(in[i]);
        if (abs > abs_max) {
          abs_max = abs;
        }
      }
      if (abs_max > 0.0f) {
        float scale = 127 / abs_max;
        for (size_t i = 0; i < dim; ++i) {
          out[i] = static_cast<int8_t>(std::round(in[i] * scale));
        }
      } else {
        std::fill(out, out + dim, static_cast<int8_t>(1));
      }
    }
    return 0;
  }

  //! Normalize results
  virtual int normalize(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/,
                        IndexDocumentList & /*result*/) const override {
    return 0;
  }
};

/*! Int4QuantizerReformer for InnerProduct Metric
 */
class Int4QuantizerReformer4IP : public IndexReformer {
 public:
  //! Initialize Reformer
  virtual int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup Reformer
  virtual int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  virtual int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  virtual int unload(void) override {
    return 0;
  }

  //! Transform query
  virtual int transform(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/, std::string * /*out*/,
                        IndexQueryMeta * /*ometa*/) const override {
    return IndexError_NotImplemented;
  }

  //! Transform queries
  virtual int transform(const void *query, const IndexQueryMeta &qmeta,
                        uint32_t count, std::string *oquery,
                        IndexQueryMeta *ometa) const override {
    if (qmeta.dimension() & 0x1) {
      LOG_ERROR("Unsuuport dim=%u for transform", qmeta.dimension());
      return IndexError_Unsupported;
    }

    size_t dim = qmeta.dimension();
    oquery->resize(count *
                   IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT4, dim));
    ometa->set_meta(IndexMeta::DataType::DT_INT4, dim);
    const float *ivec = reinterpret_cast<const float *>(query);
    uint8_t *ovec = reinterpret_cast<uint8_t *>(&(*oquery)[0]);
    for (size_t q = 0; q < count; ++q) {
      float abs_max = 0.0f;
      float max = -std::numeric_limits<float>::max();
      const float *in = &ivec[q * dim];
      uint8_t *out = &ovec[q * dim / 2];
      for (size_t i = 0; i < dim; ++i) {
        float abs = std::abs(in[i]);
        abs_max = std::max(abs_max, abs);
        max = std::max(max, in[i]);
      }
      if (abs_max > 0.0f) {
        float scale = ((7 * abs_max > 8 * max) ? 8 : 7) / abs_max;
        for (size_t i = 0; i < dim; i += 2) {
          auto v1 = static_cast<int8_t>(std::round(in[i] * scale));
          auto v2 = static_cast<int8_t>(std::round(in[i + 1] * scale));
          out[i / 2] = (static_cast<uint8_t>(v1) << 4) |
                       (static_cast<uint8_t>(v2) & 0xF);
        }
      } else {
        std::fill(out, out + dim / 2, static_cast<uint8_t>(9));
      }
    }
    return 0;
  }

  //! Normalize results
  virtual int normalize(const void * /*query*/,
                        const IndexQueryMeta & /*qmeta*/,
                        IndexDocumentList & /*result*/) const override {
    return 0;
  }
};

int IVFCentroidIndex::init(const IndexMeta &meta,
                           const ailego::Params &params) {
  meta_ = meta;

  params.get(PARAM_IVF_BUILDER_OPTIMIZER_CLASS, &builder_class_);
  params.get(PARAM_IVF_BUILDER_OPTIMIZER_PARAMS, &builder_params_);
  params.get(PARAM_IVF_SEARCHER_OPTIMIZER, &searcher_class_);
  params.get(PARAM_IVF_SEARCHER_OPTIMIZER_PARAMS, &searcher_params_);

  return 0;
}

int IVFCentroidIndex::search(const void *query, const IndexQueryMeta &qmeta,
                             size_t count,
                             IndexSearcher::Context::Pointer &ctx) {
  int ret = 0;

  if (reformer_) {
    std::string buffer;
    IndexQueryMeta ometa;
    ret = reformer_->transform(query, qmeta, count, &buffer, &ometa);
    if (ret != 0) {
      LOG_ERROR("Failed to transform querys by reformer");
      return ret;
    }
    ret = searcher_->search_impl(buffer.data(), ometa, count, ctx);
  } else {
    ret = searcher_->search_impl(query, qmeta, count, ctx);
  }

  ivf_check_with_msg(ret, "Failed to search in centroid index for %s",
                     IndexError::What(ret));

  return 0;
}

uint32_t IVFCentroidIndex::search_nearest_centroid(const void *query,
                                                   size_t len) {
  //! Called in building index precedure, so transform the query is needless
  if (len != meta_.element_size()) {
    LOG_ERROR("Invalid query size actual: %zu, expected: %u", len,
              meta_.element_size());
    return kInvalidID;
  }

  thread_local IndexSearcher::Context::Pointer context(
      searcher_->create_context());
  context->set_topk(1);

  IndexQueryMeta qmeta(meta_.data_type(), meta_.dimension());
  int ret = searcher_->search_impl(query, qmeta, context);
  if (ret != 0 || context->result().empty()) {
    LOG_ERROR("Failed to search nearest centroid, with ret %d", ret);
    return kInvalidID;
  }

  return static_cast<uint32_t>(context->result()[0].key());
}

uint32_t IVFCentroidIndex::transform_and_search_nearest_centroid(
    const void *record, const IndexQueryMeta &rmeta,
    IndexSearcher::Context::Pointer &ctx) const {
  int ret = 0;
  if (reformer_) {
    std::string buffer;
    IndexQueryMeta ometa;
    ret = reformer_->convert(record, rmeta, &buffer, &ometa);
    if (ret != 0) {
      LOG_ERROR("Failed to transform querys by reformer");
      return kInvalidID;
    }
    ret = searcher_->search_impl(buffer.data(), ometa, ctx);
  } else {
    ret = searcher_->search_impl(record, rmeta, ctx);
  }
  if (ret != 0 || ctx->result().empty()) {
    LOG_ERROR("Failed to search in centroid index for %s",
              IndexError::What(ret));
    return kInvalidID;
  }

  return static_cast<uint32_t>(ctx->result()[0].key());
}

IndexHolder::Pointer IVFCentroidIndex::quantize_holder(
    const IndexHolder::Pointer &holder) {
  auto input = holder;
  if (meta_.reformer_name() == kMipsReformerName &&
      meta_.metric_name() == kL2MetricName &&
      (quantizer_class_ == kInt8QuantizerName ||
       quantizer_class_ == kInt4QuantizerName)) {
    //! Reverse for Mips if do convert by integer quantizer
    auto reverse = IndexFactory::CreateConverter(kMipsRevConverterName);
    if (!reverse) {
      LOG_ERROR("Failed to create converter %s", kMipsRevConverterName);
      return nullptr;
    }
    ailego::Params params;
    auto p = meta_.reformer_params();
    params.set(MIPS_REVERSE_CONVERTER_M_VALUE,
               p.get_as_uint32(MIPS_REFORMER_M_VALUE));
    params.set(MIPS_REVERSE_CONVERTER_U_VALUE,
               p.get_as_float(MIPS_REFORMER_U_VALUE));
    params.set(MIPS_REVERSE_CONVERTER_L2_NORM,
               p.get_as_uint32(MIPS_REFORMER_L2_NORM));
    params.set(MIPS_REVERSE_CONVERTER_FORCED_SINGLE_FLOAT,
               p.get_as_float(MIPS_REFORMER_FORCED_HALF_FLOAT));
    int ret = reverse->init(meta_, params);
    if (ret != 0) {
      LOG_ERROR("Fail to init converter %s", kMipsRevConverterName);
      return nullptr;
    }
    ret = IndexConverter::TrainAndTransform(reverse, holder);
    if (ret != 0) {
      LOG_ERROR("Fail to transform converter %s", kMipsRevConverterName);
      return nullptr;
    }
    input = reverse->result();
    meta_ = reverse->meta();
    meta_.set_metric(kIPMetricName, 0, ailego::Params());
    meta_.set_reformer("", 0, ailego::Params());
  }

  auto converter = IndexFactory::CreateConverter(quantizer_class_);
  if (!converter) {
    LOG_ERROR("Failed to create converter %s", quantizer_class_.c_str());
    return nullptr;
  }
  int ret = converter->init(meta_, quantizer_params_);
  if (ret != 0) {
    LOG_ERROR("Fail to init converter %s", quantizer_class_.c_str());
    return nullptr;
  }

  ret = IndexConverter::TrainAndTransform(converter, input);
  if (ret != 0) {
    LOG_ERROR("Fail to tranform converter %s", quantizer_class_.c_str());
    return nullptr;
  }

  meta_ = converter->meta();
  return converter->result();
}

int IVFCentroidIndex::build_index(
    const IndexCluster::CentroidList &centroid_list,
    const IndexDumper::Pointer &dumper) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder(builder_class_);
  if (!builder) {
    LOG_ERROR("Failed to create builder %s", builder_class_.c_str());
    return IndexError_NoExist;
  }

  IndexHolder::Pointer holder =
      std::make_shared<CentroidsIndexHolder>(meta_, centroid_list);
  if (!holder) {
    return IndexError_NoMemory;
  }
  if (holder->count() == 0) {
    LOG_ERROR("No centroids to build");
    return IndexError_InvalidArgument;
  }
  centroids_count_ = holder->count();

  //! Set default params if not given
  auto count = std::to_string(
      static_cast<size_t>(std::ceil(std::sqrt(centroids_count_ / 10.0))));
  // if (IsHcBuilder(builder_class_) &&
  //     !builder_params_.has(hc::PARAM_HC_BUILDER_CENTROID_COUNT)) {
  //   builder_params_.set(hc::PARAM_HC_BUILDER_CENTROID_COUNT, count);
  // } else if (builder_class_ == "GcBuilder" &&
  //            !builder_params_.has(hc::PARAM_GC_BUILDER_CENTROID_COUNT)) {
  //   builder_params_.set(hc::PARAM_GC_BUILDER_CENTROID_COUNT, count);
  // }
  if (!quantizer_class_.empty()) {
    holder = this->quantize_holder(holder);
    if (!holder) {
      return IndexError_Runtime;
    }
  }

  const auto name = builder_class_.c_str();
  int ret = builder->init(meta_, builder_params_);
  ivf_check_with_msg(ret, "%s init failed, ret=%d", name, ret);

  // if (IsHcBuilder(builder_class_) && quantizer_class_.empty()) {
  //   auto trainer = this->prepare_trainer(centroid_list);
  //   ret = trainer ? builder->train(trainer) : builder->train(holder);
  // } else {
  //   ret = builder->train(holder);
  // }

  ret = builder->train(holder);
  ivf_check_with_msg(ret, "%s train failed, ret=%d", name, ret);

  ret = builder->build(holder);
  ivf_check_with_msg(ret, "%s build failed, ret=%d", name, ret);

  ret = builder->dump(dumper);
  ivf_check_with_msg(ret, "%s dump failed, ret=%d", name, ret);

  ret = dumper->close();
  ivf_check_error_code(ret);

  return 0;
}

int IVFCentroidIndex::build(const IndexCluster::CentroidList &centroid_list) {
  index_building_ = true;
  //! Build and dump the index
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  if (!dumper) {
    LOG_ERROR("Failed to create MemoryDumper");
    return IndexError_NoExist;
  }
  path_ = IVFUtility::GenerateRandomPath(kTempralPathPrefix);
  int ret = dumper->create(path_);
  if (ret != 0) {
    LOG_ERROR("IndexDumper create path %s failed", path_.c_str());
    return ret;
  }
  ret = this->build_index(centroid_list, dumper);
  ivf_check_error_code(ret);

  auto rope = IndexMemory::Instance()->open(path_);
  if (!rope) {
    LOG_ERROR("Open memory path %s failed.", path_.c_str());
    return ret;
  }
  if (rope->count() != 1) {
    LOG_ERROR("Graph Rope block count not equal with 1.");
    return ret;
  }
  (*rope)[0].read(0, &data_, 0);
  size_ = (*rope)[0].size();

  //! Load the index
  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MemoryReadStorage");
  if (!container) {
    LOG_ERROR("Failed to create MemoryReadStorage");
    return IndexError_NoExist;
  }
  ret = container->init(ailego::Params());
  ivf_check_with_msg(ret, "Failed to initialize MemoryReadStorage for %s",
                     IndexError::What(ret));
  ret = container->open(path_, false);
  ivf_check_with_msg(ret, "Failed to load path in MemoryReadStorage for %s",
                     IndexError::What(ret));

  ailego::Params searcher_params;
  if (!searcher_class_.empty()) {
    searcher_params.set(PARAM_IVF_SEARCHER_OPTIMIZER, searcher_class_);
  }
  if (!searcher_params_.empty()) {
    searcher_params.set(PARAM_IVF_SEARCHER_OPTIMIZER_PARAMS, searcher_params_);
  }
  ret = this->load(container, searcher_params);
  ivf_check_with_msg(ret, "IVFCentroidIndex load failed with %s",
                     IndexError::What(ret));

  return 0;
}

int IVFCentroidIndex::load(const IndexStorage::Pointer &container,
                           const ailego::Params params) {
  if (!container) {
    LOG_ERROR("Invalid container");
    return IndexError_InvalidArgument;
  }

  int ret = IndexHelper::DeserializeFromStorage(container.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from container");
    return ret;
  }

  auto reformer_name = meta_.reformer_name();
  if (!reformer_name.empty()) {
    LOG_DEBUG("Load CentroidIndex with reformer %s, metric %s",
              reformer_name.c_str(), meta_.metric_name().c_str());
    if ((reformer_name == kInt8ReformerName ||
         reformer_name == kInt4ReformerName) &&
        meta_.metric_name() == kIPMetricName) {
      if (reformer_name == kInt8ReformerName) {
        reformer_ = std::make_shared<Int8QuantizerReformer4IP>();
      } else {
        reformer_ = std::make_shared<Int4QuantizerReformer4IP>();
      }
      if (!reformer_) {
        return IndexError_NoMemory;
      }
    } else {
      reformer_ = IndexFactory::CreateReformer(reformer_name);
      if (!reformer_) {
        LOG_ERROR("Failed to create reformer %s", reformer_name.c_str());
        return IndexError_NoExist;
      }
    }
    ret = reformer_->init(meta_.reformer_params());
    ivf_check_with_msg(ret, "Failed to initialize reformer %s",
                       reformer_name.c_str());
  }

  searcher_class_ = meta_.searcher_name();
  params.get(PARAM_IVF_SEARCHER_OPTIMIZER, &searcher_class_);
  params.get(PARAM_IVF_SEARCHER_OPTIMIZER_PARAMS, &searcher_params_);
  searcher_ = IndexFactory::CreateSearcher(searcher_class_);
  if (!searcher_) {
    LOG_ERROR("Failed to create searcher %s", searcher_class_.c_str());
    return IndexError_Runtime;
  }

  auto searcher_params = meta_.searcher_params();
  searcher_params.merge(searcher_params_);
  ret = searcher_->init(searcher_params);
  ivf_check_with_msg(ret, "Failed to initialize searcher %s",
                     searcher_class_.c_str());

  IndexMetric::Pointer metric;
  if (index_building_) {
    // The searcher index metric should specified in building process,
    // otherwise the query_metric will be used in searching
    metric = IndexFactory::CreateMetric(meta_.metric_name());
    ivf_assert_with_msg(metric, IndexError_NoExist,
                        "Failed to create metric %s",
                        meta_.metric_name().c_str());
    ret = metric->init(meta_, meta_.metric_params());
    ivf_check_with_msg(ret, "Failed to initialize metric");
  }
  ret = searcher_->load(container, metric);
  ivf_check_with_msg(ret, "Failed to load searcher %s",
                     searcher_class_.c_str());

  return 0;
}

IndexTrainer::Pointer IVFCentroidIndex::prepare_trainer(
    const IndexCluster::CentroidList &centroid_list) {
  IndexCluster::CentroidList level1_centroids;
  bool two_level = false;
  for (auto &it : centroid_list) {
    auto centroid = it;
    if (!centroid.subitems().empty()) {
      two_level = true;
    }
    centroid.mutable_subitems()->clear();
    centroid.mutable_similars()->clear();
    level1_centroids.emplace_back(centroid);
  }
  if (!two_level) {
    return IndexTrainer::Pointer();
  }

  IndexBundle::Pointer bundle;
  IndexCluster::Serialize(meta_, level1_centroids, &bundle);
  return std::make_shared<FakeClusterTrainer>(meta_, bundle);
}

}  // namespace core
}  // namespace zvec
