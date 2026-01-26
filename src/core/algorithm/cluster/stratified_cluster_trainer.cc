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
#include "stratified_cluster_trainer.h"
#include <ailego/utility/time_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include "framework/index_error.h"
#include "framework/index_factory.h"
#include "framework/index_helper.h"
#include "cluster_params.h"

namespace zvec {
namespace core {

const std::string StratifiedClusterTrainer::SEP_TOKEN = "*";
const std::string StratifiedClusterTrainer::DEFAULT_CLUSTER_CLASS =
    "OptKmeansCluster";

int StratifiedClusterTrainer::init_params(const ailego::Params &params) {
  params.get(STRATIFIED_TRAINER_SAMPLE_COUNT, &sample_count_);
  params.get(STRATIFIED_TRAINER_SAMPLE_RATIO, &sample_ratio_);
  params.get(STRATIFIED_TRAINER_THREAD_COUNT, &thread_count_);
  cluster_auto_tuning_ = params.get_as_bool(STRATIFIED_TRAINER_AUTOAUNE);

  std::string centroids_num =
      params.get_as_string(STRATIFIED_TRAINER_CLUSTER_COUNT);
  if (!centroids_num.empty()) {
    ailego::StringHelper::Split(centroids_num, SEP_TOKEN, &centroid_num_vec_);
    for (size_t i = 0; i < centroid_num_vec_.size(); ++i) {
      if (centroid_num_vec_[i] == 0) {
        LOG_ERROR("Invalid centroid num %s", centroids_num.c_str());
        return IndexError_InvalidArgument;
      }
    }
  } else {
    LOG_ERROR("No centroids_num configed.");
    return IndexError_InvalidArgument;
  }

  size_t level_cnt = centroid_num_vec_.size();
  for (size_t i = 1; i <= level_cnt; ++i) {
    std::string level_params_key =
        STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + std::to_string(i);
    ailego::Params level_params;
    params.get(level_params_key, &level_params);
    cluster_params_.push_back(level_params);
  }

  std::string cluster_class(DEFAULT_CLUSTER_CLASS);
  params.get(STRATIFIED_TRAINER_CLASS_NAME, &cluster_class);
  ailego::StringHelper::Split(cluster_class, SEP_TOKEN, &cluster_class_);
  if (cluster_class_.size() == 1) {
    // repeat the cluster class to level_cnt
    for (size_t i = 1; i < level_cnt; ++i) {
      cluster_class_.push_back(cluster_class_[0]);
    }
  } else if (cluster_class_.size() != level_cnt) {
    LOG_ERROR("Cluster class should be equal to level count");
    return IndexError_InvalidArgument;
  }
  return 0;
}

int StratifiedClusterTrainer::init(const IndexMeta &index_meta,
                                   const ailego::Params &params) {
  int err = init_params(params);
  if (err != 0) {
    LOG_ERROR("init params failed, errno:%d,%s", err, IndexError::What(err));
    return err;
  }

  meta_ = index_meta;
  ailego::Params cluster_params;
  if (centroid_num_vec_.size() == 0) {
    LOG_ERROR("invalid centroid num");
    return IndexError_InvalidArgument;
  } else if (centroid_num_vec_.size() == 1) {
    // one level clustering
    class_name_ = cluster_class_[0];
    cluster_params = cluster_params_[0];
    suggest_centriod_cnt_ = centroid_num_vec_[0];
  } else if (centroid_num_vec_.size() == 2) {
    // cluster level > 1
    class_name_ = "StratifiedCluster";
    int level_cnt = centroid_num_vec_.size();
    cluster_params.set(STRATIFIED_CLUSTER_FIRST_CLASS,
                       cluster_class_[level_cnt - 2]);
    cluster_params.set(STRATIFIED_CLUSTER_SECOND_CLASS,
                       cluster_class_[level_cnt - 1]);
    cluster_params.set(STRATIFIED_CLUSTER_FIRST_COUNT,
                       centroid_num_vec_[level_cnt - 2]);
    cluster_params.set(STRATIFIED_CLUSTER_SECOND_COUNT,
                       centroid_num_vec_[level_cnt - 1]);
    cluster_params.set(STRATIFIED_CLUSTER_FIRST_PARAMS,
                       cluster_params_[level_cnt - 2]);
    cluster_params.set(STRATIFIED_CLUSTER_SECOND_PARAMS,
                       cluster_params_[level_cnt - 1]);
    cluster_params.set(STRATIFIED_CLUSTER_AUTO_TUNING, cluster_auto_tuning_);
    suggest_centriod_cnt_ =
        centroid_num_vec_[level_cnt - 1] * centroid_num_vec_[level_cnt - 2];
  } else {
    LOG_ERROR("Unsupported more than 2 level clustering.");
    return IndexError_Unsupported;
  }

  cluster_ = IndexFactory::CreateCluster(class_name_);
  if (!cluster_) {
    LOG_ERROR("Failed to create cluster[%s]", class_name_.c_str());
    return IndexError_InvalidArgument;
  }
  int result = cluster_->init(meta_, cluster_params);
  if (result != 0) {
    LOG_ERROR("Failed to initialize of cluster[%s], error: %d, %s",
              class_name_.c_str(), result, IndexError::What(result));
    return result;
  }
  if (suggest_centriod_cnt_ > 0) {
    cluster_->suggest(suggest_centriod_cnt_);
  }

  return 0;
}

int StratifiedClusterTrainer::cleanup(void) {
  cluster_ = nullptr;
  centroids_.clear();
  return 0;
}

int StratifiedClusterTrainer::train(IndexThreads::Pointer threads,
                                    IndexHolder::Pointer holder) {
  ailego::ElapsedTime timer;
  if (!holder) {
    return IndexError_InvalidArgument;
  }
  if (!holder->is_matched(meta_)) {
    return IndexError_Mismatch;
  }
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }

  size_t train_sample_count = std::max(
      sample_count_, static_cast<uint32_t>(sample_ratio_ * holder->count()));

  IndexFeatures::Pointer features;
  if (train_sample_count > 0) {
    LOG_INFO(
        "Train sampling, SampleCount=%u, SampleRatio=%f, HolderCount=%lu, "
        "TrainCount=%lu",
        sample_count_, sample_ratio_, holder->count(), train_sample_count);

    auto sampler = std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
        meta_, train_sample_count);
    size_t pre_reserve = train_sample_count < holder->count()
                             ? train_sample_count
                             : holder->count();
    sampler->reserve(pre_reserve);
    for (auto iter = holder->create_iterator(); iter && iter->is_valid();
         iter->next()) {
      sampler->emplace(iter->data());
    }
    features = sampler;
    stats_.set_trained_count(train_sample_count);
  } else {
    LOG_INFO(
        "Do no sampling, SampleCount=%u, SampleRatio=%f, "
        "HolderCount=%lu, TrainCount=%lu",
        sample_count_, sample_ratio_, holder->count(), holder->count());

    auto no_sampler = std::make_shared<CompactIndexFeatures>(meta_);
    for (auto iter = holder->create_iterator(); iter && iter->is_valid();
         iter->next()) {
      no_sampler->emplace(iter->data());
    }

    features = no_sampler;
    stats_.set_trained_count(holder->count());
  }
  stats_.set_discarded_count(0);

  // Holder is not needed, cleanup it.
  holder.reset();

  int result = cluster_->mount(features);
  if (result != 0) {
    LOG_ERROR("Failed to mount features of cluster[%s], error: %d, %s",
              class_name_.c_str(), result, IndexError::What(result));
    return result;
  }

  centroids_.clear();
  result = cluster_->cluster(std::move(threads), centroids_);
  if (result != 0) {
    LOG_ERROR("Failed to cluster features of cluster[%s], error: %d, %s",
              class_name_.c_str(), result, IndexError::What(result));
    return result;
  }

  // check build result
  std::vector<size_t> level_size;
  std::function<void(const IndexCluster::CentroidList &, size_t)>
      cal_centroid_cnt =
          [&cal_centroid_cnt, &level_size](
              const IndexCluster::CentroidList &cents, size_t level) {
            if (level > level_size.size()) {
              level_size.resize(level);
            }
            level_size[level - 1] += cents.size();
            for (const auto &it : cents) {
              if (!it.subitems().empty()) {
                cal_centroid_cnt(it.subitems(), level + 1);
              }
            }
          };
  cal_centroid_cnt(centroids_, 1);

  size_t centroids_num = level_size[level_size.size() - 1];
  if (centroids_num > suggest_centriod_cnt_) {
    LOG_WARN(
        "Built centroid(%zd level) count[%zd] bigger than expected "
        "count[%d]",
        level_size.size(), centroids_num, suggest_centriod_cnt_);
  } else {
    LOG_INFO("Built centroid(%zd level) count[%zd], expected count[%d]",
             level_size.size(), centroids_num, suggest_centriod_cnt_);
  }

  stats_.set_trained_costtime(timer.milli_seconds());

  return 0;
}

int StratifiedClusterTrainer::load(IndexStorage::Pointer cntr) {
  if (!cntr) {
    LOG_ERROR("IndexStorage is nullptr.");
    return IndexError_InvalidArgument;
  }
  std::shared_ptr<MemoryIndexBundle> bundle =
      std::make_shared<MemoryIndexBundle>();
  if (!bundle) {
    LOG_ERROR("New MemoryInndexBundle failed.");
    return IndexError_NoMemory;
  }

  auto results = cntr->get_all();
  for (auto &it : results) {
    IndexStorage::Segment::Pointer &seg = it.second;
    if (!seg) {
      LOG_ERROR("Get Segment %s failed.", it.first.c_str());
      return IndexError_InvalidArgument;
    }
    size_t data_size = seg->data_size();
    const void *data = nullptr;
    size_t actual_size = seg->read(0, &data, data_size);
    if (actual_size != data_size) {
      LOG_ERROR("Read data failed expect %zu, actual %zu.", data_size,
                actual_size);
      return IndexError_ReadData;
    }
    bundle->set(it.first, data, data_size);
  }

  int result = IndexHelper::DeserializeFromStorage(cntr.get(), &meta_);
  if (result != 0) {
    LOG_ERROR("Failed to deserialize meta from container");
    return result;
  }

  result = IndexCluster::Deserialize(meta_, std::move(bundle), &centroids_);
  if (result != 0) {
    LOG_ERROR("Failed to deserialize index: %d", result);
    return result;
  }
  return 0;
}

int StratifiedClusterTrainer::dump(const IndexDumper::Pointer &dumper) {
  IndexBundle::Pointer bundle;
  int result = IndexCluster::Serialize(meta_, centroids_, &bundle);
  if (result != 0) {
    LOG_ERROR("IndexCluster Serialize failed with ret %d.", result);
    return result;
  }

  result = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (result != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return result;
  }

  for (const auto &it : bundle->all()) {
    size_t data_size = it.second.size();
    result = dumper->append(it.first, data_size, 0, 0);
    if (result != 0) {
      LOG_ERROR("Dumper append meta %s %zu failed.", it.first.c_str(),
                data_size);
      return IndexError_PackIndex;
    }
    size_t actual_size = dumper->write(it.second.buffer(), data_size);
    if (actual_size != data_size) {
      LOG_ERROR("Dumper segment %s expect %zu, actual %zu.", it.first.c_str(),
                data_size, actual_size);
      return IndexError_PackIndex;
    }
  }
  return 0;
}

const IndexMeta &StratifiedClusterTrainer::meta(void) const {
  return meta_;
}

const IndexTrainer::Stats &StratifiedClusterTrainer::stats(void) const {
  return stats_;
}

IndexBundle::Pointer StratifiedClusterTrainer::indexes(void) const {
  IndexBundle::Pointer bundle;
  IndexCluster::Serialize(meta_, centroids_, &bundle);
  return bundle;
}

//! Register Cluster Trainer in Factory
INDEX_FACTORY_REGISTER_TRAINER(StratifiedClusterTrainer);

}  // namespace core
}  // namespace zvec
