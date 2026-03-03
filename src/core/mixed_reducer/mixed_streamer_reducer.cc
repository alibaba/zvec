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
#include "mixed_streamer_reducer.h"
#include <ailego/pattern/defer.h>
#include <utility/sparse_utility.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_logger.h>
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec {
namespace core {

int MixedStreamerReducer::init(const ailego::Params &params) {
  enable_pk_rewrite_ =
      params.get_as_bool(PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE);
  params.get(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS,
             &num_of_add_threads_);
  if (num_of_add_threads_ <= 0) {
    LOG_ERROR("Wrong parameter. %s must be set greater than 0.",
              PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS.c_str());
    return IndexError_InvalidArgument;
  }

  params_ = params;

  state_ = STATE_INITED;
  return 0;
}

int MixedStreamerReducer::cleanup(void) {
  streamers_.clear();
  target_streamer_->cleanup();

  target_builder_->cleanup();
  doc_cache_.clear();

  stats_.clear_attributes();
  state_ = STATE_UNINITED;
  return 0;
}

int MixedStreamerReducer::set_target_streamer_wiht_info(
    const IndexBuilder::Pointer builder, const IndexStreamer::Pointer streamer,
    const IndexConverter::Pointer converter,
    const IndexReformer::Pointer reformer,
    const IndexQueryMeta &original_query_meta) {
  if (state_ != STATE_INITED) {
    LOG_ERROR("Set target streamer after init");
    return IndexError_Uninitialized;
  }

  target_builder_ = builder;
  target_streamer_ = streamer;
  target_builder_converter_ = converter;
  target_streamer_reformer_ = reformer;
  original_query_meta_ = original_query_meta;

  is_sparse_ =
      target_streamer_->meta().meta_type() == IndexMeta::MetaType::MT_SPARSE;

  state_ = STATE_STREAMER_SET;
  return 0;
}

int MixedStreamerReducer::feed_streamer_with_reformer(
    IndexStreamer::Pointer streamer, const IndexReformer::Pointer reformer) {
  if (!(state_ == STATE_STREAMER_SET || state_ == STATE_FEED)) {
    LOG_ERROR("Set target streamer or feed before feed");
    return IndexError_Uninitialized;
  }

  if (!streamer) {
    LOG_ERROR("Streamer nullptr");
    return IndexError_InvalidArgument;
  }

  auto check_datatype = [&](const IndexMeta & /*target_meta*/,
                            const IndexMeta &source_meta) -> bool {
    if (!streamers_.empty()) {
      auto &last_meta = streamers_.back()->meta();
      return last_meta.data_type() == source_meta.data_type() &&
             last_meta.dimension() == source_meta.dimension() &&
             last_meta.unit_size() == source_meta.unit_size();
    }
    // TODO: check target meta
    return true;
  };

  auto check_other = [&](const IndexMeta &target_meta,
                         const IndexMeta &source_meta) -> bool {
    return target_meta.meta_type() == source_meta.meta_type();
    // when create a new index, there is a case that ip_flat merged into l2_hnsw
    // target_meta.metric_name() == source_meta.metric_name();
  };

  if (!(check_datatype(target_streamer_->meta(), streamer->meta()) &&
        check_other(target_streamer_->meta(), streamer->meta()))) {
    LOG_ERROR("Streamer meta mismatch");
    return IndexError_InvalidArgument;
  }

  if (streamers_.empty()) {
    is_target_and_source_same_reformer_ =
        target_streamer_->meta().reformer_name() ==
        streamer->meta().reformer_name();
  }

  streamers_.push_back(streamer);
  source_streamers_reformers_.push_back(reformer);

  state_ = STATE_FEED;
  return 0;
}

int MixedStreamerReducer::reduce(const IndexFilter &filter) {
  if (state_ != STATE_FEED) {
    LOG_ERROR("Feed streamers first");
    return IndexError_Uninitialized;
  }
  if (thread_pool_ == nullptr) {
    LOG_ERROR("Thread pool is not set");
    return IndexError_Uninitialized;
  }

  ailego::ElapsedTime timer;


  std::vector<int> add_results(num_of_add_threads_, -1);
  auto add_group = thread_pool_->make_group();

  std::vector<int> read_results(streamers_.size(), -1);
  // TODO: use id instead of key
  uint32_t id_offset = 0, next_id = 0;

  if (is_sparse_) {
    for (size_t i = 0; i < num_of_add_threads_; i++) {
      add_group->submit(ailego::Closure::New(
          this, &MixedStreamerReducer::add_sparse_vec, &add_results[i]));
    }

    for (size_t i = 0; i < streamers_.size(); i++) {
      // due to filter, producing can't be parallel
      read_results[i] = read_sparse_vec(i, filter, id_offset, &next_id);
      id_offset += streamers_[i]->create_sparse_provider()->count();
    }

    sparse_mt_list_.done();
  } else {
    for (size_t i = 0; i < num_of_add_threads_; i++) {
      add_group->submit(ailego::Closure::New(
          this, &MixedStreamerReducer::add_vec, &add_results[i]));
      // add_vec(&add_results[i]);
    }

    for (size_t i = 0; i < streamers_.size(); i++) {
      read_results[i] = read_vec(i, filter, id_offset, &next_id);
      id_offset += streamers_[i]->create_provider()->count();
    }

    mt_list_.done();
  }
  add_group->wait_finish();

  auto check_results = [](const std::vector<int> &results) -> bool {
    return std::all_of(std::begin(results), std::end(results),
                       [](int item) { return item == 0; });
  };

  if (!check_results(read_results)) {
    LOG_ERROR("Get vector from entities failed");
    return IndexError_Runtime;
  }

  if (!check_results(add_results)) {
    LOG_ERROR("add vector failed");
    return IndexError_Runtime;
  }

  stats_.set_reduced_costtime(timer.seconds());
  state_ = STATE_REDUCE;
  if (target_builder_ != nullptr) {
    IndexBuild();
  }

  LOG_INFO("End brute force reduce. cost time: [%zu]s",
           (size_t)timer.seconds());
  return 0;
}

int MixedStreamerReducer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Begin brute force reducer dump");

  if (state_ != STATE_REDUCE) {
    LOG_WARN("Reduce first before dump");
    return IndexError_NoReady;
  }

  ailego::ElapsedTime timer;
  int ret = 0;
  if (target_builder_ != nullptr) {
    target_builder_->dump(dumper);
  } else {
    target_streamer_->dump(dumper);
  }
  if (ret == IndexError_NotImplemented) {
    LOG_WARN("Dump index not implemented");
  } else if (ret < 0) {
    LOG_ERROR("Failed to dump in streamer");
  }

  return ret;
}

int MixedStreamerReducer::read_vec(size_t source_streamer_index,
                                   const IndexFilter &filter,
                                   const uint32_t id_offset,
                                   uint32_t *next_id) {
  const auto &streamer = streamers_[source_streamer_index];
  const auto &reformer = source_streamers_reformers_[source_streamer_index];
  const IndexQueryMeta source_streamer_query_meta{streamer->meta().data_type(),
                                                  streamer->meta().dimension()};

  bool need_revert = (target_streamer_->meta().reformer_name() !=
                          streamer->meta().reformer_name() &&
                      reformer != nullptr);
  if (target_builder_ && reformer) {
    need_revert = true;
  }

  IndexProvider::Pointer provider = streamer->create_provider();
  IndexProvider::Iterator::Pointer iterator = provider->create_iterator();

  while (iterator->is_valid()) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("read_vec cancelled.");
      return 0;
    }
    if (filter(iterator->key() + (uint64_t)id_offset)) {
      (*stats_.mutable_filtered_count())++;
      iterator->next();
      continue;
    }

    std::vector<uint8_t> bytes;
    if (need_revert) {
      std::string new_vector;
      if (reformer->revert(iterator->data(), source_streamer_query_meta,
                           &new_vector) != 0) {
        LOG_ERROR("Failed to revert the vector");
        return IndexError_Runtime;
      }
      bytes.resize(new_vector.size());
      memcpy(bytes.data(), new_vector.data(), bytes.size());
    } else {
      // TODO: eliminate the copy
      bytes.resize(provider->element_size());
      memcpy(bytes.data(), iterator->data(), bytes.size());
    }

    // TODO: use id instead of key
    if (!mt_list_.produce(VectorItem((*next_id)++, std::move(bytes)))) {
      LOG_ERROR("Produce vector to queue failed. key[%lu]",
                (size_t)iterator->key());
      return IndexError_Runtime;
    }
    iterator->next();
  }
  return 0;
}

void MixedStreamerReducer::add_vec(int *result) {
  if (target_builder_ != nullptr) {
    add_vec_with_builder(result);
    return;
  }
  ailego::ElapsedTime timer;
  auto target_streamer_context = target_streamer_->create_context();
  auto target_streamer_query_meta = IndexQueryMeta{
      IndexMeta::MetaType::MT_DENSE, target_streamer_->meta().data_type(),
      target_streamer_->meta().dimension()};
  const bool need_convert = (!is_target_and_source_same_reformer_) &&
                            target_streamer_reformer_ != nullptr;

  AILEGO_DEFER([&]() {
    // make producer quit
    mt_list_.done();
  });

  VectorItem vector_item;
  while (mt_list_.consume(&vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_vec cancelled.");
      return;
    }

    const void *vector = vector_item.vec_.data();
    std::string new_vector;


    if (need_convert) {
      IndexQueryMeta new_meta;
      if (target_streamer_reformer_->convert(vector, original_query_meta_,
                                             &new_vector, &new_meta) != 0) {
        LOG_ERROR("Failed to transform vector");
        *result = IndexError_Runtime;
        return;
      }
      vector = new_vector.data();
    }
    // 1. no reformer: target_streamer_query_meta_ = original_query_meta_
    // 2. has reformer, matched(need_convert = false): use
    // target_streamer_query_meta_
    // 3. has reformer, not matched(need_convert = true): use
    // target_streamer_query_meta_


    // TODO: use id instead of key
    int ret = target_streamer_->add_with_id_impl(
        (uint32_t)vector_item.pkey_, vector, target_streamer_query_meta,
        target_streamer_context);
    if (ret != 0) {
      LOG_ERROR("Insert target streamer failed. ret[%d] reason[%s] pkey[%zu]",
                ret, IndexError::What(ret), (size_t)vector_item.pkey_);
      *result = ret;
      return;
    }
  }

  *result = 0;
  LOG_DEBUG("add_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}

void MixedStreamerReducer::add_vec_with_builder(int *result) {
  ailego::ElapsedTime timer;
  auto target_streamer_query_meta = IndexQueryMeta{
      IndexMeta::MetaType::MT_DENSE, target_streamer_->meta().data_type(),
      target_streamer_->meta().dimension()};

  AILEGO_DEFER([&]() {
    // make producer quit
    mt_list_.done();
  });

  VectorItem vector_item;
  while (mt_list_.consume(&vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_vec cancelled.");
      return;
    }

    const void *vector = vector_item.vec_.data();
    std::string out_vector_buffer = std::string(
        static_cast<const char *>(vector),
        original_query_meta_.dimension() * original_query_meta_.unit_size());
    PushToDocCache(original_query_meta_, (uint32_t)vector_item.pkey_,
                   out_vector_buffer);
  }

  *result = 0;
  LOG_DEBUG("add_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}

void MixedStreamerReducer::add_sparse_vec(int *result) {
  ailego::ElapsedTime timer;
  auto target_streamer_context = target_streamer_->create_context();
  auto target_streamer_query_meta = IndexQueryMeta{
      IndexMeta::MetaType::MT_SPARSE,
      target_streamer_->meta().data_type(),
  };

  auto need_convert = !is_target_and_source_same_reformer_ &&
                      target_streamer_reformer_ != nullptr;

  AILEGO_DEFER([&]() {
    // make producer quit
    sparse_mt_list_.done();
  });

  SparseVectorItem sparse_vector_item;
  while (sparse_mt_list_.consume(&sparse_vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_sparse_vec cancelled.");
      return;
    }
    auto sparse_count = sparse_vector_item.sparse_indices_.size();
    auto indices = sparse_vector_item.sparse_indices_.data();
    auto values = sparse_vector_item.sparse_values_.data();

    std::string converted_sparse_values_buffer;
    if (need_convert) {
      IndexQueryMeta new_meta;
      if (target_streamer_reformer_->convert(
              sparse_count, indices, values, original_query_meta_,
              &converted_sparse_values_buffer, &new_meta) != 0) {
        LOG_ERROR("Failed to transform vector");
        *result = IndexError_Runtime;
        return;
      }
      values = converted_sparse_values_buffer.data();
      target_streamer_query_meta = new_meta;
    }

    // TODO: use id instead of key
    int ret = target_streamer_->add_with_id_impl(
        (uint32_t)sparse_vector_item.pkey_, sparse_count, indices, values,
        target_streamer_query_meta, target_streamer_context);
    if (ret != 0) {
      LOG_ERROR("Insert target streamer failed. ret[%d] reason[%s] pkey[%zu]",
                ret, IndexError::What(ret), (size_t)sparse_vector_item.pkey_);
      *result = ret;
      return;
    }
  }

  *result = 0;
  LOG_DEBUG("add_sparse_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}


int MixedStreamerReducer::read_sparse_vec(size_t source_streamer_index,
                                          const IndexFilter &filter,
                                          const uint32_t id_offset,
                                          uint32_t *next_id) {
  const auto &streamer = streamers_[source_streamer_index];
  const auto &reformer = source_streamers_reformers_[source_streamer_index];
  const bool need_revert =
      !is_target_and_source_same_reformer_ && reformer != nullptr;

  IndexStreamer::SparseProvider::Pointer provider =
      streamer->create_sparse_provider();
  IndexStreamer::SparseProvider::Iterator::Pointer iterator =
      provider->create_iterator();

  while (iterator->is_valid()) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("read_sparse_vec cancelled.");
      return 0;
    }
    if (filter(iterator->key() + (uint64_t)id_offset)) {
      (*stats_.mutable_filtered_count())++;
      iterator->next();
      continue;
    }

    auto sparse_count = iterator->sparse_count();
    std::vector<uint32_t> sparse_indices(sparse_count);
    std::string sparse_values;

    if (need_revert) {
      std::string new_sparse_values;
      if (reformer->revert(iterator->sparse_count(), iterator->sparse_indices(),
                           iterator->sparse_data(),
                           {
                               IndexMeta::MetaType::MT_SPARSE,
                               streamer->meta().data_type(),
                           },
                           &new_sparse_values) != 0) {
        LOG_ERROR("Failed to revert the sparse vector");
        return IndexError_Runtime;
      }
      sparse_values = std::move(new_sparse_values);
    } else {
      sparse_values.resize(sparse_count * streamer->meta().unit_size());
      memcpy(sparse_values.data(), iterator->sparse_data(),
             sparse_values.size());
    }

    // TODO: eliminate the copy
    memcpy(sparse_indices.data(), iterator->sparse_indices(),
           sparse_indices.size() * sizeof(uint32_t));

    // TODO: use id instead of key
    if (!sparse_mt_list_.produce(SparseVectorItem((*next_id)++,
                                                  std::move(sparse_indices),
                                                  std::move(sparse_values)))) {
      LOG_ERROR("Produce vector to queue failed. key[%lu]",
                (size_t)iterator->key());
      return IndexError_Runtime;
    }
    iterator->next();
  }
  return 0;
}

void MixedStreamerReducer::PushToDocCache(const IndexQueryMeta &meta,
                                          uint32_t doc_id, std::string &doc) {
  std::lock_guard<std::mutex> lock(mutex_);
  while (doc_cache_.size() <= doc_id) {
    std::string fake_data(meta.dimension() * meta.unit_size(), 0);
    doc_cache_.push_back(std::make_pair(kInvalidKey, fake_data));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, doc);
}

int MixedStreamerReducer::IndexBuild() {
  IndexHolder::Pointer target_holder;
  if (original_query_meta_.data_type() == core::IndexMeta::DataType::DT_FP16) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP16>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint16_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else if (original_query_meta_.data_type() ==
             core::IndexMeta::DataType::DT_FP32) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP32>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<float> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else if (original_query_meta_.data_type() ==
             core::IndexMeta::DataType::DT_INT8) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_INT8>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint8_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else {
    LOG_ERROR("data_type is not support");
    return core::IndexError_Runtime;
  }
  if (target_builder_converter_) {
    core::IndexConverter::TrainAndTransform(target_builder_converter_,
                                            target_holder);
    target_holder = target_builder_converter_->result();
  }
  target_builder_->train(target_holder);
  target_builder_->build(target_holder);
  return 0;
}

INDEX_FACTORY_REGISTER_STREAMER_REDUCER_ALIAS(MixedStreamerReducer,
                                              MixedStreamerReducer);

}  // namespace core
}  // namespace zvec
