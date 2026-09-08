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
#include <cstring>
#include <ailego/pattern/defer.h>
#include <utility/sparse_utility.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include "mixed_reducer/merged_provider_index_holder.h"
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec {
namespace core {

namespace {

template <IndexMeta::DataType DT, typename T>
int MaterializeMergedInput(const MergedProviderIndexHolder::Pointer &source,
                           IndexHolder::Pointer *output) {
  auto snapshot =
      std::make_shared<MultiPassIndexHolder<DT>>(source->dimension());
  snapshot->reserve(source->count());
  auto iter = source->create_iterator();
  if (!iter) {
    return source->status() != 0 ? source->status() : IndexError_Runtime;
  }
  for (; iter->is_valid(); iter->next()) {
    const void *data = iter->data();
    if (source->status() != 0) {
      return source->status();
    }
    if (!data) {
      return IndexError_ReadData;
    }
    ailego::NumericalVector<T> vector(source->dimension());
    std::memcpy(vector.data(), data, source->element_size());
    if (!snapshot->emplace(iter->key(), std::move(vector))) {
      return IndexError_Mismatch;
    }
  }
  if (source->status() != 0) {
    return source->status();
  }
  if (snapshot->count() != source->count()) {
    return IndexError_Mismatch;
  }
  *output = std::move(snapshot);
  return 0;
}

}  // namespace

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
  source_streamers_reformers_.clear();
  merged_holder_.reset();
  if (target_streamer_) {
    target_streamer_->cleanup();
  }

  if (target_builder_) {
    target_builder_->cleanup();
  }

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

  if (target_builder_ != nullptr) {
    if (is_sparse_) {
      LOG_ERROR("Builder-backed sparse merge is not supported");
      return IndexError_Unsupported;
    }
    int ret = this->reduce_with_builder(filter);
    if (ret != 0) {
      LOG_ERROR("Failed to build target index, ret=%d", ret);
      return ret;
    }
    stats_.set_reduced_costtime(timer.seconds());
    state_ = STATE_REDUCE;
    LOG_INFO("End provider-backed reduce. cost time: [%zu]s",
             (size_t)timer.seconds());
    return 0;
  }

  std::vector<int> add_results(num_of_add_threads_, -1);
  auto add_group = thread_pool_->make_group();

  std::vector<int> read_results(streamers_.size(), -1);
  // TODO: use id instead of key
  // When merging into a non-empty target (e.g. reusing one input as base),
  // append new docs after the existing ones instead of overwriting from 0.
  uint32_t id_offset = 0;
  uint32_t next_id = 0;
  if (target_builder_ == nullptr) {
    if (is_sparse_) {
      auto provider = target_streamer_->create_sparse_provider();
      if (provider) next_id = provider->count();
    } else {
      auto provider = target_streamer_->create_provider();
      if (provider) next_id = provider->count();
    }
  }

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
      auto provider = streamers_[i]->create_provider();
      if (!provider) {
        LOG_ERROR("Failed to create source provider, index=%zu", i);
        read_results[i] = IndexError_Runtime;
        break;
      }
      read_results[i] = read_vec(i, provider, filter, id_offset, &next_id);
      if (read_results[i] != 0) {
        break;
      }
      id_offset += provider->count();
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
    ret = target_builder_->dump(dumper);
  } else {
    ret = target_streamer_->dump(dumper);
  }
  if (ret == 0 && merged_holder_ && merged_holder_->status() != 0) {
    ret = merged_holder_->status();
  }
  if (ret == IndexError_NotImplemented) {
    LOG_WARN("Dump index not implemented");
  } else if (ret < 0) {
    LOG_ERROR("Failed to dump in streamer");
  }

  return ret;
}

int MixedStreamerReducer::read_vec(size_t source_streamer_index,
                                   const IndexProvider::Pointer &provider,
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

  if (!provider) {
    LOG_ERROR("Source provider is null, index=%zu", source_streamer_index);
    return IndexError_Runtime;
  }
  IndexProvider::Iterator::Pointer iterator = provider->create_iterator();
  if (!iterator) {
    LOG_ERROR("Failed to create source provider iterator, index=%zu",
              source_streamer_index);
    return IndexError_Runtime;
  }

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

    const void *vector_data = iterator->data();
    if (!vector_data) {
      LOG_ERROR("Failed to read source vector, index=%zu key=%zu",
                source_streamer_index, static_cast<size_t>(iterator->key()));
      return IndexError_ReadData;
    }

    std::vector<uint8_t> bytes;
    if (need_revert) {
      std::string new_vector;
      if (reformer->revert(vector_data, source_streamer_query_meta,
                           &new_vector) != 0) {
        LOG_ERROR("Failed to revert the vector");
        return IndexError_Runtime;
      }
      bytes.resize(new_vector.size());
      memcpy(bytes.data(), new_vector.data(), bytes.size());
    } else {
      // TODO: eliminate the copy
      bytes.resize(provider->element_size());
      memcpy(bytes.data(), vector_data, bytes.size());
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

int MixedStreamerReducer::reduce_with_builder(const IndexFilter &filter) {
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.reserve(streamers_.size());

  for (size_t i = 0; i < streamers_.size(); ++i) {
    MergedProviderIndexHolder::Source source;
    source.owner = streamers_[i];
    source.reformer = source_streamers_reformers_[i];
    source.provider_meta = IndexQueryMeta{streamers_[i]->meta().data_type(),
                                          streamers_[i]->meta().dimension()};
    // A builder consumes the original input meta. This intentionally matches
    // the old read_vec() builder path, which reverted whenever one existed.
    source.need_revert = source.reformer != nullptr;
    sources.emplace_back(std::move(source));
  }

  auto holder = std::make_shared<MergedProviderIndexHolder>(
      original_query_meta_, std::move(sources));
  int ret = holder->init(filter, stop_flag_);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize merged provider holder, ret=%d", ret);
    return ret;
  }

  stats_.set_filtered_count(holder->filtered_count());
  merged_holder_ = holder;

  AILEGO_DEFER([&]() { holder->set_stop_flag(nullptr); });
  IndexHolder::Pointer target_holder = holder;
  // Only IVF has been adapted to propagate source read failures during dump.
  // Other builders retain an owned multipass snapshot, as before, so their
  // dump paths never depend on a source provider or its deferred error state.
  if (target_builder_->name() != "IVFBuilder") {
    switch (holder->data_type()) {
      case IndexMeta::DataType::DT_FP32:
        ret = MaterializeMergedInput<IndexMeta::DataType::DT_FP32, float>(
            holder, &target_holder);
        break;
      case IndexMeta::DataType::DT_FP16:
        ret = MaterializeMergedInput<IndexMeta::DataType::DT_FP16,
                                     ailego::Float16>(holder, &target_holder);
        break;
      case IndexMeta::DataType::DT_INT8:
        ret = MaterializeMergedInput<IndexMeta::DataType::DT_INT8, int8_t>(
            holder, &target_holder);
        break;
      default:
        ret = IndexError_Unsupported;
        break;
    }
    if (ret != 0) {
      return ret;
    }
  }
  return this->IndexBuild(std::move(target_holder));
}

int MixedStreamerReducer::IndexBuild(IndexHolder::Pointer target_holder) {
  if (target_builder_converter_) {
    int ret = core::IndexConverter::TrainAndTransform(target_builder_converter_,
                                                      target_holder);
    if (ret != 0) {
      LOG_ERROR("Failed to convert target holder, ret=%d", ret);
      return merged_holder_ && merged_holder_->status() != 0
                 ? merged_holder_->status()
                 : ret;
    }
    target_holder = target_builder_converter_->result();
    if (!target_holder) {
      LOG_ERROR("Target builder converter returned a null holder");
      return core::IndexError_Runtime;
    }
  }
  int ret = target_builder_->train(target_holder);
  if (merged_holder_ && merged_holder_->status() != 0) {
    return merged_holder_->status();
  }
  if (ret != 0) {
    LOG_ERROR("Failed to train target builder, ret=%d", ret);
    return ret;
  }
  ret = target_builder_->build(target_holder);
  if (merged_holder_ && merged_holder_->status() != 0) {
    return merged_holder_->status();
  }
  if (ret != 0) {
    LOG_ERROR("Failed to build target index, ret=%d", ret);
    return ret;
  }
  return 0;
}

INDEX_FACTORY_REGISTER_STREAMER_REDUCER_ALIAS(MixedStreamerReducer,
                                              MixedStreamerReducer);

}  // namespace core
}  // namespace zvec
