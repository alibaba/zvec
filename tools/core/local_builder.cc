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

#include <libgen.h>
#include <signal.h>
#include <iostream>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/utility/time_helper.h>
#include "algorithm/flat/flat_utility.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "zvec/ailego/logger/logger.h"
#include "zvec/core/framework/index_dumper.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_logger.h"
#include "zvec/core/framework/index_plugin.h"
#include "zvec/core/framework/index_reformer.h"
#include "zvec/core/framework/index_streamer.h"
#include "index_meta_helper.h"
#include "meta_segment_common.h"
#include "vecs_index_holder.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <yaml-cpp/yaml.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

using namespace std;
using namespace zvec::core;
using namespace zvec;

bool g_disable_id_map = false;

enum RetrievalMode { RM_UNDEFINED = 0, RM_DENSE = 1, RM_SPARSE = 2 };

VecsIndexHolder::Pointer holder;
VecsIndexSparseHolder::Pointer sparse_holder;

bool stop_now = false;
void stop(int signo) {
  if (stop_now) {
    exit(signo);
  }
  stop_now = true;
  cout << "\rTrying to stop. press [Ctrl+C] again kill immediately." << endl
       << flush;
  if (holder) {
    holder->stop();
  }
}

void usage(void) {
  cout << "Usage: local_builder CONFIG.yaml [plugin file path]" << endl;
}

bool prepare_params(YAML::Node &&config_params, ailego::Params &params) {
  cout << "Parse params as blow:" << endl;
  for (auto it = config_params.begin(); it != config_params.end(); ++it) {
    string tag = it->second.Tag();
    if (tag == "tag:yaml.org,2002:int") {
      int64_t val = it->second.as<int64_t>();
      params.set(it->first.as<string>(), val);
      cout << it->first.as<string>() << "=" << val << endl;
    } else if (tag == "tag:yaml.org,2002:float") {
      float val = it->second.as<float>();
      params.set(it->first.as<string>(), val);
      cout << it->first.as<string>() << "=" << val << endl;
    } else if (tag == "tag:yaml.org,2002:bool") {
      bool val = it->second.as<bool>();
      params.set(it->first.as<string>(), val);
      cout << it->first.as<string>() << "=" << val << endl;
    } else {
      if (it->second.IsScalar()) {
        string val = it->second.as<string>();
        params.set(it->first.as<string>(), val);
        cout << it->first.as<string>() << "=" << val << endl;
      } else if (it->second.IsMap()) {
        ailego::Params sub_params;
        auto sub_node = it->second;
        if (!prepare_params(std::move(sub_node), sub_params)) {
          LOG_ERROR("parse params error with key[%s]",
                    it->first.as<string>().c_str());
          return false;
        }
        params.set(it->first.as<string>(), sub_params);
      }
    }
  }
  return true;
}

bool check_config(YAML::Node &config_root) {
  auto common = config_root["BuilderCommon"];
  if (!common) {
    LOG_ERROR("Can not find [BuilderClass] in config");
    return false;
  }
  if (!common["BuilderClass"]) {
    LOG_ERROR("Can not find [BuilderClass] in config");
    return false;
  }
  if (!common["BuildFile"]) {
    LOG_ERROR("Can not find [BuildFile] in config");
    return false;
  }
  if (common["NeedTrain"] && common["NeedTrain"].as<bool>()) {
    if (!common["TrainFile"]) {
      LOG_ERROR("Can not find [TrainFile] in config");
      return false;
    }
  }
  if (common["UseTrainer"]) {
    if (!common["TrainerIndexPath"]) {
      LOG_ERROR("Can not find [TrainerIndexPath] in config");
      return false;
    }
    if (!config_root["TrainerParams"]) {
      LOG_ERROR("Can not find [TrainerParams] in config");
      return false;
    }
  }
  if (!common["DumpPath"]) {
    LOG_ERROR("Can not find [DumpPath] in config");
    return false;
  }
  if (!config_root["BuilderParams"]) {
    LOG_ERROR("Can not find [BuilderParams] in config");
    return false;
  }
  return true;
}

static inline size_t AlignSize(size_t size) {
  return (size + 0x1F) & (~0x1F);
}

bool dump_meta_segment(const IndexDumper::Pointer &dumper,
                       const std::string &segment_id, const void *data,
                       size_t size, size_t &writes) {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return false;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return false;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return false;
  }

  writes = len + padding_size;

  return true;
}

int dump_taglist(IndexDumper::Pointer dumper, size_t num_vecs,
                 const void *key_base, const void *taglist_data,
                 uint64_t taglist_size) {
  TagListHeader taglist_header;

  taglist_header.num_vecs = num_vecs;

  size_t total_writes;

  bool ret =
      dump_meta_segment(dumper, TAGLIST_HEADER_SEGMENT_NAME, &taglist_header,
                        sizeof(TagListHeader), total_writes);
  if (ret == false) {
    LOG_ERROR("dump taglist meta failed");
    return IndexError_WriteData;
  }

  ret = dump_meta_segment(dumper, TAGLIST_KEY_SEGMENT_NAME, key_base,
                          num_vecs * sizeof(uint64_t), total_writes);
  if (ret == false) {
    LOG_ERROR("dump taglist key failed");
    return IndexError_WriteData;
  }

  ret = dump_meta_segment(dumper, TAGLIST_DATA_SEGMENT_NAME, taglist_data,
                          taglist_size, total_writes);
  if (ret == false) {
    LOG_ERROR("dump taglist data failed");
    return IndexError_WriteData;
  }

  return 0;
}

int do_build_sparse_by_streamer(IndexStreamer::Pointer &streamer,
                                uint32_t thread_count) {
  int ret;
  ailego::ThreadPool pool(thread_count, false);
  std::atomic<size_t> finished{0};
  int errcode = 0;
  std::mutex mutex;
  std::atomic_bool error{false};
  std::condition_variable cond{};

  auto meta = streamer->meta();
  IndexReformer::Pointer reformer;
  if (!meta.reformer_name().empty()) {
    reformer = IndexFactory::CreateReformer(meta.reformer_name());
    if (!reformer) {
      LOG_ERROR("Failed to create reformer %s", meta.reformer_name().c_str());
      return IndexError_NoExist;
    }
    reformer->init(meta.reformer_params());
  }

  IndexQueryMeta qmeta(sparse_holder->data_type());
  uint32_t keep_docs = sparse_holder->count() - sparse_holder->start_cursor();


  std::function<int(uint64_t, const uint32_t, const uint32_t *, const void *,
                    const IndexQueryMeta &, IndexContext::Pointer &)>
      add_to_streamer_sparse = [&](uint64_t pkey, const uint32_t sparse_count,
                                   const uint32_t *sparse_indices,
                                   const void *sparse_query,
                                   const IndexQueryMeta &qmeta,
                                   IndexContext::Pointer &context) -> int {
    return streamer->add_impl(pkey, sparse_count, sparse_indices, sparse_query,
                              qmeta, context);
  };
  if (g_disable_id_map) {
    add_to_streamer_sparse = [&](uint64_t pkey, const uint32_t sparse_count,
                                 const uint32_t *sparse_indices,
                                 const void *sparse_query,
                                 const IndexQueryMeta &qmeta,
                                 IndexContext::Pointer &context) -> int {
      return streamer->add_with_id_impl(static_cast<uint32_t>(pkey),
                                        sparse_count, sparse_indices,
                                        sparse_query, qmeta, context);
    };
  }

  auto do_build = [&](size_t idx) {
    AILEGO_DEFER([&]() {
      std::lock_guard<std::mutex> latch(mutex);
      cond.notify_one();
    });
    auto ctx = streamer->create_context();
    if (!ctx) {
      if (!error.exchange(true)) {
        LOG_ERROR("Failed to create streamer context");
        errcode = IndexError_NoMemory;
      }
      return;
    }
    std::string ovec;
    IndexQueryMeta ometa;
    for (uint32_t id = idx; id < sparse_holder->count() && !stop_now;
         id += thread_count) {
      uint64_t key = sparse_holder->get_key(id);
      if (reformer) {
        std::string new_vec;
        IndexQueryMeta new_meta;
        ret = reformer->convert(sparse_holder->get_sparse_count(id),
                                sparse_holder->get_sparse_indices(id),
                                sparse_holder->get_sparse_data(id), qmeta,
                                &new_vec, &new_meta);
        if (ret != 0) {
          LOG_ERROR("Failed to convert sparse vector for %s",
                    IndexError::What(ret));
          errcode = ret;
          return;
        }
        ret = add_to_streamer_sparse(key, sparse_holder->get_sparse_count(id),
                                     sparse_holder->get_sparse_indices(id),
                                     new_vec.data(), new_meta, ctx);
      } else {
        ret = add_to_streamer_sparse(key, sparse_holder->get_sparse_count(id),
                                     sparse_holder->get_sparse_indices(id),
                                     sparse_holder->get_sparse_data(id), qmeta,
                                     ctx);
      }

      if (ailego_unlikely(ret != 0)) {
        if (!error.exchange(true)) {
          LOG_ERROR("streamer all_impl failed\n");
          errcode = ret;
        }
        return;
      }
      if (id >= keep_docs) {
        ret =
            streamer->remove_impl(sparse_holder->get_key(id - keep_docs), ctx);
        if (ailego_unlikely(ret != 0)) {
          if (!error.exchange(true)) {
            LOG_ERROR("streamer remove_impl failed\n");
            errcode = ret;
          }
          return;
        }
      }
      finished++;
    }
    return;
  };

  for (size_t i = 0; i < pool.count(); ++i) {
    pool.execute(do_build, i);
  }

  while (!pool.is_finished()) {
    std::unique_lock<std::mutex> lk(mutex);
    cond.wait_until(
        lk, std::chrono::system_clock::now() + std::chrono::seconds(15));
    if (error.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to build index while waiting finish");
      return errcode;
    }
    LOG_INFO("Built cnt %zu, finished percent %.3f%%", finished.load(),
             finished.load() * 100.0f / sparse_holder->count());
  }
  if (error.load(std::memory_order_acquire)) {
    LOG_ERROR("Failed to build index while waiting finish");
    return errcode;
  }
  pool.wait_finish();

  return 0;
}

int build_sparse_by_streamer(IndexStreamer::Pointer &streamer,
                             YAML::Node &config_common) {
  if (!config_common["IndexPath"]) {
    LOG_ERROR("Miss params IndexPath for Streamer");
    return IndexError_InvalidArgument;
  }
  string path = config_common["IndexPath"].as<string>();

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  if (!storage) {
    LOG_ERROR("Failed to create storage");
    return IndexError_NoExist;
  }
  ailego::Params params;
  int ret = storage->init(params);
  if (ret != 0) {
    LOG_ERROR("Storage Failed init");
    return IndexError_Runtime;
  }
  ret = storage->open(path, true);
  if (ret != 0) {
    LOG_ERROR("Storage Failed to open");
    return IndexError_Runtime;
  }
  ret = streamer->open(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to open storage");
    return IndexError_Runtime;
  }

  size_t thread_count = config_common["ThreadCount"]
                            ? config_common["ThreadCount"].as<uint64_t>()
                            : std::thread::hardware_concurrency();

  auto meta = streamer->meta();

  LOG_DEBUG("thread count: %zu, retrieval_mode: sparse", thread_count);
  do_build_sparse_by_streamer(streamer, thread_count);

  return 0;
}

int do_build_by_streamer(IndexStreamer::Pointer &streamer,
                         uint32_t thread_count, RetrievalMode retrieval_mode) {
  int ret;
  ailego::ThreadPool pool(thread_count, false);
  std::atomic<size_t> finished{0};
  int errcode = 0;
  std::mutex mutex;
  std::atomic_bool error{false};
  std::condition_variable cond{};

  auto meta = streamer->meta();
  IndexReformer::Pointer reformer;
  if (!meta.reformer_name().empty()) {
    if (retrieval_mode != RM_DENSE) {
      LOG_ERROR("Reformer not supported");
      return IndexError_Runtime;
    } else {
      reformer = IndexFactory::CreateReformer(meta.reformer_name());
      if (!reformer) {
        LOG_ERROR("Failed to create reformer %s", meta.reformer_name().c_str());
        return IndexError_NoExist;
      }
      reformer->init(meta.reformer_params());
    }
  }

  IndexQueryMeta qmeta(holder->data_type(), holder->dimension());
  uint32_t keep_docs = holder->count() - holder->start_cursor();

  std::function<int(uint64_t, const void *, const IndexQueryMeta &,
                    IndexContext::Pointer &)>
      add_to_streamer = [&](uint64_t pkey, const void *query,
                            const IndexQueryMeta &qmeta,
                            IndexContext::Pointer &context) -> int {
    return streamer->add_impl(pkey, query, qmeta, context);
  };
  if (g_disable_id_map) {
    add_to_streamer = [&](uint64_t pkey, const void *query,
                          const IndexQueryMeta &qmeta,
                          IndexStreamer::Context::Pointer &context) -> int {
      return streamer->add_with_id_impl(static_cast<uint32_t>(pkey), query,
                                        qmeta, context);
    };
  }

  auto do_build = [&](size_t idx) {
    AILEGO_DEFER([&]() {
      std::lock_guard<std::mutex> latch(mutex);
      cond.notify_one();
    });
    auto ctx = streamer->create_context();
    if (!ctx) {
      if (!error.exchange(true)) {
        LOG_ERROR("Failed to create streamer context");
        errcode = IndexError_NoMemory;
      }
      return;
    }
    std::string ovec;
    IndexQueryMeta ometa;
    for (uint32_t id = idx; id < holder->count() && !stop_now;
         id += thread_count) {
      uint64_t key = holder->get_key(id);
      if (retrieval_mode == RM_DENSE) {
        if (reformer) {
          ret = reformer->convert(holder->get_vector(id), qmeta, &ovec, &ometa);
          if (ret != 0) {
            LOG_ERROR("Failed to convert vector for %s", IndexError::What(ret));
            errcode = ret;
            return;
          }
          ret = add_to_streamer(key, ovec.data(), ometa, ctx);
        } else {
          ret = add_to_streamer(key, holder->get_vector(id), qmeta, ctx);
        }
      } else {
        LOG_ERROR("Retrieval mode not supported");
        errcode = IndexError_Unsupported;
        return;
      }

      if (ailego_unlikely(ret != 0)) {
        if (!error.exchange(true)) {
          LOG_ERROR("streamer add_impl failed");
          errcode = ret;
        }
        return;
      }
      if (id >= keep_docs) {
        ret = streamer->remove_impl(holder->get_key(id - keep_docs), ctx);
        if (ailego_unlikely(ret != 0)) {
          if (!error.exchange(true)) {
            LOG_ERROR("streamer remove_impl failed");
            errcode = ret;
          }
          return;
        }
      }
      finished++;
    }
    return;
  };

  for (size_t i = 0; i < pool.count(); ++i) {
    pool.execute(do_build, i);
  }

  while (!pool.is_finished()) {
    std::unique_lock<std::mutex> lk(mutex);
    cond.wait_until(
        lk, std::chrono::system_clock::now() + std::chrono::seconds(15));
    if (error.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to build index while waiting finish");
      return errcode;
    }
    LOG_INFO("Built cnt %zu, finished percent %.3f%%", finished.load(),
             finished.load() * 100.0f / holder->count());
  }
  if (error.load(std::memory_order_acquire)) {
    LOG_ERROR("Failed to build index while waiting finish");
    return errcode;
  }
  pool.wait_finish();

  return 0;
}

int build_by_streamer(IndexStreamer::Pointer &streamer,
                      YAML::Node &config_common) {
  if (!config_common["IndexPath"]) {
    LOG_ERROR("Miss params IndexPath for Streamer");
    return IndexError_InvalidArgument;
  }
  string path = config_common["IndexPath"].as<string>();

  ailego::File::RemovePath(path);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  if (!storage) {
    LOG_ERROR("Failed to create storage");
    return IndexError_NoExist;
  }
  ailego::Params params;
  int ret = storage->init(params);
  if (ret != 0) {
    LOG_ERROR("Storage Failed init");
    return IndexError_Runtime;
  }
  ret = storage->open(path, true);
  if (ret != 0) {
    LOG_ERROR("Storage Failed to open");
    return IndexError_Runtime;
  }
  ret = streamer->open(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to open storage");
    return IndexError_Runtime;
  }

  size_t thread_count = config_common["ThreadCount"]
                            ? config_common["ThreadCount"].as<uint64_t>()
                            : std::thread::hardware_concurrency();

  auto meta = streamer->meta();

  RetrievalMode retrieval_mode = RM_UNDEFINED;
  if (meta.dimension() > 0) {
    retrieval_mode = RM_DENSE;
  } else {
    retrieval_mode = RM_SPARSE;
  }

  LOG_DEBUG("thread count: %zu, retrieval mode: %s", thread_count,
            retrieval_mode == 1 ? "Dense" : "Sparse");
  do_build_by_streamer(streamer, thread_count, retrieval_mode);

  return 0;
}

IndexSparseHolder::Pointer convert_sparse_holder(
    const std::string &name, const ailego::Params &params,
    VecsIndexSparseHolder::Pointer &in_holder, IndexMeta &index_meta) {
  IndexSparseHolder::Pointer cast_holder =
      std::dynamic_pointer_cast<IndexSparseHolder>(in_holder);
  if (name.empty()) {
    return cast_holder;
  }

  IndexConverter::Pointer converter = IndexFactory::CreateConverter(name);
  if (!converter) {
    LOG_ERROR("Failed to create sparse converter %s", name.c_str());
    return IndexSparseHolder::Pointer();
  }

  int ret = converter->init(in_holder->index_meta(), params);
  if (ret != 0) {
    LOG_ERROR("Failed to init converter %d", ret);
    return IndexSparseHolder::Pointer();
  }

  ret = converter->train(cast_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train sparse converter %d", ret);
    return IndexSparseHolder::Pointer();
  }

  ret = converter->transform(cast_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to transform converter %d", ret);
    return IndexSparseHolder::Pointer();
  }

  index_meta = converter->meta();

  return converter->sparse_result();
}

IndexHolder::Pointer convert_holder(const std::string &name,
                                    const ailego::Params &params,
                                    VecsIndexHolder::Pointer &in_holder,
                                    IndexMeta &index_meta) {
  IndexHolder::Pointer cast_holder =
      std::dynamic_pointer_cast<IndexHolder>(in_holder);
  if (name.empty()) {
    return cast_holder;
  }

  IndexConverter::Pointer converter = IndexFactory::CreateConverter(name);
  if (!converter) {
    LOG_ERROR("Failed to create converter %s", name.c_str());
    return IndexHolder::Pointer();
  }

  int ret = converter->init(in_holder->index_meta(), params);
  if (ret != 0) {
    LOG_ERROR("Failed to init converter %d", ret);
    return IndexHolder::Pointer();
  }

  ret = converter->train(cast_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train converter %d", ret);
    return IndexHolder::Pointer();
  }

  ret = converter->transform(cast_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to transform converter %d", ret);
    return IndexHolder::Pointer();
  }

  index_meta = converter->meta();

  return converter->result();
}

int do_build_sparse(YAML::Node &config_root, YAML::Node &config_common) {
  string build_file = config_common["BuildFile"].as<string>();
  VecsIndexSparseHolder::Pointer build_holder(new VecsIndexSparseHolder);
  if (!build_holder->load(build_file)) {
    LOG_ERROR("Load input error: %s", build_file.c_str());
    return -1;
  }
  IndexMeta meta;
  meta = build_holder->index_meta();

  std::string metric_name;
  ailego::Params metric_params;
  if (config_common["MetricName"] &&
      !config_common["MetricName"].as<string>().empty()) {
    metric_name = config_common["MetricName"].as<string>();
    if (config_root["MetricParams"] &&
        !prepare_params(config_root["MetricParams"], metric_params)) {
      LOG_ERROR("Failed to prepare metric params");
      return -1;
    }
    build_holder->set_metric(metric_name, metric_params);
    meta.set_metric(metric_name, 0, metric_params);
  }

  string converter_name;
  ailego::Params converter_params;
  if (config_common["ConverterName"] &&
      !config_common["ConverterName"].as<string>().empty()) {
    converter_name = config_common["ConverterName"].as<string>();
    if (config_root["ConverterParams"] &&
        !prepare_params(config_root["ConverterParams"], converter_params)) {
      LOG_ERROR("Failed to prepare converter params");
      return -1;
    }
  }

  if (config_common["MaxDocs"] && config_common["MaxDocs"].as<uint32_t>()) {
    auto max_docs = config_common["MaxDocs"].as<uint32_t>();
    build_holder->set_max_doc_count(max_docs);
  }
  if (config_common["KeepDocs"] && config_common["KeepDocs"].as<uint32_t>()) {
    auto keep_docs = config_common["KeepDocs"].as<uint32_t>();
    if (keep_docs < build_holder->count()) {
      build_holder->set_start_cursor(build_holder->count() - keep_docs);
    }
  }

  // Create a Builder
  string builder_class = config_common["BuilderClass"].as<string>();
  IndexStreamer::Pointer streamer;
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder(builder_class.c_str());
  if (!builder) {
    streamer = IndexFactory::CreateStreamer(builder_class.c_str());
  }
  if (!builder && !streamer) {
    LOG_ERROR("Failed to create builder %s", builder_class.c_str());
    return -1;
  }
  cout << "Created builder " << builder_class << endl;

  IndexSparseHolder::Pointer cv_build_holder = convert_sparse_holder(
      converter_name, converter_params, build_holder, meta);
  if (!cv_build_holder) {
    LOG_ERROR("Convert holder failed.");
    return -1;
  }

  ailego::Params params;
  if (!prepare_params(config_root["BuilderParams"], params)) {
    LOG_ERROR("Failed to prepare params");
    return -1;
  }

  // INIT
  int ret =
      builder ? builder->init(meta, params) : streamer->init(meta, params);
  if (ret < 0) {
    LOG_ERROR("Failed to init builder, ret=%d", ret);
    return -1;
  }
  ailego::ElapsedTime timer;

  // TRAIN
  if (builder && config_common["NeedTrain"] &&
      config_common["NeedTrain"].as<bool>()) {
    string train_file = config_common["TrainFile"].as<string>();
    VecsIndexSparseHolder::Pointer train_holder(new VecsIndexSparseHolder);
    if (!train_holder->load(train_file)) {
      LOG_ERROR("Load input error: %s", train_file.c_str());
      return -1;
    }

    if (!metric_name.empty()) {
      train_holder->set_metric(metric_name, metric_params);
    }

    IndexSparseHolder::Pointer cv_train_holder = convert_sparse_holder(
        converter_name, converter_params, train_holder, meta);
    if (!cv_train_holder) {
      LOG_ERROR("Convert train holder failed.");
      return -1;
    }

    std::cout << "Prepare train data done!" << std::endl;
    timer.reset();
    ret = builder->train(std::move(cv_train_holder));
    size_t train_time = timer.milli_seconds();

    if (ret < 0) {
      LOG_ERROR("Failed to train in builder, ret=%d", ret);
      return -1;
    }
    cout << "Train finished, consume " << train_time << "ms." << endl;
  } else {
    cout << "Skip train procedure" << endl;
  }

  // BUILD
  sparse_holder = build_holder;
  signal(SIGINT, stop);
  timer.reset();
  if (builder != nullptr) {
    ret = builder->build(std::move(cv_build_holder));
  } else {
    ret = build_sparse_by_streamer(streamer, config_common);
  }
  size_t build_time = timer.milli_seconds();
  if (ret < 0) {
    LOG_ERROR("Failed to build in builder, ret=%d", ret);
    return -1;
  }
  cout << "Build finished, consume " << build_time << "ms." << endl;
  signal(SIGINT, SIG_DFL);

  // DUMP
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  if (!dumper) {
    LOG_ERROR("Failed to create FileDumper.");
    return -1;
  }
  string dump_prefix = config_common["DumpPath"].as<string>();
  ret = dumper->create(dump_prefix);
  if (ret != 0) {
    LOG_ERROR("Failed to create in dumper, ret=%d", ret);
    return -1;
  }
  timer.reset();
  ret = streamer ? streamer->dump(dumper) : builder->dump(dumper);
  size_t dump_time = timer.milli_seconds();
  if (ret == IndexError_NotImplemented) {
    LOG_WARN("Dump index not implemented");
  } else if (ret < 0) {
    LOG_ERROR("Failed to dump in builder, ret=%d", ret);
    return -1;
  }

  if (build_holder->has_taglist()) {
    size_t taglist_size{0};
    const void *taglist_data = build_holder->get_taglist_data(taglist_size);
    const void *key_base = build_holder->get_key_base();

    dump_taglist(dumper, build_holder->get_num_vecs(), key_base, taglist_data,
                 taglist_size);
  }

  ret = dumper->close();
  if (ret != 0) {
    LOG_ERROR("Dumper failed to close, ret=%d", ret);
    return -1;
  }
  std::cout << "Dump to [" << dump_prefix << "] finished, consume " << dump_time
            << "ms." << std::endl;

  if (builder) {
    auto &stats =
        reinterpret_cast<const IndexBuilder *>(builder.get())->stats();
    std::cout << "STATS: \n\tTrained count[" << stats.trained_count()
              << "]\n\tBuilt count[" << stats.built_count()
              << "]\n\tDump count[" << stats.dumped_count()
              << "]\n\tDiscarded count[" << stats.discarded_count() << "]\n";
  } else {
    auto &stats = streamer->stats();
    std::cout << "STATS: \n\tTrained count[" << 0 << "]\n\tBuilt count["
              << stats.added_count() << "]\n\tDump size ["
              << stats.dumped_size() << "]\n\tDiscarded count["
              << stats.discarded_count() << "]\n";
  }

  // CLEANUP
  builder ? builder->cleanup() : streamer->cleanup();

  return 0;
}

int do_build(YAML::Node &config_root, YAML::Node &config_common) {
  string build_file = config_common["BuildFile"].as<string>();
  VecsIndexHolder::Pointer build_holder(new VecsIndexHolder);
  if (!build_holder->load(build_file)) {
    LOG_ERROR("Load input error: %s", build_file.c_str());
    return -1;
  }
  IndexMeta meta;
  meta = build_holder->index_meta();

  std::string metric_name;
  ailego::Params metric_params;
  if (config_common["MetricName"] &&
      !config_common["MetricName"].as<string>().empty()) {
    metric_name = config_common["MetricName"].as<string>();
    if (config_root["MetricParams"] &&
        !prepare_params(config_root["MetricParams"], metric_params)) {
      LOG_ERROR("Failed to prepare metric params");
      return -1;
    }
    build_holder->set_metric(metric_name, metric_params);
    meta.set_metric(metric_name, 0, metric_params);
  }
  string converter_name;
  ailego::Params converter_params;
  if (config_common["ConverterName"] &&
      !config_common["ConverterName"].as<string>().empty()) {
    converter_name = config_common["ConverterName"].as<string>();
    if (config_root["ConverterParams"] &&
        !prepare_params(config_root["ConverterParams"], converter_params)) {
      LOG_ERROR("Failed to prepare converter params");
      return -1;
    }
  }
  IndexMeta::MajorOrder order = IndexMeta::MO_UNDEFINED;
  if (config_common["MajorOrder"]) {
    std::string order_str = config_common["MajorOrder"].as<string>();
    if (order_str == "row") {
      order = IndexMeta::MajorOrder::MO_ROW;
    } else {
      order = IndexMeta::MajorOrder::MO_COLUMN;
    }
  }

  if (config_common["MaxDocs"] && config_common["MaxDocs"].as<uint32_t>()) {
    auto max_docs = config_common["MaxDocs"].as<uint32_t>();
    build_holder->set_max_doc_count(max_docs);
  }
  if (config_common["KeepDocs"] && config_common["KeepDocs"].as<uint32_t>()) {
    auto keep_docs = config_common["KeepDocs"].as<uint32_t>();
    if (keep_docs < build_holder->count()) {
      build_holder->set_start_cursor(build_holder->count() - keep_docs);
    }
  }

  // Create a Builder
  string builder_class = config_common["BuilderClass"].as<string>();
  IndexStreamer::Pointer streamer;
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder(builder_class.c_str());
  if (!builder) {
    streamer = IndexFactory::CreateStreamer(builder_class.c_str());
  }
  if (!builder && !streamer) {
    LOG_ERROR("Failed to create builder %s", builder_class.c_str());
    return -1;
  }
  cout << "Created builder " << builder_class << endl;


  IndexHolder::Pointer cv_build_holder =
      convert_holder(converter_name, converter_params, build_holder, meta);
  if (!cv_build_holder) {
    LOG_ERROR("Convert holder failed.");
    return -1;
  }
  meta.set_major_order(order);
  cout << IndexMetaHelper::to_string(meta) << endl;
  cout << "Prepare data done!" << endl;

  ailego::Params params;
  if (g_disable_id_map) {
    params.set(PARAM_HNSW_STREAMER_USE_ID_MAP, false);
    params.set(PARAM_FLAT_USE_ID_MAP, false);
  }
  if (!prepare_params(config_root["BuilderParams"], params)) {
    LOG_ERROR("Failed to prepare params");
    return -1;
  }

  // INIT
  int ret =
      builder ? builder->init(meta, params) : streamer->init(meta, params);
  if (ret < 0) {
    LOG_ERROR("Failed to init builder, ret=%d", ret);
    return -1;
  }
  ailego::ElapsedTime timer;

  // TRAIN
  if (config_common["UseTrainer"] && config_common["UseTrainer"].as<bool>()) {
    ailego::Params trainer_params;
    if (!prepare_params(config_root["TrainerParams"], trainer_params)) {
      LOG_ERROR("Failed to prepare trainer params");
      return -1;
    }

    string train_index_path;
    if (config_common["TrainerIndexPath"]) {
      train_index_path = config_common["TrainerIndexPath"].as<string>();
      if (train_index_path.empty()) {
        LOG_ERROR("invalid TrainerIndexPath format");
        return -1;
      }
      cout << "Trainer index path: " << train_index_path << "\n";
    } else {
      LOG_ERROR("Need [TrainerIndexPath] config");
      return -1;
    }

    IndexTrainer::Pointer trainer =
        IndexFactory::CreateTrainer("StratifiedClusterTrainer");
    if (trainer->init(meta, trainer_params) != 0) {
      LOG_ERROR("trainer init failed");
      return -1;
    }

    if (ailego::File::IsExist(train_index_path)) {
      IndexStorage::Pointer container =
          IndexFactory::CreateStorage("MMapFileReadStorage");
      if (!container) {
        LOG_ERROR("Failed to create MMapFileReadStorage");
        return -1;
      }
      container->init(ailego::Params());
      if (container->open(train_index_path, false) != 0) {
        LOG_ERROR("MMapFileReadStorage failed to load %s",
                  train_index_path.c_str());
        return -1;
      }
      if (trainer->load(container) != 0) {
        LOG_ERROR("Trainer failed to load container");
        return -1;
      };
    } else {
      std::cout << "Prepare trainer data..." << std::endl;
      string train_file = config_common["TrainFile"].as<string>();
      VecsIndexHolder::Pointer train_holder(new VecsIndexHolder);
      if (!train_holder->load(train_file)) {
        LOG_ERROR("Load input error: %s", train_file.c_str());
        return -1;
      }
      if (!metric_name.empty()) {
        train_holder->set_metric(metric_name, metric_params);
      }

      // support fp16 convert

      IndexHolder::Pointer cv_train_holder =
          convert_holder(converter_name, converter_params, train_holder, meta);
      if (!cv_train_holder) {
        LOG_ERROR("Convert train holder failed.");
        return -1;
      }

      std::cout << "Prepare trainer data done!" << std::endl;
      std::cout << "Prepare train data!" << std::endl;

      ret = trainer->train(cv_train_holder);
      if (ret != 0) {
        LOG_ERROR("trainer train_index failed with %d", ret);
        return -1;
      }

      std::cout << "train data done!" << std::endl;
      IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
      if (!dumper) {
        LOG_ERROR("Failed to create FileDumper.");
        return -1;
      }
      if (dumper->init(ailego::Params()) != 0) {
        LOG_ERROR("Failed to init FileDumper.");
        return -1;
      }
      ret = dumper->create(train_index_path);
      if (ret != 0) {
        LOG_ERROR("Failed to create in dumper, ret=%d", ret);
        return -1;
      }
      if (trainer->dump(dumper) != 0) {
        LOG_ERROR("trainer dump_index failed");
        return -1;
      }
      dumper->close();
    }

    ret = builder->train(trainer);
    size_t train_time = timer.milli_seconds();
    if (ret < 0) {
      LOG_ERROR("Failed to train in builder, ret=%d", ret);
      return -1;
    }
    cout << "Train finished, consume " << train_time << "ms." << endl;
  } else if (builder && config_common["NeedTrain"] &&
             config_common["NeedTrain"].as<bool>()) {
    string train_file = config_common["TrainFile"].as<string>();
    VecsIndexHolder::Pointer train_holder(new VecsIndexHolder);
    if (!train_holder->load(train_file)) {
      LOG_ERROR("Load input error: %s", train_file.c_str());
      return -1;
    }

    if (!metric_name.empty()) {
      train_holder->set_metric(metric_name, metric_params);
    }
    IndexHolder::Pointer cv_train_holder =
        convert_holder(converter_name, converter_params, train_holder, meta);
    if (!cv_train_holder) {
      LOG_ERROR("Convert train holder failed.");
      return -1;
    }

    std::cout << "Prepare train data done!" << std::endl;
    timer.reset();
    ret = builder->train(std::move(cv_train_holder));
    size_t train_time = timer.milli_seconds();
    if (ret < 0) {
      LOG_ERROR("Failed to train in builder, ret=%d", ret);
      return -1;
    }
    cout << "Train finished, consume " << train_time << "ms." << endl;
  } else {
    cout << "Skip train procedure" << endl;
  }

  // BUILD
  holder = build_holder;
  signal(SIGINT, stop);
  timer.reset();
  if (builder != nullptr) {
    ret = builder->build(std::move(cv_build_holder));
  } else {
    std::string retrieval_mode = "dense";
    if (meta.dimension() > 0) {
      retrieval_mode = "sparse";
    } else {
      retrieval_mode = "dense";
    }

    ret = build_by_streamer(streamer, config_common);
  }
  size_t build_time = timer.milli_seconds();
  if (ret < 0) {
    LOG_ERROR("Failed to build in builder, ret=%d", ret);
    return -1;
  }
  cout << "Build finished, consume " << build_time << "ms." << endl;
  signal(SIGINT, SIG_DFL);

  // DUMP
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  if (!dumper) {
    LOG_ERROR("Failed to create FileDumper.");
    return -1;
  }
  string dump_prefix = config_common["DumpPath"].as<string>();
  ret = dumper->create(dump_prefix);
  if (ret != 0) {
    LOG_ERROR("Failed to create in dumper, ret=%d", ret);
    return -1;
  }
  timer.reset();
  ret = streamer ? streamer->dump(dumper) : builder->dump(dumper);
  size_t dump_time = timer.milli_seconds();
  if (ret == IndexError_NotImplemented) {
    LOG_WARN("Dump index not implemented");
  } else if (ret < 0) {
    LOG_ERROR("Failed to dump in builder, ret=%d", ret);
    return -1;
  }

  if (build_holder->has_taglist()) {
    size_t taglist_size{0};
    const void *taglist_data = build_holder->get_taglist_data(taglist_size);
    const void *key_base = build_holder->get_key_base();

    dump_taglist(dumper, build_holder->get_num_vecs(), key_base, taglist_data,
                 taglist_size);
  }

  ret = dumper->close();
  if (ret != 0) {
    LOG_ERROR("Dumper failed to close, ret=%d", ret);
    return -1;
  }
  std::cout << "Dump to [" << dump_prefix << "] finished, consume " << dump_time
            << "ms." << std::endl;

  if (builder) {
    auto &stats =
        reinterpret_cast<const IndexBuilder *>(builder.get())->stats();
    std::cout << "STATS: \n\tTrained count[" << stats.trained_count()
              << "]\n\tBuilt count[" << stats.built_count()
              << "]\n\tDump count[" << stats.dumped_count()
              << "]\n\tDiscarded count[" << stats.discarded_count() << "]\n";
  } else {
    auto &stats = streamer->stats();
    std::cout << "STATS: \n\tTrained count[" << 0 << "]\n\tBuilt count["
              << stats.added_count() << "]\n\tDump size ["
              << stats.dumped_size() << "]\n\tDiscarded count["
              << stats.discarded_count() << "]\n";
  }

  // CLEANUP
  builder ? builder->cleanup() : streamer->cleanup();

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    usage();
    return -1;
  }
  IndexPluginBroker broker;
  std::string error;
  for (int i = 2; i < argc; ++i) {
    if (!broker.emplace(argv[i], &error)) {
      LOG_ERROR("Failed to load plugin: %s (%s)", argv[i], error.c_str());
      return -1;
    }
  }
  YAML::Node config_root;
  try {
    config_root = YAML::LoadFile(argv[1]);
  } catch (...) {
    LOG_ERROR("Load YAML file[%s] failed!", argv[1]);
    return -1;
  }
  if (!check_config(config_root)) {
    return -1;
  }
  auto config_common = config_root["BuilderCommon"];

  map<string, int> LOG_LEVEL = {{"debug", IndexLogger::LEVEL_DEBUG},
                                {"info", IndexLogger::LEVEL_INFO},
                                {"warn", IndexLogger::LEVEL_WARN},
                                {"error", IndexLogger::LEVEL_ERROR},
                                {"fatal", IndexLogger::LEVEL_FATAL}};

  string log_level = config_common["LogLevel"]
                         ? config_common["LogLevel"].as<string>()
                         : "debug";

  transform(log_level.begin(), log_level.end(), log_level.begin(), ::tolower);
  if (LOG_LEVEL.find(log_level) != LOG_LEVEL.end()) {
    IndexLoggerBroker::SetLevel(LOG_LEVEL[log_level]);
    zvec::ailego::LoggerBroker::SetLevel(LOG_LEVEL[log_level]);
  }

  RetrievalMode retrieval_mode{RM_DENSE};
  if (config_common["RetrievalMode"]) {
    std::string retrieval_mode_str =
        config_common["RetrievalMode"].as<string>();
    if (retrieval_mode_str == "dense") {
      retrieval_mode = RM_DENSE;
    } else if (retrieval_mode_str == "sparse") {
      retrieval_mode = RM_SPARSE;
    }
  }

  if (config_common["DisableIdMap"]) {
    g_disable_id_map = config_common["DisableIdMap"].as<bool>();
    if (g_disable_id_map) {
      cout << "Disable ID map" << endl;
    } else {
      cout << "Enable ID map" << endl;
    }
  }

  if (retrieval_mode == RM_SPARSE) {
    return do_build_sparse(config_root, config_common);
  } else {
    return do_build(config_root, config_common);
  }

  return 0;
}
