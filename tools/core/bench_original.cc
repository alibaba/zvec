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

#include <signal.h>
#include <iostream>
#include <ailego/container/bitmap.h>
#include <ailego/utility/time_helper.h>
#include "framework/index_plugin.h"
#include "interface/index_factory.h"
#include "interface/index_param.h"
#include "zvec/ailego/utility/string_helper.h"
#include "bench_result.h"
#include "filter_result_cache.h"
#include "flow.h"
#include "txt_input_reader.h"

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
using namespace zvec::ailego;

using Flow = Flow;

static bool g_debug_mode = 0;

//------------------------------------------------------------
// Bench
//------------------------------------------------------------
enum RetrievalMode { RM_UNDEFINED = 0, RM_DENSE = 1, RM_SPARSE = 2 };

enum FilterMode { FM_UNDEFINED = 0, FM_NONE = 1, FM_TAG = 2 };

template <typename T>
class Bench {
 public:
  Bench(size_t threads, size_t bench_secs, size_t batch_count,
        RetrievalMode &retrieval_mode, FilterMode filter_mode)
      : threads_(threads),
        bench_secs_(bench_secs),
        batch_count_(batch_count),
        retrieval_mode_{retrieval_mode},
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>(false);
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, false);
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
    }
  }

  static void stop(int signo) {
    if (STOP_NOW) {
      exit(signo);
    }
    STOP_NOW = true;
    cout << "\rTrying to stop. press [Ctrl+C] again kill immediately." << endl
         << flush;
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;
    vector<vector<T>> queries;
    vector<SparseData<T>> sparse_data;
    vector<vector<uint64_t>> taglists;

    if (!reader.load_query(query_file, first_sep, second_sep, queries,
                           sparse_data, taglists)) {
      cerr << "Load query error" << endl;
      return false;
    }

    if (batch_count_ == 1) {
      batch_queries_ = queries;

      for (size_t i = 0; i < sparse_data.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(sparse_data[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(sparse_data[i].indices);
        batch_sparse_features_.push_back(sparse_data[i].features);
      }

      for (size_t i = 0; i < taglists.size(); ++i) {
        vector<vector<uint64_t>> new_taglists;
        new_taglists.push_back(taglists[i]);

        batch_taglists_.push_back(std::move(new_taglists));
      }
    } else {
      size_t num_batch = (queries.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<T> batch_query;
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;
        vector<vector<uint64_t>> batch_taglists;

        for (size_t i = 0; i < batch_count_; ++i) {
          for (size_t k = 0; k < queries[idx].size(); ++k) {
            batch_query.push_back(queries[idx][k]);
          }

          batch_sparse_count.push_back(sparse_data[idx].count);

          for (size_t k = 0; k < sparse_data[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(sparse_data[idx].indices[k]);
          }

          for (size_t k = 0; k < sparse_data[idx].features.size(); ++k) {
            batch_sparse_feature.push_back(sparse_data[idx].features[k]);
          }

          if (taglists.size() > idx) {
            batch_taglists.push_back(taglists[idx]);
          }

          idx = (idx + 1) % queries.size();
        }

        batch_queries_.push_back(batch_query);
        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
        batch_taglists_.push_back(batch_taglists);
      }
    }

    size_t dim = queries[0].size();
    if (typeid(T) == typeid(float)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_FP32, dim);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_INT8, dim);
    } else {
      cerr << "unsupported type";
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(Flow *flower, int max_iter, int topk) {
    // Check
    if (batch_queries_.size() == 0) {
      return;
    }

    for (size_t i = 0; i < threads_; i++) {
      contexts_.emplace_back(flower->create_context());
      contexts_[i]->set_topk(topk);
      contexts_[i]->set_debug_mode(g_debug_mode);
    }

    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &Bench<T>::start_bench, flower, max_iter, &STOP_NOW);
    }

    while (!pool_->is_finished()) {
      this_thread::sleep_for(chrono::milliseconds(1));
      if (Monotime::MilliSeconds() - start_time > bench_secs_ * 1000) {
        STOP_NOW = true;
      }
    }

    pool_->wait_finish();

    bench_result_.mark_end();
    bench_result_.print();

    // for (size_t i = 0; i < threads_; i++) {
    //   if (contexts_[i]->flow_context() != nullptr) {
    //     std::cout << "context id: " << i << ": \n" <<
    //     contexts_[i]->flow_context()->searcher_context()->profiler().display();
    //   }
    // }
  }

 private:
  void start_bench(Flow *flower, size_t max_iter, const bool *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    for (; i < max_iter && !*is_stop; i += threads_) {
      int idx = i % batch_queries_.size();

      // prefilter
      FilterResultCache filter_cache;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          cerr << "query tag list not equal to one!" << endl;
          return;
        }

        int ret = filter_cache.filter(flower->id_to_tags_list(),
                                      batch_taglists_[idx][0],
                                      flower->tag_key_list());
        if (ret != 0) {
          cerr << "prefilter failed, idx: " << idx << std::endl;

          return;
        }

        auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

        contexts_[thread_index]->set_filter(filterFunc);
      }

      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (retrieval_mode_ == RM_DENSE) {
        if (batch_count_ == 1) {
          ret = do_knn_search<T>(flower, contexts_[thread_index],
                                 batch_queries_[idx]);
        } else {
          ret = do_knn_search<T>(flower, contexts_[thread_index],
                                 batch_queries_[idx], batch_count_);
        }

        if (ret != 0) {
          cerr << "Failed to knn search, ret=" << ret << endl;
          return;
        }
      } else {
        std::string mode = retrieval_mode_ == 1 ? "Dense" : "Sparse";
        cerr << "unsupported retrieval mode: " << mode << endl;
      }

      uint64_t end = Monotime::MicroSeconds();

      // Check result
      auto &result = contexts_[thread_index]->result();
      if (result.empty()) {
        cerr << "Search results is small than queries" << endl;
      }

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn search
    return flower->search_impl(query.data(), qmeta_, context);
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  shared_ptr<ThreadPool> pool_;
  vector<Flow::Context::Pointer> contexts_;

  vector<vector<T>> batch_queries_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  BenchResult bench_result_;
  RetrievalMode retrieval_mode_{RM_UNDEFINED};
  FilterMode filter_mode_{FM_NONE};
  static bool STOP_NOW;
};

template <typename T>
bool Bench<T>::STOP_NOW = false;

//------------------------------------------------------------
// Sparse Bench
//------------------------------------------------------------
template <typename T>
class SparseBench {
 public:
  SparseBench(size_t threads, size_t bench_secs, size_t batch_count,
              FilterMode filter_mode)
      : threads_(threads),
        bench_secs_(bench_secs),
        batch_count_(batch_count),
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>(false);
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, false);
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
    }
  }

  static void stop(int signo) {
    if (STOP_NOW) {
      exit(signo);
    }
    STOP_NOW = true;
    cout << "\rTrying to stop. press [Ctrl+C] again kill immediately." << endl
         << flush;
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;
    vector<vector<T>> queries;
    vector<SparseData<T>> sparse_data;
    vector<vector<uint64_t>> taglists;

    if (!reader.load_query(query_file, first_sep, second_sep, queries,
                           sparse_data, taglists)) {
      cerr << "Load query error" << endl;
      return false;
    }

    if (batch_count_ == 1) {
      for (size_t i = 0; i < sparse_data.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(sparse_data[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(sparse_data[i].indices);
        batch_sparse_features_.push_back(sparse_data[i].features);
      }
    } else {
      size_t num_batch = (queries.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;

        for (size_t i = 0; i < batch_count_; ++i) {
          batch_sparse_count.push_back(sparse_data[idx].count);

          for (size_t k = 0; k < sparse_data[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(sparse_data[idx].indices[k]);
          }

          for (size_t k = 0; k < sparse_data[idx].features.size(); ++k) {
            batch_sparse_feature.push_back(sparse_data[idx].features[k]);
          }

          idx = (idx + 1) % queries.size();
        }

        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
      }
    }

    if (typeid(T) == typeid(float)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_FP32);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_INT8);
    } else {
      cerr << "unsupported type";
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(SparseFlow *flower, int max_iter, int topk) {
    for (size_t i = 0; i < threads_; i++) {
      contexts_.emplace_back(flower->create_context());
      contexts_[i]->set_topk(topk);
      contexts_[i]->set_debug_mode(g_debug_mode);
    }

    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &SparseBench<T>::start_bench, flower, max_iter,
                     &STOP_NOW);
    }

    while (!pool_->is_finished()) {
      this_thread::sleep_for(chrono::milliseconds(1));
      if (Monotime::MilliSeconds() - start_time > bench_secs_ * 1000) {
        STOP_NOW = true;
      }
    }

    pool_->wait_finish();

    bench_result_.mark_end();
    bench_result_.print();
  }

 private:
  void start_bench(SparseFlow *flower, size_t max_iter, const bool *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    size_t sparse_query_size = batch_sparse_indices_.size();
    for (; i < max_iter && !*is_stop; i += threads_) {
      int idx = i % sparse_query_size;

      // prefilter
      FilterResultCache filter_cache;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          cerr << "query tag list not equal to one!" << endl;
          return;
        }

        int ret = filter_cache.filter(flower->id_to_tags_list(),
                                      batch_taglists_[idx][0],
                                      flower->tag_key_list());
        if (ret != 0) {
          cerr << "prefilter failed, idx: " << idx << std::endl;

          return;
        }

        auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

        contexts_[thread_index]->set_filter(filterFunc);
      }

      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (batch_count_ == 1) {
        if (batch_sparse_counts_[idx].size() != 1) {
          cerr << "Sparse count size should be 1, since batch count is 1"
               << endl;
          return;
        }
        ret = do_knn_search<T>(
            flower, contexts_[thread_index], batch_sparse_counts_[idx][0],
            batch_sparse_indices_[idx], batch_sparse_features_[idx]);
      } else {
        ret = do_knn_search<T>(flower, contexts_[thread_index],
                               batch_sparse_counts_[idx],
                               batch_sparse_indices_[idx],
                               batch_sparse_features_[idx], batch_count_);
      }

      if (ret != 0) {
        cerr << "Failed to sparse knn search, ret=" << ret << endl;
        return;
      }

      uint64_t end = Monotime::MicroSeconds();

      // Check result
      auto &result = contexts_[thread_index]->result();
      if (result.empty()) {
        cerr << "Search results is small than queries" << endl;
      }

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  // sparse search
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                const vector<uint32_t> &sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature, size_t count) {
    // Do sparse knn search
    return flower->search_impl(sparse_count.data(), sparse_indices.data(),
                               sparse_feature.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const vector<uint32_t> & /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/, size_t /*count*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const vector<uint32_t> & /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/, size_t /*count*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const vector<uint32_t> & /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/, size_t /*count*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                const uint32_t sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature) {
    // Do sparse knn search
    return flower->search_impl(sparse_count, sparse_indices.data(),
                               sparse_feature.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const uint32_t /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const uint32_t /*parse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(SparseFlow * /*flower*/,
                SparseFlow::Context::Pointer & /*context*/,
                const uint32_t /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/) {
    // Do sparse knn search
    return IndexError_Unsupported;
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  shared_ptr<ThreadPool> pool_;
  vector<SparseFlow::Context::Pointer> contexts_;

  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  FilterMode filter_mode_{FM_NONE};
  BenchResult bench_result_;
  static bool STOP_NOW;
};
template <typename T>
bool SparseBench<T>::STOP_NOW = false;

// do
bool prepare_params(YAML::Node &&config_params, Params &params) {
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
        Params sub_params;
        auto sub_node = it->second;
        if (!prepare_params(std::move(sub_node), sub_params)) {
          cerr << "parse params error with key[" << it->first.as<string>()
               << "]" << endl;
          return false;
        }
        params.set(it->first.as<string>(), sub_params);
      }
    }
  }
  return true;
}

bool check_config(YAML::Node &config_node) {
  auto common = config_node["SearcherCommon"];
  if (!common) {
    cerr << "Can not find [SearcherCommon] in config" << endl;
    return false;
  }
  if (!common["SearcherClass"] && !common["SearcherConfig"]) {
    cerr << "Can not find [SearcherClass] or [SearcherConfig] in config"
         << endl;
    return false;
  }
  if (!common["IndexPath"]) {
    cerr << "Can not find [IndexPath] in config" << endl;
    return false;
  }
  if (!common["TopK"]) {
    cerr << "Can not find [TopK] in config" << endl;
    return false;
  }
  if (!common["QueryFile"]) {
    cerr << "Can not find [QueryFile] in config" << endl;
    return false;
  }
  return true;
}

void usage(void) {
  cout << "Usage: bench CONFIG.yaml [plugin file path]" << endl;
}

bool load_index(Flow &flower, string &index_dir) {
  int ret = flower.load(index_dir);
  if (0 != ret) {
    cerr << "Flow load failed with ret " << ret << endl;
    return false;
  }
  cout << "Load index done!" << endl;
  return true;
};

int bench(std::string &query_type, size_t thread_count, size_t batch_count,
          size_t top_k, string query_file, string &first_sep,
          string &second_sep, size_t bench_secs, size_t iter_count,
          Flow &flower, string &index_dir, RetrievalMode retrieval_mode,
          FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    cerr << "filter mode can not be run in batch mode" << endl;
    return -1;
  }

  if (query_type == "float") {
    Bench<float> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                       filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else if (query_type == "int8") {
    Bench<int8_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                        filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else if (query_type == "binary") {
    Bench<uint32_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else if (query_type == "binary64") {
    Bench<uint64_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else {
    cerr << "Can not recognize type: " << query_type << endl;
  }

  return 0;
}

bool load_index(SparseFlow &flower, string &index_dir) {
  int ret = flower.load(index_dir);
  if (0 != ret) {
    cerr << "Flow load failed with ret " << ret << endl;
    return false;
  }
  cout << "Load index done!" << endl;
  return true;
};

int bench_sparse(std::string &query_type, size_t thread_count,
                 size_t batch_count, size_t top_k, string query_file,
                 string &first_sep, string &second_sep, size_t bench_secs,
                 size_t iter_count, SparseFlow &flower, string &index_dir,
                 FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    cerr << "filter mode can not be run in batch mode" << endl;
    return -1;
  }

  if (query_type == "float") {
    SparseBench<float> bench(thread_count, bench_secs, batch_count,
                             filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else if (query_type == "int8") {
    SparseBench<int8_t> bench(thread_count, bench_secs, batch_count,
                              filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    if (load_index(flower, index_dir)) {
      bench.run(&flower, iter_count, top_k);
    } else {
      return -1;
    }
  } else {
    cerr << "Can not recognize type: " << query_type << endl;
  }

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
      cerr << "Failed to load plugin: " << argv[i] << " (" << error << ")"
           << endl;
      return -1;
    }
  }

  YAML::Node config_node;
  try {
    config_node = YAML::LoadFile(argv[1]);
  } catch (...) {
    cerr << "Load YAML file[" << argv[1] << "] failed!" << endl;
    return -1;
  }

  if (!check_config(config_node)) {
    return -1;
  }
  auto config_common = config_node["SearcherCommon"];

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
  }

  // Calculate Bench
  size_t thread_count = config_common["BenchThreadCount"]
                            ? config_common["BenchThreadCount"].as<uint64_t>()
                            : 0;
  size_t iter_count = config_common["BenchIterCount"]
                          ? config_common["BenchIterCount"].as<uint64_t>()
                          : 10000;
  size_t batch_count = config_common["BenchBatchCount"]
                           ? config_common["BenchBatchCount"].as<uint64_t>()
                           : 0;
  g_debug_mode = config_common["DebugMode"]
                     ? config_common["DebugMode"].as<bool>()
                     : false;
  string topk_str = config_common["TopK"].as<string>();

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

  FilterMode filter_mode{FM_NONE};
  if (config_common["FilterMode"]) {
    std::string filter_mode_str = config_common["FilterMode"].as<string>();
    if (filter_mode_str == "tag") {
      filter_mode = FM_TAG;
    }
  }

  vector<int32_t> topk_values;
  StringHelper::Split(topk_str, ",", &topk_values);
  size_t top_k = *topk_values.rbegin();
  string query_file = config_common["QueryFile"].as<string>();
  string first_sep = config_common["QueryFirstSep"]
                         ? config_common["QueryFirstSep"].as<string>()
                         : ";";
  string second_sep = config_common["QuerySecondSep"]
                          ? config_common["QuerySecondSep"].as<string>()
                          : " ";
  string query_type = config_common["QueryType"]
                          ? config_common["QueryType"].as<string>()
                          : "float";
  string container_type = config_common["ContainerType"]
                              ? config_common["ContainerType"].as<string>()
                              : "MMapFileStorage";
  size_t bench_secs = config_common["BenchSecs"]
                          ? config_common["BenchSecs"].as<uint64_t>()
                          : 60;

  if (retrieval_mode == RM_SPARSE) {
    SparseFlow flower;

    // Create container params
    Params container_params;
    if (config_node["ContainerParams"]) {
      // Get index params of Searcher in flower object
      if (!prepare_params(config_node["ContainerParams"], container_params)) {
        return -1;
      }
      cout << "Created index params of a container in flower object " << endl;
    }

    container_params.set("proxima.mmap_file.container.memory_warmup", true);
    // Create a container
    int ret = flower.set_container(container_type, container_params);
    if (0 != ret) {
      cerr << "Create " << container_type << " failed." << endl;
      return -1;
    }

    if (config_common["SearcherClass"]) {
      Params params;
      if (config_node["SearcherParams"]) {
        // Get index params of Searcher in flower object
        if (!prepare_params(config_node["SearcherParams"], params)) {
          return -1;
        }
        cout << "Created index params of a searcher in flower object " << endl;
      }

      // Set a Searcher
      string searcher_class = config_common["SearcherClass"].as<string>();
      ret = flower.set_searcher(searcher_class, params);
      if (0 != ret) {
        cerr << "Failed to create searcher " << searcher_class << endl;
        return -1;
      }
      cout << "Created searcher " << searcher_class << endl;
    } else {  // SearcherConfig
      std::cout << config_common["SearcherConfig"].as<string>() << std::endl;
      auto params =
          zvec::core_interface::IndexFactory::DeserializeIndexParamFromJson(
              config_common["SearcherConfig"].as<string>());

      auto index =
          zvec::core_interface::IndexFactory::CreateAndInitIndex(*params);

      flower.set_searcher(index->index_searcher());
    }

    string index_dir = config_common["IndexPath"].as<string>();

    bench_sparse(query_type, thread_count, batch_count, top_k, query_file,
                 first_sep, second_sep, bench_secs, iter_count, flower,
                 index_dir, filter_mode);

    cout << "Bench Sparse done." << endl;
  } else {
    Flow flower;

    // Create container params
    Params container_params;
    if (config_node["ContainerParams"]) {
      // Get index params of Searcher in flower object
      if (!prepare_params(config_node["ContainerParams"], container_params)) {
        return -1;
      }
      cout << "Created index params of a container in flower object " << endl;
    }

    container_params.set("proxima.mmap_file.container.memory_warmup", true);
    // Create a container
    int ret = flower.set_container(container_type, container_params);
    if (0 != ret) {
      cerr << "Create " << container_type << " failed." << endl;
      return -1;
    }

    // Set a Searcher
    if (config_common["SearcherClass"]) {
      Params params;
      if (config_node["SearcherParams"]) {
        // Get index params of Searcher in flower object
        if (!prepare_params(config_node["SearcherParams"], params)) {
          return -1;
        }
        cout << "Created index params of a searcher in flower object " << endl;
      }

      string searcher_class = config_common["SearcherClass"].as<string>();
      ret = flower.set_searcher(searcher_class, params);
      if (0 != ret) {
        cerr << "Failed to create searcher " << searcher_class << endl;
        return -1;
      }
      cout << "Created searcher " << searcher_class << endl;
    } else {  // SearcherConfig
      std::cout << config_common["SearcherConfig"].as<string>() << std::endl;
      auto params =
          zvec::core_interface::IndexFactory::DeserializeIndexParamFromJson(
              config_common["SearcherConfig"].as<string>());

      auto index =
          zvec::core_interface::IndexFactory::CreateAndInitIndex(*params);

      flower.set_searcher(index->index_searcher());
    }

    string index_dir = config_common["IndexPath"].as<string>();

    bench(query_type, thread_count, batch_count, top_k, query_file, first_sep,
          second_sep, bench_secs, iter_count, flower, index_dir, retrieval_mode,
          filter_mode);

    flower.unload();

    cout << "Bench done." << endl;
  }

  return 0;
}
