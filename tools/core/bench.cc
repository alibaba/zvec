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

#include "bench_result.h"
#include "helper.h"

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
      LOG_ERROR("Load query error");
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

    dim_ = queries[0].size();
    if (typeid(T) == typeid(float)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_FP32, dim_);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_INT8, dim_);
    } else {
      LOG_ERROR("unsupported type");
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(core_interface::Index::Pointer index,
           core_interface::BaseIndexQueryParam::Pointer query_param,
           int max_iter, int topk) {
    // Check
    if (batch_queries_.size() == 0) {
      return;
    }

    query_param_ = query_param;
    query_param_->topk = topk;
    query_param_->is_linear = false;

    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &Bench<T>::start_bench, index, max_iter, &STOP_NOW);
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

  void set_tag_lists(const std::vector<std::vector<uint64_t>> &id_to_tags_list,
                     const std::vector<uint64_t> &tag_key_list) {
    id_to_tags_list_ = id_to_tags_list;
    tag_key_list_ = tag_key_list;
  }

 private:
  void start_bench(core_interface::Index::Pointer index, size_t max_iter,
                   const bool *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    for (; i < max_iter && !*is_stop; i += threads_) {
      int idx = i % batch_queries_.size();

      // prefilter
      FilterResultCache filter_cache;
      std::shared_ptr<IndexFilter> filter_ptr = nullptr;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          LOG_ERROR("query tag list not equal to one!");
          return;
        }

        int ret = filter_cache.filter(id_to_tags_list_, batch_taglists_[idx][0],
                                      tag_key_list_);
        if (ret != 0) {
          LOG_ERROR("prefilter failed, idx: %d", idx);
          return;
        }

        auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

        filter_ptr = std::make_shared<IndexFilter>();
        filter_ptr->set(filterFunc);
      }

      auto query_param = query_param_->Clone();
      query_param->filter = filter_ptr;


      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (retrieval_mode_ == RM_DENSE) {
        if (batch_count_ == 1) {
          ret = do_knn_search<T>(index, batch_queries_[idx], query_param);
        } else {
          ret = do_knn_search_batch<T>(index, batch_queries_[idx], query_param);
        }

        if (ret != 0) {
          LOG_ERROR("Failed to knn search, ret=%d %s", ret,
                    IndexError::What(ret));
          return;
        }
      } else {
        std::string mode = retrieval_mode_ == 1 ? "Dense" : "Sparse";
        LOG_ERROR("unsupported retrieval mode: %s", mode.c_str());
      }

      uint64_t end = Monotime::MicroSeconds();

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  template <typename U>
  typename std::enable_if<
      std::is_same<float, U>::value || std::is_same<int8_t, U>::value ||
          std::is_same<uint32_t, U>::value || std::is_same<uint64_t, U>::value,
      int>::type
  do_knn_search(core_interface::Index::Pointer index, const vector<U> &query,
                core_interface::BaseIndexQueryParam::Pointer query_param) {
    core_interface::DenseVector dense_query;
    dense_query.data = query.data();
    core_interface::VectorData query_data;
    query_data.vector = dense_query;

    core_interface::SearchResult search_result;
    int ret = index->Search(query_data, query_param, &search_result);
    if (ret < 0) {
      return ret;
    }

    if (search_result.doc_list_.empty()) {
      LOG_ERROR("Search results is empty");
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<
      std::is_same<float, U>::value || std::is_same<int8_t, U>::value ||
          std::is_same<uint32_t, U>::value || std::is_same<uint64_t, U>::value,
      int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer index, const vector<U> &query,
      core_interface::BaseIndexQueryParam::Pointer query_param) {
    // For batch search, we search each query separately
    size_t qnum = query.size() / dim_;
    for (size_t i = 0; i < qnum; ++i) {
      core_interface::DenseVector dense_query;
      dense_query.data = query.data() + i * dim_;
      core_interface::VectorData query_data;
      query_data.vector = dense_query;

      core_interface::SearchResult search_result;
      int ret = index->Search(query_data, query_param, &search_result);
      if (ret < 0) {
        return ret;
      }

      if (search_result.doc_list_.empty()) {
        LOG_ERROR("Search results is empty for batch query %zu", i);
      }
    }

    return 0;
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  size_t dim_;
  shared_ptr<ThreadPool> pool_;
  core_interface::BaseIndexQueryParam::Pointer query_param_;

  vector<vector<T>> batch_queries_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  // Tag lists for filtering
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;

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
      LOG_ERROR("Load query error");
      return false;
    }

    linear_sparse_data_ = sparse_data;

    if (batch_count_ == 1) {
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
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;
        vector<vector<uint64_t>> batch_taglists;

        for (size_t i = 0; i < batch_count_; ++i) {
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

        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
        batch_taglists_.push_back(batch_taglists);
      }
    }

    if (typeid(T) == typeid(float)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_FP32);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_INT8);
    } else {
      LOG_ERROR("unsupported type");
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(core_interface::Index::Pointer index,
           core_interface::BaseIndexQueryParam::Pointer query_param,
           int max_iter, int topk) {
    // Check
    if (batch_sparse_counts_.size() == 0) {
      return;
    }

    query_param_ = query_param;
    query_param_->topk = topk;
    query_param_->is_linear = false;

    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &SparseBench<T>::start_bench, index, max_iter,
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

  void set_tag_lists(const std::vector<std::vector<uint64_t>> &id_to_tags_list,
                     const std::vector<uint64_t> &tag_key_list) {
    id_to_tags_list_ = id_to_tags_list;
    tag_key_list_ = tag_key_list;
  }

 private:
  void start_bench(core_interface::Index::Pointer index, size_t max_iter,
                   const bool *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    size_t sparse_query_size = batch_sparse_indices_.size();
    for (; i < max_iter && !*is_stop; i += threads_) {
      int idx = i % sparse_query_size;

      // prefilter
      FilterResultCache filter_cache;
      std::shared_ptr<IndexFilter> filter_ptr = nullptr;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          LOG_ERROR("query tag list not equal to one!");
          return;
        }

        int ret = filter_cache.filter(id_to_tags_list_, batch_taglists_[idx][0],
                                      tag_key_list_);
        if (ret != 0) {
          LOG_ERROR("prefilter failed, idx: %d", idx);
          return;
        }

        auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

        filter_ptr = std::make_shared<IndexFilter>();
        filter_ptr->set(filterFunc);
      }

      auto query_param = query_param_->Clone();
      query_param->filter = filter_ptr;

      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (batch_count_ == 1) {
        if (batch_sparse_counts_[idx].size() != 1) {
          LOG_ERROR("Sparse count size should be 1, since batch count is 1");
          return;
        }
        ret = do_knn_search<T>(index, batch_sparse_counts_[idx][0],
                               batch_sparse_indices_[idx],
                               batch_sparse_features_[idx], query_param);
      } else {
        ret = do_knn_search_batch<T>(
            index, batch_sparse_counts_[idx], batch_sparse_indices_[idx],
            batch_sparse_features_[idx], idx, query_param);
      }

      if (ret != 0) {
        LOG_ERROR("Failed to sparse knn search, ret=%d %s", ret,
                  IndexError::What(ret));
        return;
      }

      uint64_t end = Monotime::MicroSeconds();

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  // sparse search - single query
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(core_interface::Index::Pointer index,
                const uint32_t sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature,
                core_interface::BaseIndexQueryParam::Pointer query_param) {
    core_interface::SparseVector sparse_query;
    sparse_query.count = sparse_count;
    sparse_query.indices = sparse_indices.data();
    sparse_query.values = sparse_feature.data();
    core_interface::VectorData query_data;
    query_data.vector = sparse_query;

    core_interface::SearchResult search_result;
    int ret = index->Search(query_data, query_param, &search_result);
    if (ret < 0) {
      return ret;
    }

    if (search_result.doc_list_.empty()) {
      LOG_ERROR("Search results is empty");
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value ||
                              std::is_same<uint32_t, U>::value ||
                              std::is_same<uint64_t, U>::value,
                          int>::type
  do_knn_search(core_interface::Index::Pointer /*index*/,
                const uint32_t /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/,
                core_interface::BaseIndexQueryParam::Pointer /*query_param*/) {
    return IndexError_Unsupported;
  }

  // sparse search - batch
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer index,
      const vector<uint32_t> &sparse_count,
      const vector<uint32_t> & /*sparse_indices*/,
      const vector<U> & /*sparse_feature*/, size_t batch_idx,
      core_interface::BaseIndexQueryParam::Pointer query_param) {
    // For batch search, search each query separately
    for (size_t i = 0; i < sparse_count.size(); ++i) {
      size_t query_idx = batch_idx * batch_count_ + i;
      if (query_idx >= linear_sparse_data_.size()) {
        break;
      }

      const auto &single_sparse = linear_sparse_data_[query_idx];
      core_interface::SparseVector sparse_query;
      sparse_query.count = single_sparse.count;
      sparse_query.indices = single_sparse.indices.data();
      sparse_query.values = single_sparse.features.data();
      core_interface::VectorData query_data;
      query_data.vector = sparse_query;

      core_interface::SearchResult search_result;
      int ret = index->Search(query_data, query_param, &search_result);
      if (ret < 0) {
        return ret;
      }

      if (search_result.doc_list_.empty()) {
        LOG_ERROR("Search results is empty for batch query %zu", i);
      }
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value ||
                              std::is_same<uint32_t, U>::value ||
                              std::is_same<uint64_t, U>::value,
                          int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer /*index*/,
      const vector<uint32_t> & /*sparse_count*/,
      const vector<uint32_t> & /*sparse_indices*/,
      const vector<U> & /*sparse_feature*/, size_t /*batch_idx*/,
      core_interface::BaseIndexQueryParam::Pointer /*query_param*/) {
    return IndexError_Unsupported;
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  core_interface::BaseIndexQueryParam::Pointer query_param_;
  shared_ptr<ThreadPool> pool_;

  vector<SparseData<T>> linear_sparse_data_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  // Tag lists for filtering
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;

  FilterMode filter_mode_{FM_NONE};
  BenchResult bench_result_;
  static bool STOP_NOW;
};
template <typename T>
bool SparseBench<T>::STOP_NOW = false;

bool check_config(YAML::Node &config_node) {
  auto common = config_node["IndexCommon"];
  if (!common) {
    LOG_ERROR("Can not find [IndexCommon] in config");
    return false;
  }
  if (!common["IndexConfig"]) {
    LOG_ERROR("Can not find [IndexConfig] in config");
    return false;
  }
  if (!common["IndexPath"]) {
    LOG_ERROR("Can not find [IndexPath] in config");
    return false;
  }
  if (!common["TopK"]) {
    LOG_ERROR("Can not find [TopK] in config");
    return false;
  }
  if (!common["QueryFile"]) {
    LOG_ERROR("Can not find [QueryFile] in config");
    return false;
  }

  auto query_config = config_node["QueryConfig"];
  if (!query_config) {
    LOG_ERROR("Can not find [QueryConfig] in config");
    return false;
  }
  if (!query_config["QueryParam"]) {
    LOG_ERROR("Can not find [QueryConfig.QueryParam] in config");
    return false;
  }


  return true;
}

void usage(void) {
  cout << "Usage: bench CONFIG.yaml [plugin file path]" << endl;
}


int bench(std::string &query_type, size_t thread_count, size_t batch_count,
          size_t top_k, string query_file, string &first_sep,
          string &second_sep, size_t bench_secs, size_t iter_count,
          core_interface::Index::Pointer index,
          core_interface::BaseIndexQueryParam::Pointer query_param,
          string &index_dir, RetrievalMode retrieval_mode,
          FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    LOG_ERROR("filter mode can not be run in batch mode");
    return -1;
  }

  std::vector<std::vector<uint64_t>> id_to_tags_list;
  std::vector<uint64_t> tag_key_list;
  // Load tag lists if available
  load_taglists(index_dir, id_to_tags_list, tag_key_list);

  if (query_type == "float") {
    Bench<float> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                       filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "int8") {
    Bench<int8_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                        filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "binary") {
    Bench<uint32_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "binary64") {
    Bench<uint64_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else {
    LOG_ERROR("Can not recognize type: %s", query_type.c_str());
  }

  return 0;
}

int bench_sparse(std::string &query_type, size_t thread_count,
                 size_t batch_count, size_t top_k, string query_file,
                 string &first_sep, string &second_sep, size_t bench_secs,
                 size_t iter_count, core_interface::Index::Pointer index,
                 core_interface::BaseIndexQueryParam::Pointer query_param,
                 string &index_dir, FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    LOG_ERROR("filter mode can not be run in batch mode");
    return -1;
  }

  std::vector<std::vector<uint64_t>> id_to_tags_list;
  std::vector<uint64_t> tag_key_list;
  // Load tag lists if available
  load_taglists(index_dir, id_to_tags_list, tag_key_list);

  if (query_type == "float") {
    SparseBench<float> bench(thread_count, bench_secs, batch_count,
                             filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "int8") {
    SparseBench<int8_t> bench(thread_count, bench_secs, batch_count,
                              filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else {
    LOG_ERROR("Can not recognize type: %s", query_type.c_str());
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
      LOG_ERROR("Failed to load plugin: %s (%s)", argv[i], error.c_str());
      return -1;
    }
  }

  YAML::Node config_node;
  try {
    config_node = YAML::LoadFile(argv[1]);
  } catch (...) {
    LOG_ERROR("Load YAML file[%s] failed!", argv[1]);
    return -1;
  }

  if (!check_config(config_node)) {
    return -1;
  }
  auto config_common = config_node["IndexCommon"];

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
  size_t bench_secs = config_common["BenchSecs"]
                          ? config_common["BenchSecs"].as<uint64_t>()
                          : 60;

  string index_dir = config_common["IndexPath"].as<string>();

  core_interface::Index::Pointer index;
  core_interface::BaseIndexQueryParam::Pointer query_param;
  if (0 !=
      parse_and_load_index_param(config_node, index_dir, index, query_param)) {
    LOG_ERROR("Failed to parse and load index param");
    return -1;
  }

  if (retrieval_mode == RM_SPARSE) {
    bench_sparse(query_type, thread_count, batch_count, top_k, query_file,
                 first_sep, second_sep, bench_secs, iter_count, index,
                 query_param, index_dir, filter_mode);

    cout << "Bench Sparse done." << endl;
  } else {
    bench(query_type, thread_count, batch_count, top_k, query_file, first_sep,
          second_sep, bench_secs, iter_count, index, query_param, index_dir,
          retrieval_mode, filter_mode);

    cout << "Bench done." << endl;
  }

  // Cleanup
  index->Close();

  return 0;
}
