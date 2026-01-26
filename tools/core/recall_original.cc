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

#include <sys/stat.h>
#include <signal.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ailego/hash/crc32c.h>
#include <ailego/io/file.h>
#include <ailego/parallel/lock.h>
#include <ailego/parallel/thread_pool.h>
#include <ailego/utility/time_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include "framework/index_plugin.h"
#include "interface/index_factory.h"
#include "interface/index_param.h"
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
using SparseFlow = SparseFlow;

mutex recall_lock;
bool g_compare_by_id = false;
float g_recall_precision;

//--------------------------------------------------
// Recall
//--------------------------------------------------
enum RetrievalMode { RM_UNDEFINED = 0, RM_DENSE = 1, RM_SPARSE = 2 };

enum FilterMode { FM_UNDEFINED = 0, FM_NONE = 1, FM_TAG = 2 };

template <typename T>
class Recall {
 public:
  Recall(size_t threads, const string &output, size_t batch_count,
         FilterMode filter_mode)
      : threads_(threads),
        output_(output),
        batch_count_(batch_count),
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>(true);
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, true);
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
      call_batch_api_ = false;
    } else {
      call_batch_api_ = true;
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

  void run_dense(Flow *flower, const string &recall_tops, size_t gt_count) {
    StringHelper::Split(recall_tops, ",", &topk_ids_);
    std::sort(topk_ids_.begin(), topk_ids_.end());

    for (auto i : topk_ids_) {
      recall_res_[i] = 0.0f;
    }
    size_t topk = recall_res_.rbegin()->first;

    gt_count = topk < gt_count ? gt_count : topk;

    if (external_gt_file_enabled_) {
      cout << "Internal ground truth file NOT used since external ground truth "
              "file has been loaded"
           << endl;
    } else {
      cout << "Loading internal ground truth file" << endl;

      if (!load_gt_dense(flower, gt_count)) {
        cerr << "Load ground truth file failed!" << endl;
        return;
      }
    }

    if (batch_queries_.size() < threads_) {
      threads_ = batch_queries_.size();
      pool_ = make_shared<ThreadPool>(true, threads_);
      cout << "Query size too small, resize thread pool count[" << threads_
           << "]" << endl;
    }

    // Prepare file handler
    vector<pair<fstream *, fstream *>> output_fs;
    if (!output_.empty()) {
      string cmd = "mkdir -p " + output_;
      int ret = system(cmd.c_str());
      if (ret != 0) {
        std::cerr << "execute cmd " << cmd << " failed" << std::endl;
        return;
      }
      struct stat sb;
      if (stat(output_.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        cout << "logs output to : " << output_ << endl;
        for (size_t i = 0; i < threads_; ++i) {
          fstream *fs_k = new fstream();
          fs_k->open(output_ + "/t" + to_string(i) + ".knn", ios::out);
          fstream *fs_l = new fstream();
          fs_l->open(output_ + "/t" + to_string(i) + ".linear", ios::out);
          output_fs.push_back(make_pair(fs_k, fs_l));
        }
      }
    }

    signal(SIGINT, stop);
    size_t i = 0;
    for (; !STOP_NOW && i < batch_queries_.size();) {
      if (pool_->pending_count() >= pool_->count()) {
        this_thread::sleep_for(chrono::microseconds(1));
        continue;
      }

      Closure::Pointer task = Closure::New(this, &Recall::recall_one_dense,
                                           flower, topk, i, output_fs);
      pool_->enqueue_and_wake(task);

      i++;
    }
    pool_->wait_finish();

    for (auto fs : output_fs) {
      fs.first->close();
      fs.second->close();
      delete fs.first;
      delete fs.second;
    }
    cout << "Process query: " << i << endl;
    for (auto it : recall_res_) {
      cout << "Recall@" << it.first << ": "
           << it.second / linear_queries_.size() << endl;
    }
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;

    if (!reader.load_query(query_file, first_sep, second_sep, linear_queries_,
                           linear_sparse_data_, linear_taglists_)) {
      cerr << "Load query error" << endl;
      return false;
    }

    if (batch_count_ == 1) {
      batch_queries_ = linear_queries_;

      for (size_t i = 0; i < linear_sparse_data_.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(linear_sparse_data_[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(linear_sparse_data_[i].indices);
        batch_sparse_features_.push_back(linear_sparse_data_[i].features);
      }

      for (size_t i = 0; i < linear_taglists_.size(); ++i) {
        vector<vector<uint64_t>> new_taglists;
        new_taglists.push_back(linear_taglists_[i]);

        batch_taglists_.push_back(std::move(new_taglists));
      }
    } else {
      size_t num_batch =
          (linear_queries_.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<T> batch_query;
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;
        vector<vector<uint64_t>> batch_taglists;

        for (size_t i = 0; i < batch_count_; ++i) {
          for (size_t k = 0; k < linear_queries_[idx].size(); ++k) {
            batch_query.push_back(linear_queries_[idx][k]);
          }

          batch_sparse_count.push_back(linear_sparse_data_[idx].count);

          for (size_t k = 0; k < linear_sparse_data_[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(linear_sparse_data_[idx].indices[k]);
          }

          for (size_t k = 0; k < linear_sparse_data_[idx].features.size();
               ++k) {
            batch_sparse_feature.push_back(
                linear_sparse_data_[idx].features[k]);
          }

          idx = (idx + 1) % linear_queries_.size();
        }

        batch_queries_.push_back(batch_query);
        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
        batch_taglists_.push_back(batch_taglists);
      }
    }

    dim_ = linear_queries_[0].size();
    total_querys_ = linear_queries_.size();
    if (typeid(T) == typeid(float)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_FP32, dim_);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_INT8, dim_);
    } else {
      cerr << "unsupported type";
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  bool load_external_gt_file(const std::string &external_gt_file,
                             const std::string &first_sep,
                             const std::string &second_sep) {
    TxtInputReader<T> reader;
    bool ret =
        reader.load_external_gt(external_gt_file, first_sep, second_sep, gt_);
    if (ret) {
      cout << "Load external ground truth file["
           << File::BaseName(external_gt_file) << "] done!" << endl;
      external_gt_file_enabled_ = true;
    } else {
      cerr << "Failed to load ground truth file!" << endl;
    }

    return ret;
  }

 private:
  std::string compute_crc(size_t gt_count) {
    uint32_t crc = 0u;
    // dense
    if (batch_queries_.size() > 0) {
      size_t one_size = dim_ * sizeof(T);
      size_t data_size = total_querys_ * one_size + sizeof(size_t);
      char *data = new char[data_size];
      size_t q = 0;
      char *p = data;
      for (; q < batch_queries_.size(); ++q) {
        memcpy(p, batch_queries_[q].data(),
               batch_queries_[q].size() * sizeof(T));
        p += batch_queries_[q].size() * sizeof(T);
      }
      memcpy(p, &gt_count, sizeof(size_t));
      crc = Crc32c::Hash(data, data_size, crc);
      delete[] data;
    }

    // sparse
    if (linear_sparse_data_.size() > 0) {
      for (size_t i = 0; i < linear_sparse_data_.size(); ++i) {
        crc = Crc32c::Hash(&(linear_sparse_data_[i].count), sizeof(uint32_t),
                           crc);
        crc =
            Crc32c::Hash(linear_sparse_data_[i].indices.data(),
                         linear_sparse_data_[i].count * sizeof(uint32_t), crc);
        crc = Crc32c::Hash(linear_sparse_data_[i].features.data(),
                           linear_sparse_data_[i].count * sizeof(T), crc);
      }
    }

    char crc_str[64];
    snprintf(crc_str, sizeof(crc_str), "%X", crc);

    return std::string(crc_str);
  }

  bool load_gt_dense(Flow *flower, size_t gt_count) {
    std::string crc_str = compute_crc(gt_count);

    string gt_file = string("gt.") + crc_str;

    File gtf;
    if (!gtf.IsRegular(gt_file.c_str())) {
      cout << "Ground truth file[" << gt_file << "] not exist, try to create it"
           << endl;
      ElapsedTime timer;

      size_t size = sizeof(uint64_t) + sizeof(float);
      size_t file_size =
          linear_queries_.size() * (sizeof(int) + size * gt_count);

      std::string gt_file_temp = gt_file + ".tmp";
      gtf.create(gt_file_temp.c_str(), file_size);

      gt_.resize(linear_queries_.size());

      atomic_bool error(false);
      size_t count = 0;
      float s = linear_queries_.size() / 100.0;
      size_t pc = 0;
      SpinMutex spin_lock;

      function<void(size_t)> fun = [&](size_t i) {
        spin_lock.lock();
        count++;
        size_t process = (size_t)ceil(count / s);
        if (process > pc) {
          pc = process;
          stringstream msg;
          msg << "\r" << setw(3) << setfill(' ') << process << "% " << left
              << setfill('=') << setw(process / 2 + 1) << "[" << right
              << setfill(' ') << setw(51 - process / 2) << "]";
          cout << msg.str() << flush;
        }
        spin_lock.unlock();

        auto query = linear_queries_[i];
        Flow::Context::Pointer context = flower->create_context();
        if (!context) {
          cerr << "Failed to create search context" << endl;
          return;
        }

        FilterResultCache filter_cache;
        if (filter_mode_ == FM_TAG) {
          if (batch_taglists_[i].size() != 1) {
            cerr << "query tag list not equal to one!" << endl;
            return;
          }

          int ret = filter_cache.filter(flower->id_to_tags_list(),
                                        batch_taglists_[i][0],
                                        flower->tag_key_list());
          if (ret != 0) {
            cerr << "prefilter failed, idx: " << i << std::endl;

            return;
          }

          auto filterFunc = [&](uint64_t key) {
            return filter_cache.find(key);
          };

          context->set_filter(filterFunc);
        }

        context->set_topk(gt_count);
        int ret = do_linear_search<T>(flower, context, query);
        if (ret < 0) {
          cerr << "Failed to linear search, ret=" << ret << endl;
          error.exchange(true);
          return;
        }
        auto result = context->result();
        vector<pair<uint64_t, float>> one_gt;
        one_gt.reserve(gt_count);

        for (auto knn : result) {
          one_gt.emplace_back(knn.key(), knn.score());
        }
        gt_[i] = one_gt;
      };
      for (size_t i = 0; i < linear_queries_.size(); ++i) {
        if (error) {
          break;
        }
        pool_->enqueue_and_wake(Closure::New(fun, i));
      }
      pool_->wait_finish();

      if (error) {
        cout << endl
             << "Ground truth file[" << gt_file << "] create failed!" << endl;
        gtf.close();
        remove(gt_file.c_str());
        return false;
      }

      for (size_t i = 0; i < gt_.size(); ++i) {
        auto &gt = gt_[i];

        gtf.write(&gt_count, sizeof(int));

        for (size_t j = 0; j < gt.size(); j++) {
          auto &one_gt = gt[j];

          gtf.write(&one_gt.first, sizeof(uint64_t));
          gtf.write(&one_gt.second, sizeof(float));
        }

        // if ground truth is less than gt count, fill it up
        if (gt.size() != gt_count) {
          std::cout
              << "WARN: GT result count less than GT expected count, index: "
              << i << ", expected GT count: " << gt_count
              << ", actual GT count: " << gt.size() << std::endl;

          uint64_t key{-1LLU};
          float score{std::nanf("")};

          for (size_t j = gt.size(); j < gt_count; ++j) {
            gtf.write(&key, sizeof(uint64_t));
            gtf.write(&score, sizeof(float));
          }
        }
      }

      gtf.close();

      if (!File::Rename(gt_file_temp, gt_file)) {
        cerr << "failed to rename ground truth file, src: " << gt_file_temp
             << ", dst: " << gt_file << endl;

        return false;
      }

      cout << endl
           << "Ground truth file create successful in "
           << timer.milli_seconds() / 1000 << "s." << endl;
    } else {
      if (!gtf.open(gt_file.c_str(), true)) {
        cerr << "Failed to open ground truth file[" << gt_file << "]" << endl;
        return false;
      }
      size_t file_size = gtf.size();

      constexpr size_t LENGTH = 10240;
      constexpr size_t GT_PAIR_SIZE = sizeof(uint64_t) + sizeof(float);

      char *buffer = new char[LENGTH];
      gtf.read(buffer, sizeof(int));

      size_t gt_count_input = (size_t) * (int *)buffer;
      size_t one_query_line_size = sizeof(int) + GT_PAIR_SIZE * gt_count_input;

      if (gt_count != gt_count_input || file_size % one_query_line_size != 0) {
        cerr << "Ground truth file[" << gt_file << "] content error!" << endl;
        gtf.close();
        return false;
      }

      size_t query_num = file_size / one_query_line_size;
      if (one_query_line_size > LENGTH) {
        delete[] buffer;
        buffer = new char[one_query_line_size];
      }

      for (size_t n = 0; n < query_num; ++n) {
        gtf.read(n * one_query_line_size, buffer, one_query_line_size);
        vector<pair<uint64_t, float>> one_gt;
        one_gt.reserve(gt_count);

        for (size_t i = 0; i < gt_count; ++i) {
          uint64_t key = *(uint64_t *)(buffer + sizeof(int) + GT_PAIR_SIZE * i);
          float score = *(float *)(buffer + sizeof(int) + GT_PAIR_SIZE * i +
                                   sizeof(uint64_t));

          if (key != -1LLU) {
            one_gt.emplace_back(key, score);
          }
        }
        gt_.emplace_back(one_gt);
      }
      delete[] buffer;
      cout << "Load ground truth file[" << gt_file << "] done!" << endl;
    }

    return true;
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
    // query.size() / count * sizeof(float), count);
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8,
    // query.size() / count, count);
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() / count * sizeof(uint32_t), count);
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query, size_t count) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() / count * sizeof(uint32_t), count);
    return flower->search_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
    // query.size() * sizeof(float), 1);
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8,
    // query.size() , 1);
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() * sizeof(uint32_t), 1);
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_knn_search(Flow *flower, Flow::Context::Pointer &context,
                const vector<U> &query) {
    // Do knn_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() * sizeof(uint32_t), 1);
    return flower->search_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
    // query.size() * sizeof(float), 1);
    return flower->search_bf_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8,
    // query.size() , 1);
    return flower->search_bf_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() * sizeof(uint32_t), 1);
    return flower->search_bf_impl(query.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() * sizeof(uint32_t), 1);
    return flower->search_bf_impl(query.data(), qmeta_, context);
  }


  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query, size_t count) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
    // query.size() / count * sizeof(float), count);
    return flower->search_bf_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query, size_t count) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8,
    // query.size()  / count, count);
    return flower->search_bf_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query, size_t count) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() / count * sizeof(uint32_t), count);
    return flower->search_bf_impl(query.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<uint64_t, U>::value, int>::type
  do_linear_search(Flow *flower, Flow::Context::Pointer &context,
                   const vector<U> &query, size_t count) {
    // Do linear_search
    // IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32,
    // query.size() / count * sizeof(uint32_t), count);
    return flower->search_bf_impl(query.data(), qmeta_, count, context);
  }

  void recall_one_dense(
      Flow *flower, size_t topk, size_t index,
      std::vector<pair<std::fstream *, std::fstream *>> &output_fs) {
    const auto &query = batch_queries_[index];

    size_t thread_index = pool_->indexof_this();
    fstream *knn_fs = nullptr;
    fstream *linear_fs = nullptr;
    if (output_fs.size() > thread_index) {
      knn_fs = output_fs[thread_index].first;
      linear_fs = output_fs[thread_index].second;
    }

    Flow::Context::Pointer knn_context = flower->create_context();
    if (!knn_context) {
      cerr << "Failed to create search context" << endl;
      return;
    }
    knn_context->set_topk(topk);

    auto cal_recall = [&, this](const std::vector<IndexDocument> &knn_res,
                                size_t idx) {
      vector<IndexDocument> linear_res;

      size_t result_size = std::min(topk, gt_[idx].size());
      if (result_size == 0) {
        return;
      }

      for (size_t i = 0; i < result_size; ++i) {
        auto gt_node = gt_[idx][i];

        linear_res.emplace_back(gt_node.first, gt_node.second, gt_node.first);
      }


      if (knn_fs) {
        for (auto knn : knn_res) {
          string str = "query[" + to_string(idx) + "]\tkey[" +
                       to_string(knn.key()) + "], dist[" +
                       to_string(knn.score()) + "]\n";
          knn_fs->write(str.c_str(), str.size());
        }
      }
      size_t match = 0;
      bool asc =
          (linear_res.size() > 1 &&
           (linear_res[0].score() > linear_res[linear_res.size() - 1].score()))
              ? false
              : true;

      map<int32_t, size_t> topk_matchs;
      if (g_compare_by_id) {
        for (size_t i = 0; i < topk_ids_.size(); ++i) {
          topk_matchs[topk_ids_[i]] = 0;
        }
      }
      for (size_t i = 0, j = 0; i < linear_res.size();) {
        bool m = false;       // if current doc matched in max topk
        bool changed = true;  // if i changed
        if (g_compare_by_id) {
          for (size_t k = 0; k < topk_ids_.size(); ++k) {
            size_t dynamic_size = (size_t)topk_ids_[k];
            for (; dynamic_size + 1 < knn_res.size(); ++dynamic_size) {
              if (fabs(knn_res[dynamic_size - 1].score() -
                       knn_res[dynamic_size].score()) >=
                  numeric_limits<float>::epsilon()) {
                break;
              }
            }
            for (size_t l = 0; l < dynamic_size && l < knn_res.size(); ++l) {
              if (linear_res[i].key() == knn_res[l].key()) {
                topk_matchs[topk_ids_[k]]++;
                if (k == topk_ids_.size() - 1) {
                  m = true;
                }
                break;
              }
            }
          }
          ++i;
          auto it = recall_res_.find(i);
          if (it != recall_res_.end()) {
            lock_guard<mutex> lock(recall_lock);
            it->second += 100.0 * topk_matchs[i] / i;
          }
        } else {
          size_t cur_topk = i + 1;
          if (j < knn_res.size()) {
            if (fabs(linear_res[i].score() - knn_res[j].score()) <
                g_recall_precision) {
              ++j;
              ++i;
              match++;
              m = true;
            } else {
              if ((asc && linear_res[i].score() < knn_res[j].score()) ||
                  (!asc && linear_res[i].score() > knn_res[j].score())) {
                ++i;
              } else {
                changed = false;
                ++j;
              }
            }
          } else {
            ++i;
          }
          auto it = recall_res_.find(cur_topk);
          if (changed && it != recall_res_.end()) {
            lock_guard<mutex> lock(recall_lock);
            it->second += 100.0 * match / cur_topk;
          }
        }
        if (linear_fs && changed) {
          string str = string(m ? "    HIT" : "NOT HIT") + "  query[" +
                       to_string(idx) + "]\tkey[" +
                       to_string(linear_res[i - 1].key()) + "], dist[" +
                       to_string(linear_res[i - 1].score()) + "]\n";
          linear_fs->write(str.c_str(), str.size());
        }
      }
    };

    // prefilter
    FilterResultCache filter_cache;
    if (filter_mode_ == FM_TAG) {
      if (batch_taglists_[index].size() != 1) {
        cerr << "query tag list not equal to one!" << endl;
        return;
      }

      int ret = filter_cache.filter(flower->id_to_tags_list(),
                                    batch_taglists_[index][0],
                                    flower->tag_key_list());
      if (ret != 0) {
        cerr << "prefilter failed, idx: " << index << std::endl;

        return;
      }

      auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

      knn_context->set_filter(filterFunc);
    }

    if (call_batch_api_) {
      size_t qnum = query.size() / dim_;
      int ret = do_knn_search<T>(flower, knn_context, query, qnum);
      if (ret < 0) {
        cerr << "Failed to knn_search batch, ret=" << ret << " "
             << IndexError::What(ret) << endl;
        return;
      }
      for (size_t i = 0; i < qnum; ++i) {
        size_t idx = index * batch_count_ + i;
        if (idx >= linear_queries_.size()) {
          break;
        }

        auto &knn_res = knn_context->result(i);
        cal_recall(knn_res, idx);
      }
    } else {
      int ret = do_knn_search<T>(flower, knn_context, query);
      if (ret < 0) {
        cerr << "Failed to knn_search, ret=" << ret << " "
             << IndexError::What(ret) << endl;
        return;
      }
      auto &knn_res = knn_context->result();
      cal_recall(knn_res, index);
    }

    // std::cout << "id: " << index << ": \n" <<
    // knn_context->flow_context()->searcher_context()->profiler().display();
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  bool call_batch_api_;
  string output_;
  size_t batch_count_;
  shared_ptr<ThreadPool> pool_;

  // for gt
  vector<vector<T>> linear_queries_;
  vector<SparseData<T>> linear_sparse_data_;
  vector<vector<uint64_t>> linear_taglists_;

  // for recall
  vector<vector<T>> batch_queries_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  size_t dim_;
  size_t total_querys_;

  map<size_t, float> recall_res_;
  vector<int32_t> topk_ids_;
  vector<vector<pair<uint64_t, float>>> gt_;

  bool external_gt_file_enabled_{false};

  FilterMode filter_mode_{FM_NONE};

  static bool STOP_NOW;
};

template <typename T>
bool Recall<T>::STOP_NOW = false;

//--------------------------------------------------
// Sparse Recall
//--------------------------------------------------
template <typename T>
class SparseRecall {
 public:
  SparseRecall(size_t threads, const string &output, size_t batch_count,
               FilterMode filter_mode)
      : threads_(threads),
        output_(output),
        batch_count_(batch_count),
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>(true);
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, true);
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
      call_batch_api_ = false;
    } else {
      call_batch_api_ = true;
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

  int transform_queries_without_hybrid_scale(
      const vector<vector<T>> &queries,
      const vector<vector<T>> &sparse_features,
      vector<vector<T>> *queries_output,
      vector<vector<T>> *sparse_features_output) {
    if (!queries_output || !sparse_features_output) {
      std::cerr << "input should not be empty in transfrom queries"
                << std::endl;

      return -1;
    }

    queries_output->clear();
    sparse_features_output->clear();

    for (size_t i = 0; i < queries.size(); ++i) {
      vector<T> query_output;
      vector<T> sparse_feature_output;

      transform_query_without_hybrid_scale(queries[i], sparse_features[i],
                                           &query_output,
                                           &sparse_feature_output);

      queries_output->push_back(query_output);
      sparse_features_output->push_back(sparse_feature_output);
    }

    return 0;
  }

  void run_sparse(SparseFlow *flower, const string &recall_tops,
                  size_t gt_count) {
    StringHelper::Split(recall_tops, ",", &topk_ids_);
    std::sort(topk_ids_.begin(), topk_ids_.end());

    for (auto i : topk_ids_) {
      recall_res_[i] = 0.0f;
    }
    size_t topk = recall_res_.rbegin()->first;

    gt_count = topk < gt_count ? gt_count : topk;

    if (external_gt_file_enabled_) {
      cout << "Internal ground truth file NOT used since external ground truth "
              "file has been loaded"
           << endl;
    } else {
      cout << "Loading internal ground truth file" << endl;

      if (!load_gt_sparse(flower, gt_count)) {
        cerr << "Load ground truth file failed!" << endl;
        return;
      }
    }

    if (batch_sparse_counts_.size() < threads_) {
      threads_ = batch_sparse_counts_.size();
      pool_ = make_shared<ThreadPool>(true, threads_);
      cout << "Query size too small, resize thread pool count[" << threads_
           << "]" << endl;
    }

    // Prepare file handler
    vector<pair<fstream *, fstream *>> output_fs;
    if (!output_.empty()) {
      string cmd = "mkdir -p " + output_;
      int ret = system(cmd.c_str());
      if (ret != 0) {
        std::cerr << "execute cmd " << cmd << " failed" << std::endl;
        return;
      }
      struct stat sb;
      if (stat(output_.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        cout << "logs output to : " << output_ << endl;
        for (size_t i = 0; i < threads_; ++i) {
          fstream *fs_k = new fstream();
          fs_k->open(output_ + "/t" + to_string(i) + ".knn", ios::out);
          fstream *fs_l = new fstream();
          fs_l->open(output_ + "/t" + to_string(i) + ".linear", ios::out);
          output_fs.push_back(make_pair(fs_k, fs_l));
        }
      }
    }

    signal(SIGINT, stop);
    size_t i = 0;
    for (; !STOP_NOW && i < batch_sparse_counts_.size();) {
      if (pool_->pending_count() >= pool_->count()) {
        this_thread::sleep_for(chrono::microseconds(1));
        continue;
      }

      Closure::Pointer task = Closure::New(
          this, &SparseRecall::recall_one_sparse, flower, topk, i, output_fs);
      pool_->enqueue_and_wake(task);

      i++;
    }
    pool_->wait_finish();

    for (auto fs : output_fs) {
      fs.first->close();
      fs.second->close();
      delete fs.first;
      delete fs.second;
    }
    cout << "Process query: " << i << endl;
    for (auto it : recall_res_) {
      cout << "Recall@" << it.first << ": "
           << it.second / linear_queries_.size() << endl;
    }
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;

    if (!reader.load_query(query_file, first_sep, second_sep, linear_queries_,
                           linear_sparse_data_, linear_taglists_)) {
      cerr << "Load query error" << endl;
      return false;
    }

    if (batch_count_ == 1) {
      for (size_t i = 0; i < linear_sparse_data_.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(linear_sparse_data_[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(linear_sparse_data_[i].indices);
        batch_sparse_features_.push_back(linear_sparse_data_[i].features);
      }
    } else {
      size_t num_batch =
          (linear_queries_.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;

        for (size_t i = 0; i < batch_count_; ++i) {
          batch_sparse_count.push_back(linear_sparse_data_[idx].count);

          for (size_t k = 0; k < linear_sparse_data_[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(linear_sparse_data_[idx].indices[k]);
          }

          for (size_t k = 0; k < linear_sparse_data_[idx].features.size();
               ++k) {
            batch_sparse_feature.push_back(
                linear_sparse_data_[idx].features[k]);
          }

          idx = (idx + 1) % linear_queries_.size();
        }
        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
      }
    }

    total_querys_ = linear_queries_.size();
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

  bool load_gt_sparse(SparseFlow *flower, size_t gt_count) {
    std::string crc_str = compute_crc();

    string gt_file = string("gt.") + crc_str;

    File gtf;
    if (!gtf.IsRegular(gt_file.c_str())) {
      cout << "Ground truth file[" << gt_file << "] not exist, try to create it"
           << endl;
      ElapsedTime timer;
      size_t size = sizeof(uint64_t) + sizeof(float);
      size_t file_size =
          linear_sparse_data_.size() * (sizeof(int) + size * gt_count);

      std::string gt_file_temp = gt_file + ".tmp";
      gtf.create(gt_file_temp.c_str(), file_size);

      gt_.resize(linear_sparse_data_.size());

      atomic_bool error(false);
      size_t count = 0;
      float s = linear_sparse_data_.size() / 100.0;
      size_t pc = 0;
      SpinMutex spin_lock;

      function<void(size_t)> fun = [&](size_t i) {
        spin_lock.lock();
        count++;
        size_t process = (size_t)ceil(count / s);
        if (process > pc) {
          pc = process;
          stringstream msg;
          msg << "\r" << setw(3) << setfill(' ') << process << "% " << left
              << setfill('=') << setw(process / 2 + 1) << "[" << right
              << setfill(' ') << setw(51 - process / 2) << "]";
          cout << msg.str() << flush;
        }
        spin_lock.unlock();

        SparseFlow::Context::Pointer context = flower->create_context();
        if (!context) {
          cerr << "Failed to create search context" << endl;
          return;
        }

        context->set_topk(gt_count);
        SparseData<T> sparse_data = linear_sparse_data_[i];

        // prefilter
        FilterResultCache filter_cache;
        if (filter_mode_ == FM_TAG) {
          if (batch_taglists_[i].size() != 1) {
            cerr << "query tag list not equal to one!" << endl;
            return;
          }

          int ret = filter_cache.filter(flower->id_to_tags_list(),
                                        batch_taglists_[i][0],
                                        flower->tag_key_list());
          if (ret != 0) {
            cerr << "prefilter failed, idx: " << i << std::endl;

            return;
          }

          auto filterFunc = [&](uint64_t key) {
            return filter_cache.find(key);
          };

          context->set_filter(filterFunc);
        }

        int ret =
            do_linear_search<T>(flower, context, sparse_data.count,
                                sparse_data.indices, sparse_data.features);
        if (ret < 0) {
          cerr << "Failed to sparse linear search, ret=" << ret << endl;
          error.exchange(true);
          return;
        }
        auto result = context->result();

        vector<pair<uint64_t, float>> one_gt;
        one_gt.reserve(gt_count);

        for (auto knn : result) {
          one_gt.emplace_back(knn.key(), knn.score());
        }
        gt_[i] = one_gt;
      };

      for (size_t i = 0; i < linear_sparse_data_.size(); ++i) {
        if (error) {
          break;
        }
        pool_->enqueue_and_wake(Closure::New(fun, i));
      }
      pool_->wait_finish();

      if (error) {
        cout << endl
             << "Ground truth file[" << gt_file << "] create failed!" << endl;
        gtf.close();
        remove(gt_file.c_str());
        return false;
      }

      for (size_t i = 0; i < gt_.size(); ++i) {
        auto &gt = gt_[i];

        gtf.write(&gt_count, sizeof(int));

        for (size_t j = 0; j < gt.size(); j++) {
          auto &one_gt = gt[j];

          gtf.write(&one_gt.first, sizeof(uint64_t));
          gtf.write(&one_gt.second, sizeof(float));
        }

        // if ground truth is less than gt count, fill it up
        if (gt.size() != gt_count) {
          std::cout
              << "WARN: GT result count less than GT expected count, index: "
              << i << ", expected GT count: " << gt_count
              << ", actual GT count: " << gt.size() << std::endl;

          uint64_t key{-1LLU};
          float score{std::nanf("")};

          for (size_t j = gt.size(); j < gt_count; ++j) {
            gtf.write(&key, sizeof(uint64_t));
            gtf.write(&score, sizeof(float));
          }
        }
      }
      gtf.close();

      if (!File::Rename(gt_file_temp, gt_file)) {
        cerr << "failed to rename ground truth file, src: " << gt_file_temp
             << ", dst: " << gt_file << endl;

        return false;
      }

      cout << endl
           << "Ground truth file create successful in "
           << timer.milli_seconds() / 1000 << "s." << endl;
    } else {
      if (!gtf.open(gt_file.c_str(), true)) {
        cerr << "Failed to open ground truth file[" << gt_file << "]" << endl;
        return false;
      }
      size_t file_size = gtf.size();

      constexpr size_t LENGTH = 10240;
      constexpr size_t GT_PAIR_SIZE = sizeof(uint64_t) + sizeof(float);

      char *buffer = new char[LENGTH];
      gtf.read(buffer, sizeof(int));

      size_t gt_count_input = (size_t) * (int *)buffer;
      size_t one_query_line_size = sizeof(int) + GT_PAIR_SIZE * gt_count_input;

      if (gt_count != gt_count_input || file_size % one_query_line_size != 0) {
        cerr << "Ground truth file[" << gt_file << "] content error!" << endl;
        gtf.close();
        return false;
      }

      size_t query_num = file_size / one_query_line_size;
      if (one_query_line_size > LENGTH) {
        delete[] buffer;
        buffer = new char[one_query_line_size];
      }

      for (size_t n = 0; n < query_num; ++n) {
        gtf.read(n * one_query_line_size, buffer, one_query_line_size);
        vector<pair<uint64_t, float>> one_gt;
        one_gt.reserve(gt_count);

        for (size_t i = 0; i < gt_count; ++i) {
          uint64_t key = *(uint64_t *)(buffer + sizeof(int) + GT_PAIR_SIZE * i);
          float score = *(float *)(buffer + sizeof(int) + GT_PAIR_SIZE * i +
                                   sizeof(uint64_t));

          if (key != -1LLU) {
            one_gt.emplace_back(key, score);
          }
        }

        gt_.emplace_back(one_gt);
      }

      delete[] buffer;
      cout << "Load ground truth file[" << gt_file << "] done!" << endl;
    }

    return true;
  }

  bool load_external_gt_file(const std::string &external_gt_file,
                             const std::string &first_sep,
                             const std::string &second_sep) {
    TxtInputReader<T> reader;
    bool ret =
        reader.load_external_gt(external_gt_file, first_sep, second_sep, gt_);
    if (ret) {
      cout << "Load external ground truth file["
           << File::BaseName(external_gt_file) << "] done!" << endl;
      external_gt_file_enabled_ = true;
    } else {
      cerr << "Failed to load ground truth file!" << endl;
    }

    return ret;
  }

 private:
  std::string compute_crc() {
    uint32_t crc = 0u;
    // sparse
    if (linear_sparse_data_.size() > 0) {
      for (size_t i = 0; i < linear_sparse_data_.size(); ++i) {
        crc = Crc32c::Hash(&(linear_sparse_data_[i].count), sizeof(uint32_t),
                           crc);
        crc =
            Crc32c::Hash(linear_sparse_data_[i].indices.data(),
                         linear_sparse_data_[i].count * sizeof(uint32_t), crc);
        crc = Crc32c::Hash(linear_sparse_data_[i].features.data(),
                           linear_sparse_data_[i].count * sizeof(T), crc);
      }
    }

    char crc_str[64];
    snprintf(crc_str, sizeof(crc_str), "%X", crc);

    return std::string(crc_str);
  }

  // sparse search
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                const vector<uint32_t> &sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature, size_t count) {
    return flower->search_impl(sparse_count.data(), sparse_indices.data(),
                               sparse_feature.data(), qmeta_, count, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                const uint32_t sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature) {
    return flower->search_impl(sparse_count, sparse_indices.data(),
                               sparse_feature.data(), qmeta_, context);
  }

  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_linear_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                   const vector<uint32_t> &sparse_count,
                   const vector<uint32_t> &sparse_indices,
                   const vector<U> &sparse_feature, size_t count) {
    return flower->search_bf_impl(sparse_count.data(), sparse_indices.data(),
                                  sparse_feature.data(), qmeta_, count,
                                  context);
  }


  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_linear_search(SparseFlow *flower, SparseFlow::Context::Pointer &context,
                   const uint32_t sparse_count,
                   const vector<uint32_t> &sparse_indices,
                   const vector<U> &sparse_feature) {
    return flower->search_bf_impl(sparse_count, sparse_indices.data(),
                                  sparse_feature.data(), qmeta_, context);
  }

  void recall_one_sparse(
      SparseFlow *flower, size_t topk, size_t index,
      std::vector<pair<std::fstream *, std::fstream *>> &output_fs) {
    const auto &sparse_count = batch_sparse_counts_[index];
    const auto &sparse_index = batch_sparse_indices_[index];
    const auto &sparse_feature = batch_sparse_features_[index];

    size_t thread_index = pool_->indexof_this();
    fstream *knn_fs = nullptr;
    fstream *linear_fs = nullptr;
    if (output_fs.size() > thread_index) {
      knn_fs = output_fs[thread_index].first;
      linear_fs = output_fs[thread_index].second;
    }

    SparseFlow::Context::Pointer knn_context = flower->create_context();
    if (!knn_context) {
      cerr << "Failed to create search context" << endl;
      return;
    }
    knn_context->set_topk(topk);

    auto cal_recall = [&, this](const std::vector<IndexDocument> &knn_res,
                                size_t idx) {
      vector<IndexDocument> linear_res;

      size_t result_size = std::min(topk, gt_[idx].size());
      if (result_size == 0) {
        return;
      }

      for (size_t i = 0; i < result_size; ++i) {
        auto gt_node = gt_[idx][i];

        linear_res.emplace_back(gt_node.first, gt_node.second, gt_node.first);
      }

      if (knn_fs) {
        for (auto knn : knn_res) {
          string str = "query[" + to_string(idx) + "]\tkey[" +
                       to_string(knn.key()) + "], dist[" +
                       to_string(knn.score()) + "]\n";
          knn_fs->write(str.c_str(), str.size());
        }
      }

      size_t match = 0;
      bool asc =
          (linear_res.size() > 1 &&
           (linear_res[0].score() > linear_res[linear_res.size() - 1].score()))
              ? false
              : true;

      map<int32_t, size_t> topk_matchs;
      if (g_compare_by_id) {
        for (size_t i = 0; i < topk_ids_.size(); ++i) {
          topk_matchs[topk_ids_[i]] = 0;
        }
      }

      for (size_t i = 0, j = 0; i < linear_res.size();) {
        bool m = false;       // if current doc matched in max topk
        bool changed = true;  // if i changed
        if (g_compare_by_id) {
          for (size_t k = 0; k < topk_ids_.size(); ++k) {
            size_t dynamic_size = (size_t)topk_ids_[k];
            for (; dynamic_size + 1 < knn_res.size(); ++dynamic_size) {
              if (fabs(knn_res[dynamic_size - 1].score() -
                       knn_res[dynamic_size].score()) >=
                  numeric_limits<float>::epsilon()) {
                break;
              }
            }
            for (size_t l = 0; l < dynamic_size && l < knn_res.size(); ++l) {
              if (linear_res[i].key() == knn_res[l].key()) {
                topk_matchs[topk_ids_[k]]++;
                if (k == topk_ids_.size() - 1) {
                  m = true;
                }
                break;
              }
            }
          }
          ++i;

          auto it = recall_res_.find(i);
          if (it != recall_res_.end()) {
            lock_guard<mutex> lock(recall_lock);
            it->second += 100.0 * topk_matchs[i] / i;
          }
        } else {
          size_t cur_topk = i + 1;
          if (j < knn_res.size()) {
            if (fabs(linear_res[i].score() - knn_res[j].score()) <
                g_recall_precision) {
              ++j;
              ++i;
              match++;
              m = true;
            } else {
              if ((asc && linear_res[i].score() < knn_res[j].score()) ||
                  (!asc && linear_res[i].score() > knn_res[j].score())) {
                ++i;
              } else {
                changed = false;
                ++j;
              }
            }
          } else {
            ++i;
          }

          auto it = recall_res_.find(cur_topk);
          if (changed && it != recall_res_.end()) {
            lock_guard<mutex> lock(recall_lock);
            it->second += 100.0 * match / cur_topk;
          }
        }

        if (linear_fs && changed) {
          string str = string(m ? "    HIT" : "NOT HIT") + "  query[" +
                       to_string(idx) + "]\tkey[" +
                       to_string(linear_res[i - 1].key()) + "], dist[" +
                       to_string(linear_res[i - 1].score()) + "]\n";
          linear_fs->write(str.c_str(), str.size());
        }
      }
    };

    FilterResultCache filter_cache;
    if (filter_mode_ == FM_TAG) {
      if (batch_taglists_[index].size() != 1) {
        cerr << "query tag list not equal to one!" << endl;
        return;
      }

      int ret = filter_cache.filter(flower->id_to_tags_list(),
                                    batch_taglists_[index][0],
                                    flower->tag_key_list());
      if (ret != 0) {
        cerr << "prefilter failed, idx: " << index << std::endl;

        return;
      }

      auto filterFunc = [&](uint64_t key) { return filter_cache.find(key); };

      knn_context->set_filter(filterFunc);
    }

    if (call_batch_api_) {
      // size_t qnum = sparse_count.size() / dim_;
      // int ret = do_knn_search<T>(flower, knn_context, sparse_count,
      // sparse_index, sparse_feature, qnum); if (ret < 0) {
      //   cerr << "Failed to sparse_knn_search batch, ret=" << ret << " "
      //       << IndexError::What(ret) << endl;
      //   return;
      // }
      // for (size_t i = 0; i < qnum; ++i) {
      //   size_t idx = index * batch_count_ + i;

      //   if (idx >= linear_queries_.size()) {
      //     break;
      //   }

      //   auto &knn_res = knn_context->result(i);
      //   cal_recall(knn_res, idx);
      // }
    } else {
      int ret = do_knn_search<T>(flower, knn_context, sparse_count[0],
                                 sparse_index, sparse_feature);
      if (ret < 0) {
        cerr << "Failed to sparse_knn_search, ret=" << ret << " "
             << IndexError::What(ret) << endl;
        return;
      }
      auto &knn_res = knn_context->result();
      cal_recall(knn_res, index);
    }
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  bool call_batch_api_;
  string output_;
  size_t batch_count_;
  shared_ptr<ThreadPool> pool_;

  // for gt
  vector<vector<T>> linear_queries_;
  vector<SparseData<T>> linear_sparse_data_;
  vector<uint32_t> linear_partitions_;
  vector<vector<uint64_t>> linear_taglists_;

  std::map<std::string, vector<vector<T>>> linear_queries_scaled_;
  std::map<std::string, vector<vector<T>>> linear_sparse_features_scaled_;

  // for recall
  vector<vector<T>> batch_queries_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<uint32_t>> batch_partitions_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  std::map<std::string, vector<vector<T>>> batch_queries_scaled_;
  std::map<std::string, vector<vector<T>>> batch_sparse_features_scaled_;

  size_t total_querys_;

  map<size_t, float> recall_res_;
  vector<int32_t> topk_ids_;
  vector<vector<pair<uint64_t, float>>> gt_;

  map<string, vector<vector<pair<uint64_t, float>>>> gt_hybrid_;
  bool external_gt_file_enabled_{false};

  FilterMode filter_mode_{FM_NONE};
  static bool STOP_NOW;
};

template <typename T>
bool SparseRecall<T>::STOP_NOW = false;

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
  cout << "Usage: recall CONFIG.yaml [plugin file path]" << endl;
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

int recall_dense(std::string &query_type, size_t thread_count,
                 size_t batch_count, string top_k, size_t gt_count,
                 string query_file, string &first_sep, string &second_sep,
                 string &ground_truth_file, string &ground_truth_first_sep,
                 string ground_truth_second_sep, Flow &flower,
                 string &index_dir, string &log_dir, FilterMode filter_mode) {
  if (query_type == "float") {
    Recall<float> recall(thread_count, log_dir, batch_count, filter_mode);
    if (!recall.load_query(query_file, first_sep, second_sep)) {
      return -1;
    }

    if (ground_truth_file != "") {
      if (!recall.load_external_gt_file(ground_truth_file,
                                        ground_truth_first_sep,
                                        ground_truth_second_sep)) {
        return -1;
      }
    }

    if (load_index(flower, index_dir)) {
      recall.run_dense(&flower, top_k, gt_count);
    } else {
      return -1;
    }
  } else if (query_type == "int8") {
    Recall<int8_t> recall(thread_count, log_dir, batch_count, filter_mode);
    if (!recall.load_query(query_file, first_sep, second_sep)) {
      return -1;
    }

    if (ground_truth_file != "") {
      if (!recall.load_external_gt_file(ground_truth_file,
                                        ground_truth_first_sep,
                                        ground_truth_second_sep)) {
        return -1;
      }
    }

    if (load_index(flower, index_dir)) {
      recall.run_dense(&flower, top_k, gt_count);
    } else {
      return -1;
    }
  } else if (query_type == "binary") {
    Recall<uint32_t> recall(thread_count, log_dir, batch_count, filter_mode);
    if (!recall.load_query(query_file, first_sep, second_sep)) {
      return -1;
    }

    if (ground_truth_file != "") {
      if (!recall.load_external_gt_file(ground_truth_file,
                                        ground_truth_first_sep,
                                        ground_truth_second_sep)) {
        return -1;
      }
    }

    if (load_index(flower, index_dir)) {
      recall.run_dense(&flower, top_k, gt_count);
    } else {
      return -1;
    }
  } else if (query_type == "binary64") {
    Recall<uint64_t> recall(thread_count, log_dir, batch_count, filter_mode);
    if (!recall.load_query(query_file, first_sep, second_sep)) {
      return -1;
    }

    if (ground_truth_file != "") {
      if (!recall.load_external_gt_file(ground_truth_file,
                                        ground_truth_first_sep,
                                        ground_truth_second_sep)) {
        return -1;
      }
    }

    if (load_index(flower, index_dir)) {
      recall.run_dense(&flower, top_k, gt_count);
    } else {
      return -1;
    }
  } else {
    cerr << "Can not recognize type: " << query_type << endl;
  }

  return 0;
}

bool load_sparse_index(SparseFlow &flower, string &index_dir) {
  int ret = flower.load(index_dir);
  if (0 != ret) {
    cerr << "Flow load failed with ret " << ret << endl;
    return false;
  }
  cout << "Load index done!" << endl;

  return true;
};

int recall_sparse(std::string &query_type, size_t thread_count,
                  size_t batch_count, string top_k, size_t gt_count,
                  string &query_file, string &first_sep, string &second_sep,
                  string &ground_truth_file, string &ground_truth_first_sep,
                  string &ground_truth_second_sep, SparseFlow &flower,
                  string &index_dir, string &log_dir, FilterMode filter_mode) {
  if (query_type == "float") {
    SparseRecall<float> recall(thread_count, log_dir, batch_count, filter_mode);
    if (!recall.load_query(query_file, first_sep, second_sep)) {
      return -1;
    }

    if (ground_truth_file != "") {
      if (!recall.load_external_gt_file(ground_truth_file,
                                        ground_truth_first_sep,
                                        ground_truth_second_sep)) {
        return -1;
      }
    }

    if (load_sparse_index(flower, index_dir)) {
      recall.run_sparse(&flower, top_k, gt_count);
    } else {
      return -1;
    }
  } else {
    cerr << "Can not recognize type: " << query_type << endl;
  }

  return 0;
}

int get_recall_precision(string &recall_precision_string) {
  constexpr float DEFAULT_RECALL_PRECISION = 1e-6;

  if (recall_precision_string == "") {
    g_recall_precision = DEFAULT_RECALL_PRECISION;
    return true;
  }

  try {
    g_recall_precision = std::stof(recall_precision_string);
    std::cout << "Recall Score Precesion: " << g_recall_precision << std::endl;
  } catch (const std::invalid_argument &e) {
    std::cerr << "Exeception in getting recall precision: " << e.what()
              << ", value: " << recall_precision_string << std::endl;
    return false;
  } catch (const std::out_of_range &e) {
    std::cerr << "Out of range exception in getting recall precision: "
              << e.what() << ", value: " << recall_precision_string
              << std::endl;
    return false;
  }

  return true;
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

  // Calculate Recall
  string log_dir = "";
  if (config_common["RecallLogDir"]) {
    log_dir = config_common["RecallLogDir"].as<string>();
  }
  size_t thread_count = config_common["RecallThreadCount"]
                            ? config_common["RecallThreadCount"].as<uint64_t>()
                            : 0;
  size_t gt_count = config_common["RecallGTCount"]
                        ? config_common["RecallGTCount"].as<uint64_t>()
                        : 100;
  size_t batch_count = config_common["RecallBatchCount"]
                           ? config_common["RecallBatchCount"].as<uint64_t>()
                           : 0;
  g_compare_by_id = config_common["CompareById"]
                        ? config_common["CompareById"].as<bool>()
                        : 0;
  string top_k = config_common["TopK"].as<string>();

  string recall_precision_string =
      config_common["RecallScorePrecision"]
          ? config_common["RecallScorePrecision"].as<string>()
          : "";

  if (!get_recall_precision(recall_precision_string)) {
    cerr << "Get recall precision failed, value: " << recall_precision_string
         << endl;
    return -1;
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

  FilterMode filter_mode{FM_NONE};
  if (config_common["FilterMode"]) {
    std::string filter_mode_str = config_common["FilterMode"].as<string>();
    if (filter_mode_str == "tag") {
      filter_mode = FM_TAG;
    }
  }

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

  string ground_truth_file = "";
  string ground_truth_first_sep = ";";
  string ground_truth_second_sep = " ";

  if (config_common["GroundTruthFile"]) {
    ground_truth_file = config_common["GroundTruthFile"].as<string>();

    if (config_common["GroundTruthFirstSep"]) {
      ground_truth_first_sep =
          config_common["GroundTruthFirstSep"].as<string>();
    }

    if (config_common["GroundTruthSecondSep"]) {
      ground_truth_second_sep =
          config_common["GroundTruthSecondSep"].as<string>();
    }
  }

  if (retrieval_mode == RM_SPARSE) {
    SparseFlow flower;
    Params container_params;
    if (config_node["ContainerParams"]) {
      // Get index params of Searcher in flower object
      if (!prepare_params(config_node["ContainerParams"], container_params)) {
        return -1;
      }
      cout << "Created index params of a container in flower object " << endl;
    }

    int ret = flower.set_container(container_type, container_params);
    if (0 != ret) {
      cerr << "Create" << container_type << "failed." << endl;
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
    recall_sparse(query_type, thread_count, batch_count, top_k, gt_count,
                  query_file, first_sep, second_sep, ground_truth_file,
                  ground_truth_first_sep, ground_truth_second_sep, flower,
                  index_dir, log_dir, filter_mode);

    flower.unload();

    cout << "Recall done." << endl;
  } else {
    Flow flower;
    Params container_params;
    if (config_node["ContainerParams"]) {
      // Get index params of Searcher in flower object
      if (!prepare_params(config_node["ContainerParams"], container_params)) {
        return -1;
      }
      cout << "Created index params of a container in flower object " << endl;
    }

    int ret = flower.set_container(container_type, container_params);
    if (0 != ret) {
      cerr << "Create" << container_type << "failed." << endl;
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
    if (retrieval_mode == RM_DENSE) {
      recall_dense(query_type, thread_count, batch_count, top_k, gt_count,
                   query_file, first_sep, second_sep, ground_truth_file,
                   ground_truth_first_sep, ground_truth_second_sep, flower,
                   index_dir, log_dir, filter_mode);
    } else {
      std::string mode = retrieval_mode == 1 ? "Dense" : "Sparse";
      cerr << "unsupported retrieval mode: " << mode << endl;

      return -1;
    }

    // Cleanup
    flower.unload();

    cout << "Recall done." << endl;
  }

  return 0;
}
