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

#include "zvec/core/framework/index_flow.h"
#include "meta_segment_common.h"

using namespace std;

namespace zvec {
namespace core {

#define SEARCH_DENSE_BATCH(_FUNC_NAME)                                         \
  int _FUNC_NAME(const void *query, const IndexQueryMeta &qmeta,               \
                 uint32_t count, Context::Pointer &context) const {            \
    if (streamer_) {                                                           \
      if (reformer_) {                                                         \
        std::string ovec;                                                      \
        IndexQueryMeta ometa;                                                  \
        int ret = reformer_->convert(query, qmeta, count, &ovec, &ometa);      \
        if (ret != 0) {                                                        \
          return ret;                                                          \
        }                                                                      \
        return streamer_->_FUNC_NAME(ovec.data(), ometa, count,                \
                                     context->context());                      \
      } else {                                                                 \
        return streamer_->_FUNC_NAME(query, qmeta, count, context->context()); \
      }                                                                        \
    } else {                                                                   \
      return flow_._FUNC_NAME(query, qmeta, count, context->flow_context());   \
    }                                                                          \
  }

#define SEARCH_DENSE(_FUNC_NAME)                                              \
  int _FUNC_NAME(const void *query, const IndexQueryMeta &qmeta,              \
                 Context::Pointer &context) const {                           \
    if (streamer_) {                                                          \
      if (reformer_) {                                                        \
        std::string ovec;                                                     \
        IndexQueryMeta ometa;                                                 \
        int ret = reformer_->convert(query, qmeta, &ovec, &ometa);            \
        if (ret != 0) {                                                       \
          return ret;                                                         \
        }                                                                     \
        return streamer_->_FUNC_NAME(ovec.data(), ometa, context->context()); \
      } else {                                                                \
        return streamer_->_FUNC_NAME(query, qmeta, context->context());       \
      }                                                                       \
    } else {                                                                  \
      return flow_._FUNC_NAME(query, qmeta, context->flow_context());         \
    }                                                                         \
  }

#define SEARCH_SPRASE_BATCH(_FUNC_NAME)                                        \
  int _FUNC_NAME(const uint32_t *sparse_count, const uint32_t *sparse_indices, \
                 const void *sparse_query, const IndexQueryMeta &qmeta,        \
                 uint32_t count, Context::Pointer &context) const {            \
    if (streamer_) {                                                           \
      if (reformer_) {                                                         \
        LOG_ERROR("reformer not supported in sparse search");                  \
        return IndexError_Runtime;                                             \
      } else {                                                                 \
        return streamer_->_FUNC_NAME(sparse_count, sparse_indices,             \
                                     sparse_query, qmeta, count,               \
                                     context->context());                      \
      }                                                                        \
    } else {                                                                   \
      return flow_._FUNC_NAME(sparse_count, sparse_indices, sparse_query,      \
                              qmeta, count, context->flow_context());          \
    }                                                                          \
  }

#define SEARCH_SPARSE(_FUNC_NAME)                                              \
  int _FUNC_NAME(const uint32_t sparse_count, const uint32_t *sparse_indices,  \
                 const void *sparse_query, const IndexQueryMeta &qmeta,        \
                 Context::Pointer &context) const {                            \
    if (streamer_) {                                                           \
      if (reformer_) {                                                         \
        LOG_ERROR("reformer not supported in sparse search");                  \
        return IndexError_Runtime;                                             \
      } else {                                                                 \
        return streamer_->_FUNC_NAME(sparse_count, sparse_indices,             \
                                     sparse_query, qmeta, context->context()); \
      }                                                                        \
    } else {                                                                   \
      return flow_._FUNC_NAME(sparse_count, sparse_indices, sparse_query,      \
                              qmeta, context->flow_context());                 \
    }                                                                          \
  }

class Flow {
 public:
  class Context {
   public:
    typedef std::unique_ptr<Context> Pointer;

    Context(IndexContext::Pointer &ctx, IndexFlow::Context::Pointer &flow_ctx)
        : ctx_(std::move(ctx)), flow_ctx_(std::move(flow_ctx)) {}

    void set_debug_mode(bool debug_mode) {
      ctx_ ? ctx_->set_debug_mode(debug_mode)
           : flow_ctx_->set_debug_mode(debug_mode);
    }

    std::string debug_string() {
      return ctx_ ? ctx_->debug_string() : flow_ctx_->debug_string();
    }

    void set_topk(uint32_t topk) {
      ctx_ ? ctx_->set_topk(topk) : flow_ctx_->set_topk(topk);
    }

    template <typename T>
    void set_filter(T &&func) {
      ctx_ ? ctx_->set_filter(func) : flow_ctx_->set_filter(func);
    }

    const IndexDocumentList &result(void) const {
      return ctx_ ? ctx_->result() : flow_ctx_->result();
    }

    const IndexDocumentList &result(size_t index) const {
      return ctx_ ? ctx_->result(index) : flow_ctx_->result(index);
    }

   public:
    friend class Flow;

    IndexFlow::Context::Pointer &flow_context(void) {
      return flow_ctx_;
    }

    IndexContext::Pointer &context(void) {
      return ctx_;
    }

   private:
    IndexContext::Pointer ctx_;
    IndexFlow::Context::Pointer flow_ctx_;
  };

  Context::Pointer create_context(void) const {
    IndexContext::Pointer ctx;
    IndexFlow::Context::Pointer flow_ctx;
    if (streamer_) {
      ctx = streamer_->create_context();
    } else {
      flow_ctx = flow_.create_context();
    }
    return Context::Pointer(new (std::nothrow) Context(ctx, flow_ctx));
  }

  int set_container(const std::string &name, const ailego::Params &params) {
    return flow_.set_storage(name, params);
  }

  int load_taglists(const std::string &path) {
    // load tag lists
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");

    int ret = storage->open(path, false);
    if (ret != 0) {
      LOG_ERROR("Failed to load index with storage %s",
                storage->name().c_str());
      return ret;
    }

    auto segment_taglist_header = storage->get(TAGLIST_HEADER_SEGMENT_NAME);
    if (!segment_taglist_header) {
      LOG_INFO("No Tag Lists Found!");

      return 0;
    }

    TagListHeader taglist_header;
    void *data_ptr;
    if (segment_taglist_header->read(0, (const void **)(&data_ptr),
                                     sizeof(TagListHeader)) !=
        sizeof(TagListHeader)) {
      LOG_ERROR("Read tag list meta failed");
      return IndexError_ReadData;
    }

    memcpy(&taglist_header, data_ptr, sizeof(TagListHeader));

    auto segment_taglist_key = storage->get(TAGLIST_KEY_SEGMENT_NAME);
    if (!segment_taglist_key) {
      LOG_ERROR("IndexStorage get segment %s failed",
                TAGLIST_KEY_SEGMENT_NAME.c_str());
      return IndexError_InvalidValue;
    }

    size_t offset = 0;
    for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
      if (segment_taglist_key->read(offset, (const void **)(&data_ptr),
                                    sizeof(uint64_t)) != sizeof(uint64_t)) {
        LOG_ERROR("Read tag list key failed");
        return IndexError_ReadData;
      }

      uint64_t key = *reinterpret_cast<const uint64_t *>(data_ptr);
      tag_key_list_.push_back(key);

      offset += sizeof(uint64_t);
    }

    auto segment_taglist_data = storage->get(TAGLIST_DATA_SEGMENT_NAME);
    if (!segment_taglist_data) {
      LOG_ERROR("IndexStorage get segment %s failed",
                TAGLIST_DATA_SEGMENT_NAME.c_str());
      return IndexError_InvalidValue;
    }

    std::vector<uint64_t> taglist_offsets;
    offset = 0;
    for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
      if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                     sizeof(uint64_t)) != sizeof(uint64_t)) {
        LOG_ERROR("Read tag list data failed");
        return IndexError_ReadData;
      }

      uint64_t tag_offset = *reinterpret_cast<const uint64_t *>(data_ptr);
      taglist_offsets.push_back(tag_offset);

      offset += sizeof(uint64_t);
    }

    offset = taglist_header.num_vecs * sizeof(uint64_t);
    for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
      if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                     sizeof(uint64_t)) != sizeof(uint64_t)) {
        LOG_ERROR("Read tag list data failed");
        return IndexError_ReadData;
      }
      offset += sizeof(uint64_t);

      uint64_t tag_count = *reinterpret_cast<const uint64_t *>(data_ptr);

      if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                     tag_count * sizeof(uint64_t)) !=
          tag_count * sizeof(uint64_t)) {
        LOG_ERROR("Read tag list data failed");
        return IndexError_ReadData;
      }
      offset += tag_count * sizeof(uint64_t);

      std::vector<uint64_t> tag_list;
      for (size_t j = 0; j < tag_count; ++j) {
        uint64_t tag_id = *(reinterpret_cast<const uint64_t *>(data_ptr) + j);
        tag_list.push_back(tag_id);
      }

      // order tags
      sort(tag_list.begin(), tag_list.end());

      id_to_tags_list_.push_back(std::move(tag_list));
    }

    storage->cleanup();
    storage = nullptr;

    return 0;
  }

  int load(const std::string &path) {
    int ret = load_taglists(path);
    if (ret != 0) {
      LOG_ERROR("Failed to load tag lists");
      return ret;
    }

    if (streamer_) {
      stg_ = IndexFactory::CreateStorage("MMapFileStorage");
      if (!stg_) {
        return IndexError_NoExist;
      }
      ailego::Params params;
      params.set("proxima.mmap_file.storage.memory_warmup", true);
      ret = stg_->init(params);
      if (ret != 0) {
        return ret;
      }
      ret = stg_->open(path, true);
      if (ret != 0) {
        return ret;
      }

      if (!inited_) {
        IndexMeta meta;
        ret = IndexHelper::DeserializeFromStorage(stg_.get(), &meta);
        if (ret != 0) {
          LOG_ERROR("Failed to get IndexMeta from Storage");
          return ret;
        }
        ret = streamer_->init(meta, searcher_params_);
        if (ret != 0) {
          return ret;
        }

        if (!meta.reformer_name().empty()) {
          reformer_ = IndexFactory::CreateReformer(meta.reformer_name());
          if (!reformer_) {
            LOG_ERROR("Failed to create reformer %s",
                      meta.reformer_name().c_str());
            return IndexError_NoExist;
          }
          reformer_->init(meta.reformer_params());
        }
      }

      return streamer_->open(stg_);
    } else {
      return flow_.load(path);
    }
  }

  int unload(void) {
    if (streamer_) {
      streamer_->close();
      return stg_->close();
    } else {
      return flow_.unload();
    }
  }

  int set_searcher(const std::string &name, const ailego::Params &params) {
    //! If the searcher is streamer, create it
    streamer_ = IndexFactory::CreateStreamer(name);
    if (!streamer_) {
      return flow_.set_searcher(name, params);
    }
    searcher_params_ = params;
    return 0;
  }

  int set_searcher(IndexStreamer::Pointer streamer) {
    streamer_ = streamer;

    inited_ = true;

    return 0;
  }

  const std::vector<std::vector<uint64_t>> &id_to_tags_list() const {
    return id_to_tags_list_;
  }

  const std::vector<uint64_t> &tag_key_list() const {
    return tag_key_list_;
  }

  SEARCH_DENSE_BATCH(search_impl);
  SEARCH_DENSE(search_impl);
  SEARCH_DENSE_BATCH(search_bf_impl);
  SEARCH_DENSE(search_bf_impl);

 private:
  IndexFlow flow_{};

  IndexStreamer::Pointer streamer_{};
  IndexReformer::Pointer reformer_{};

  bool inited_{false};

  IndexStorage::Pointer stg_{};
  ailego::Params searcher_params_{};
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;
};

class SparseFlow {
 public:
  class Context {
   public:
    typedef std::unique_ptr<Context> Pointer;

    Context(IndexContext::Pointer &ctx,
            IndexSparseFlow::Context::Pointer &flow_ctx)
        : ctx_(std::move(ctx)), flow_ctx_(std::move(flow_ctx)) {}

    void set_debug_mode(bool debug_mode) {
      ctx_ ? ctx_->set_debug_mode(debug_mode)
           : flow_ctx_->set_debug_mode(debug_mode);
    }

    std::string debug_string() {
      return ctx_ ? ctx_->debug_string() : flow_ctx_->debug_string();
    }

    template <typename T>
    void set_filter(T &&func) {
      ctx_ ? ctx_->set_filter(func) : flow_ctx_->set_filter(func);
    }

    void set_topk(uint32_t topk) {
      ctx_ ? ctx_->set_topk(topk) : flow_ctx_->set_topk(topk);
    }

    const IndexDocumentList &result(void) const {
      return ctx_ ? ctx_->result() : flow_ctx_->result();
    }

    const IndexDocumentList &result(size_t index) const {
      return ctx_ ? ctx_->result(index) : flow_ctx_->result(index);
    }

   private:
    friend class SparseFlow;

    IndexSparseFlow::Context::Pointer &flow_context(void) {
      return flow_ctx_;
    }

    IndexContext::Pointer &context(void) {
      return ctx_;
    }


   private:
    IndexContext::Pointer ctx_;
    IndexSparseFlow::Context::Pointer flow_ctx_;
  };

  Context::Pointer create_context(void) const {
    IndexContext::Pointer ctx;
    IndexSparseFlow::Context::Pointer flow_ctx;
    if (streamer_) {
      ctx = streamer_->create_context();
    } else {
      flow_ctx = flow_.create_context();
    }
    return Context::Pointer(new (std::nothrow) Context(ctx, flow_ctx));
  }

  int set_container(const std::string &name, const ailego::Params &params) {
    return flow_.set_storage(name, params);
  }

  int load(const std::string &path) {
    if (streamer_) {
      stg_ = IndexFactory::CreateStorage("MMapFileStorage");
      if (!stg_) {
        return IndexError_NoExist;
      }
      ailego::Params params;
      params.set("proxima.mmap_file.storage.memory_warmup", true);
      int ret = stg_->init(params);
      if (ret != 0) {
        return ret;
      }
      ret = stg_->open(path, true);
      if (ret != 0) {
        return ret;
      }

      if (!inited_) {
        IndexMeta meta;
        ret = IndexHelper::DeserializeFromStorage(stg_.get(), &meta);
        if (ret != 0) {
          LOG_ERROR("Failed to get IndexMeta from Storage");
          return ret;
        }

        ret = streamer_->init(meta, searcher_params_);
        if (ret != 0) {
          return ret;
        }

        if (!meta.reformer_name().empty()) {
          reformer_ = IndexFactory::CreateReformer(meta.reformer_name());
          if (!reformer_) {
            LOG_ERROR("Failed to create reformer %s",
                      meta.reformer_name().c_str());
            return IndexError_NoExist;
          }
          reformer_->init(meta.reformer_params());
        }
      }

      return streamer_->open(stg_);
    } else {
      return flow_.load(path);
    }

    return 0;
  }

  int unload(void) {
    if (streamer_) {
      streamer_->close();
      return stg_->close();
    } else {
      return flow_.unload();
    }
  }

  int set_searcher(const std::string &name, const ailego::Params &params) {
    //! If the searcher is streamer, create it
    streamer_ = IndexFactory::CreateStreamer(name);
    if (!streamer_) {
      return flow_.set_searcher(name, params);
    }
    searcher_params_ = params;
    return 0;
  }

  int set_searcher(IndexStreamer::Pointer streamer) {
    streamer_ = streamer;

    inited_ = true;

    return 0;
  }

  const std::vector<std::vector<uint64_t>> &id_to_tags_list() const {
    return id_to_tags_list_;
  }

  const std::vector<uint64_t> &tag_key_list() const {
    return tag_key_list_;
  }

  SEARCH_SPRASE_BATCH(search_impl);
  SEARCH_SPARSE(search_impl);
  SEARCH_SPRASE_BATCH(search_bf_impl);
  SEARCH_SPARSE(search_bf_impl);

 private:
  IndexSparseFlow flow_{};

  IndexStreamer::Pointer streamer_{};
  IndexReformer::Pointer reformer_{};

  bool inited_{false};

  IndexStorage::Pointer stg_{};
  ailego::Params searcher_params_{};
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;
};

}  // namespace core
}  // namespace zvec
