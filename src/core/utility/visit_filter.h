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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>
#include <ailego/container/bloom_filter.h>
#include <ailego/utility/bitset_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace core {

struct VisitFilterHeader {
  VisitFilterHeader() : maxDocCnt(0), maxScanNum(0) {}
  uint64_t maxDocCnt;
  uint64_t maxScanNum;
};

constexpr int PROXIMA_HNSW_VISITFILTER_CUSTOM_PARAMS_INDEX_NEGPROB = 0;

class VisitBloomFilter {
 public:
  static constexpr int mode = 1;

  static constexpr int N = 5;
  struct Context {
    Context()
        : mt(std::chrono::system_clock::now().time_since_epoch().count()) {};
    VisitFilterHeader h;
    std::mt19937 mt;
    ailego::BloomFilter<N> *filter{nullptr};
    int offset[N] = {0};
  };
#define BLOOM_FILTER_HASH_BITS_OFFSETS(i)                                 \
  i + c->offset[0], i + c->offset[1], i + c->offset[2], i + c->offset[3], \
      i + c->offset[4]

  VisitBloomFilter() = delete;

  inline static void set_visited(Context *c, id_t idx) {
    c->filter->force_insert(BLOOM_FILTER_HASH_BITS_OFFSETS(idx));
    return;
  }

  inline static void *get_visited(Context *, id_t) {
    // TODO
    return nullptr;
  }

  inline static bool visited(Context *c, id_t idx) {
    return c->filter->has(BLOOM_FILTER_HASH_BITS_OFFSETS(idx));
  }

  inline static int set_max_scan_num(Context *c, uint64_t maxScanNum) {
    if (maxScanNum == c->h.maxScanNum) {
      return 0;
    }
    c->h.maxScanNum = maxScanNum;
    if (c->filter->reset(maxScanNum, c->filter->probability()) != 0) {
      LOG_ERROR("reset BloomFilter failed");
      return IndexError_Runtime;
    }
    genRandomHashBits(c);
    return 0;
  }

  inline static void clear(Context *c) {
    c->filter->clear();
    return;
  }

  inline static bool reset(Context *c, uint64_t maxDocCnt,
                           uint64_t max_scan_num) {
    if (ailego_unlikely(maxDocCnt > c->h.maxDocCnt ||
                        max_scan_num > c->h.maxScanNum)) {
      // Create a new one, if failed, we can reuse the old one
      auto filter = new (std::nothrow) ailego::BloomFilter<VisitBloomFilter::N>(
          max_scan_num, c->filter->probability());
      if (ailego_unlikely(filter == nullptr)) {
        LOG_ERROR("reset bloomfilter failed, maxScanNum %zu prob %f",
                  (size_t)max_scan_num, c->filter->probability());
        c->filter->clear();
        return false;
      }

      delete c->filter;
      c->filter = filter;
      c->h.maxScanNum = max_scan_num;
      c->h.maxDocCnt = maxDocCnt;
      genRandomHashBits(c);
    }
    return true;
  }

  inline static void genRandomHashBits(Context *c) {
    std::uniform_int_distribution<int> dt(0, c->h.maxDocCnt);
    for (size_t i = 0; i < sizeof(c->offset) / sizeof(c->offset[0]); ++i) {
      int r = dt(c->mt);
      size_t j = 0;
      do {  // gen distinct number
        for (j = 0; j < i; ++j) {
          if (c->offset[j] == r) {
            r = dt(c->mt);
            break;
          }
        }
      } while (j < i);
      c->offset[i] = r;
    }
    std::sort(c->offset, c->offset + N);
  }

  template <class... T>
  static int init(Context *, void **ctx, uint64_t maxDocCnt,
                  uint64_t maxScanNum, std::tuple<T...> &&tpl) {
    Context *c = new (std::nothrow) Context;
    if (c == nullptr) {
      LOG_ERROR("New memory in initVisitBitMap failed");
      return IndexError_NoMemory;
    }
    c->h.maxDocCnt = maxDocCnt;
    c->h.maxScanNum = maxScanNum;
    float p =
        std::get<PROXIMA_HNSW_VISITFILTER_CUSTOM_PARAMS_INDEX_NEGPROB>(tpl);
    c->filter = new (std::nothrow)
        ailego::BloomFilter<VisitBloomFilter::N>(maxScanNum, p);
    if (c->filter == nullptr) {
      LOG_ERROR("New BloomFilter failed, reuse old one");
      return IndexError_NoMemory;
    }
    genRandomHashBits(c);
    *ctx = c;
    return 0;
  }

  inline static void destroy(Context *c) {
    delete c->filter;
    delete c;
  }
#undef BLOOM_FILTER_HASH_BITS_OFFSETS
};  // end of VisitBloomFilter

class VisitBitMap {
 public:
  static constexpr int mode = 2;

  struct Context {
    VisitFilterHeader h;
    ailego::BitsetHelper bitset;
    char *buf{nullptr};
  };

  VisitBitMap() = delete;

  inline static void set_visited(Context *c, id_t idx) {
    c->bitset.set(idx);
    return;
  }

  inline static void *get_visited(Context *c, id_t idx) {
    return &c->buf[idx >> 3];
  }

  inline static bool visited(Context *c, id_t idx) {
    return c->bitset.test(idx);
  }

  inline static int set_max_scan_num(Context *c, uint64_t maxScanNum) {
    c->h.maxScanNum = maxScanNum;
    return 0;
  }

  inline static void clear(Context *c) {
    c->bitset.clear();
    return;
  }

  inline static bool reset(Context *c, uint64_t maxDocCnt,
                           uint64_t maxScanNum) {
    if (ailego_unlikely(maxDocCnt > c->h.maxDocCnt ||
                        maxScanNum > c->h.maxScanNum)) {
      uint64_t len = ((maxDocCnt + 31) >> 5) << 2;  // round to uint32_t
      auto buf = new (std::nothrow) char[len];
      if (buf == nullptr) {
        LOG_ERROR("New memory in initVisitBitMap failed");
        c->bitset.clear();
        return false;
      }

      c->h.maxDocCnt = maxDocCnt;
      c->h.maxScanNum = maxScanNum;
      delete[] c->buf;
      c->buf = buf;
      memset(c->buf, 0, len);
      c->bitset.mount(c->buf, len);
    }
    return true;
  }

  template <class... T>
  static int init(Context *, void **ctx, uint64_t maxDocCnt,
                  uint64_t maxScanNum, std::tuple<T...> &&tpl) {
    (void)tpl;  // unused warning
    Context *c = new (std::nothrow) Context;
    if (c == nullptr) {
      LOG_ERROR("New memory in initVisitBitMap failed");
      return IndexError_NoMemory;
    }
    c->h.maxDocCnt = maxDocCnt;
    c->h.maxScanNum = maxScanNum;
    uint64_t len = ((maxDocCnt + 31) >> 5) << 2;  // round to uint32_t
    c->buf = new (std::nothrow) char[len];
    if (c->buf == nullptr) {
      LOG_ERROR("New memory in initVisitBitMap failed, reuse old one");
      delete c;
      return IndexError_NoMemory;
    }
    memset(c->buf, 0, len);
    c->bitset.mount(c->buf, len);
    *ctx = c;
    return 0;
  }

  inline static void destroy(Context *c) {
    delete[] c->buf;
    delete c;
  }
};  // end of VisitBitMap

class VisitByteMap {
 public:
  static constexpr int mode = 3;
  struct Context {
    VisitFilterHeader h;
    uint8_t curNum{0};
    std::vector<uint8_t> buf;
  };

  VisitByteMap() = delete;

  inline static void set_visited(Context *c, id_t idx) {
    if (ailego_unlikely(idx > c->h.maxDocCnt)) {
      c->h.maxDocCnt = idx + 1024;  // reserved
      c->buf.resize(c->h.maxDocCnt);
    }
    c->buf[idx] = c->curNum;
    return;
  }

  inline static void *get_visited(Context *c, id_t idx) {
    return c->buf.data() + idx;
  }

  inline static bool visited(Context *c, id_t idx) {
    if (ailego_unlikely(idx > c->h.maxDocCnt)) {
      return false;
    }
    return c->buf[idx] == c->curNum;
  }

  inline static int set_max_scan_num(Context *c, uint64_t maxScanNum) {
    c->h.maxScanNum = maxScanNum;
    return 0;
  }

  inline static void clear(Context *c) {
    c->curNum++;
    if (c->curNum == 0) {
      memset(c->buf.data(), 0, c->h.maxDocCnt * sizeof(uint8_t));
      c->curNum = 1;
    }
    return;
  }

  inline static bool reset(Context *c, uint64_t maxDocCnt,
                           uint64_t maxScanNum) {
    if (ailego_unlikely(maxDocCnt > c->h.maxDocCnt ||
                        maxScanNum > c->h.maxScanNum)) {
      try {
        c->buf.resize(maxDocCnt);
      } catch (const std::exception &e) {
        LOG_ERROR("New memory in initVisitByteMap failed, reuse old one");
        return false;
      }
      memset(c->buf.data(), 0, maxDocCnt * sizeof(uint8_t));
      c->curNum = 1;
      c->h.maxDocCnt = maxDocCnt;
      c->h.maxScanNum = maxScanNum;
      return true;
    }
    return true;
  }

  template <class... T>
  static int init(Context *, void **ctx, uint64_t maxDocCnt,
                  uint64_t maxScanNum, std::tuple<T...> &&tpl) {
    (void)tpl;  // unused warning
    Context *c = new (std::nothrow) Context;
    if (c == nullptr) {
      LOG_ERROR("New memory in initVisitByteMap failed");
      return IndexError_NoMemory;
    }
    c->h.maxDocCnt = maxDocCnt;
    c->h.maxScanNum = maxScanNum;
    try {
      c->buf.resize(maxDocCnt);
    } catch (const std::exception &e) {
      LOG_ERROR("New memory in initVisitByteMap failed");
      delete c;
      return IndexError_NoMemory;
    }
    memset(c->buf.data(), 0, maxDocCnt * sizeof(uint8_t));
    c->curNum = 1;
    *ctx = c;
    return 0;
  }

  inline static void destroy(Context *c) {
    delete c;
  }
};  // end of VisitByteMap


#define PROXIMA_HNSW_VISITFILTER_SWITCH_CASE(cls, impl, ctx, ...) \
  case cls::mode:                                                 \
    return cls::impl(static_cast<cls::Context *>(ctx), ##__VA_ARGS__);

#define PROXIMA_HNSW_VISITFILTER_CALL_IMPL(impl, ...)                  \
  switch (mode_) {                                                     \
    PROXIMA_HNSW_VISITFILTER_SWITCH_CASE(VisitBloomFilter, impl, ctx_, \
                                         ##__VA_ARGS__)                \
    PROXIMA_HNSW_VISITFILTER_SWITCH_CASE(VisitBitMap, impl, ctx_,      \
                                         ##__VA_ARGS__)                \
    PROXIMA_HNSW_VISITFILTER_SWITCH_CASE(VisitByteMap, impl, ctx_,     \
                                         ##__VA_ARGS__)                \
  }


// visit list will be called with high frequency,
// so using switch instead of std::function or virtual class
// funtion point, lambda, virtual class all cannot be inlined
class VisitFilter {
 public:
  enum Mode {
    Default = 0,
    BloomFilter = VisitBloomFilter::mode,
    BitMap = VisitBitMap::mode,
    ByteMap = VisitByteMap::mode
  };

  VisitFilter() : mode_(0), ctx_(nullptr) {};

  inline bool visited(id_t idx) {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(visited, idx);
    return true;  // place holder
  }

  inline void set_visited(id_t idx) {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(set_visited, idx);
  }

  inline void *get_visited(id_t idx) {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(get_visited, idx);
    return nullptr;  // place holder
  }

  inline int set_max_scan_num(id_t idx) {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(set_max_scan_num, idx);
    return 0;  // place holder
  }

  inline void clear() {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(clear);
  }

  inline bool reset(uint64_t maxDocCnt, uint64_t maxScanNum) {
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(reset, maxDocCnt, maxScanNum);
    return true;
  }

  inline void destroy() {
    if (ctx_ != nullptr) {
      PROXIMA_HNSW_VISITFILTER_CALL_IMPL(destroy);
    }
  }

  int init(int mode, uint64_t maxDocCnt, uint64_t maxScanNum,
           float negativeProbability) {
    mode_ = mode;
    PROXIMA_HNSW_VISITFILTER_CALL_IMPL(init, &ctx_, maxDocCnt, maxScanNum,
                                       std::make_tuple(negativeProbability));
    return 0;  // place holder
  }

  int get_mode(void) const {
    return mode_;
  }


 private:
  VisitFilter(const VisitFilter &) = delete;
  VisitFilter &operator=(const VisitFilter &) = delete;

  int mode_{0U};  // custom data for each method
  void *ctx_{nullptr};
};

}  // namespace core
}  // namespace zvec
