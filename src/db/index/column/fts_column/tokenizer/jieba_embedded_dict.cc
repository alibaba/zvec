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

#include "jieba_embedded_dict.h"
#include <lz4.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>
#include <vector>

namespace zvec::fts {
namespace {

#include "jieba_dict_lz4.inc"
#include "jieba_hmm_model_lz4.inc"

// Reference the decompressed text without istringstream's C++17 string copy.
// The caller keeps the text alive until parsing has finished.
class MemoryInputBuffer : public std::streambuf {
 public:
  explicit MemoryInputBuffer(std::string &text) {
    char *begin = text.data();
    setg(begin, begin, begin + text.size());
  }
};

template <std::size_t N>
Result<std::string> decompress(const unsigned char (&data)[N],
                               std::size_t original_size) {
  static_assert(N <= std::numeric_limits<int>::max());
  if (original_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return tl::make_unexpected(Status::InternalError(
        "Embedded Jieba resource exceeds LZ4 size limit"));
  }
  std::string text(original_size, '\0');
  int size =
      LZ4_decompress_safe(reinterpret_cast<const char *>(data), text.data(),
                          static_cast<int>(N), static_cast<int>(original_size));
  if (size < 0 || static_cast<std::size_t>(size) != original_size) {
    return tl::make_unexpected(
        Status::InternalError("Failed to decompress embedded Jieba resource"));
  }
  return text;
}

uint32_t read_u16(const unsigned char *p) {
  return uint32_t{p[0]} | (uint32_t{p[1]} << 8);
}

uint32_t read_u32(const unsigned char *p) {
  return read_u16(p) | (read_u16(p + 2) << 16);
}

Result<EmbeddedJiebaDict> load_compiled_dictionary() {
  auto data = decompress(kJiebaDictLz4, kJiebaDictLz4OriginalSize);
  if (!data) {
    return tl::make_unexpected(data.error());
  }
  // The generator writes columns without padding, independent of the target
  // ABI. Only DictUnit and the trie need to be materialized at first use.
  static_assert(kJiebaWordCount * 8ULL + kJiebaRuneCount * 4ULL <=
                kJiebaDictLz4OriginalSize);
  const auto *word_lengths =
      reinterpret_cast<const unsigned char *>(data->data());
  const auto *tag_lengths = word_lengths + 2 * kJiebaWordCount;
  const auto *frequencies = tag_lengths + 2 * kJiebaWordCount;
  const auto *word = frequencies + 4 * kJiebaWordCount;
  const auto *word_end = word + 4 * kJiebaRuneCount;
  const auto *tag = word_end;
  const auto *tag_end = word_lengths + data->size();
  auto entries = std::make_shared<std::vector<cppjieba::DictUnit>>();
  entries->reserve(kJiebaWordCount);
  for (size_t i = 0; i < kJiebaWordCount; ++i) {
    const size_t runes = read_u16(word_lengths + 2 * i);
    const size_t tag_size = read_u16(tag_lengths + 2 * i);
    const auto frequency = read_u32(frequencies + 4 * i);
    if (runes == 0 || frequency == 0 ||
        runes * 4 > static_cast<size_t>(word_end - word) ||
        tag_size > static_cast<size_t>(tag_end - tag)) {
      return tl::make_unexpected(
          Status::InternalError("Invalid embedded Jieba dictionary entry"));
    }
    auto &entry = entries->emplace_back();
    entry.word.reserve(runes);
    for (size_t j = 0; j < runes; ++j, word += 4) {
      entry.word.push_back(read_u32(word));
    }
    entry.tag.assign(reinterpret_cast<const char *>(tag), tag_size);
    tag += tag_size;
    entry.weight = std::log(frequency / kJiebaFrequencySum);
  }
  if (word != word_end || tag != tag_end) {
    return tl::make_unexpected(
        Status::InternalError("Invalid embedded Jieba dictionary size"));
  }
  auto cache = std::make_shared<cppjieba::DictTrie::DictCacheEntry>();
  cache->node_infos = std::move(entries);
  cache->freq_sum = kJiebaFrequencySum;
  cache->min_weight = std::log(kJiebaMinFrequency / kJiebaFrequencySum);
  cache->median_weight = std::log(kJiebaMedianFrequency / kJiebaFrequencySum);
  cache->max_weight = std::log(kJiebaMaxFrequency / kJiebaFrequencySum);
  return EmbeddedJiebaDict(std::move(cache));
}

}  // namespace

Result<EmbeddedJiebaDict> get_embedded_jieba_dict() {
  static const auto dict = load_compiled_dictionary();
  return dict;
}

Result<std::shared_ptr<const cppjieba::DictTrie>> get_embedded_jieba_trie() {
  using Trie = std::shared_ptr<const cppjieba::DictTrie>;
  static const Result<Trie> trie = []() -> Result<Trie> {
    auto dict = get_embedded_jieba_dict();
    if (!dict) {
      return tl::make_unexpected(dict.error());
    }
    return std::make_shared<const cppjieba::DictTrie>(**dict);
  }();
  return trie;
}

Result<EmbeddedJiebaHmmModel> get_embedded_jieba_hmm_model() {
  static const Result<EmbeddedJiebaHmmModel> model =
      []() -> Result<EmbeddedJiebaHmmModel> {
    auto text = decompress(kJiebaHmmModelLz4, kJiebaHmmModelLz4OriginalSize);
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    MemoryInputBuffer buffer(*text);
    std::istream stream(&buffer);
    return std::make_shared<const cppjieba::HMMModel>(stream);
  }();
  return model;
}

}  // namespace zvec::fts
