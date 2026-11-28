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
#include <algorithm>
#include <array>
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

class EmbeddedTrie final : public cppjieba::DictTrie::ReadOnlyTrie {
 public:
  explicit EmbeddedTrie(const std::vector<cppjieba::DictUnit> &entries)
      : entries_(entries) {}

  Status build() {
    static_assert(kJiebaRootSize <= 0x110000);
    auto next = kJiebaLevelOffsets;
    nodes_.resize(static_cast<size_t>(next.back()) + 1);
    nodes_.front().first_child = 1;
    nodes_.back().first_child = next.back();

    const cppjieba::Unicode empty;
    const auto *previous = &empty;
    for (const auto &[first, end] : kJiebaSortedRowRanges) {
      if (end > entries_.size()) {
        return Status::InternalError("Invalid embedded Jieba row range");
      }
      for (uint32_t row = first; row < end; ++row) {
        const auto &word = entries_[row].word;
        if (word.empty() || word.size() >= next.size()) {
          return Status::InternalError("Invalid embedded Jieba word length");
        }
        const auto mismatch = std::mismatch(word.begin(), word.end(),
                                            previous->begin(), previous->end());
        // In lexical order only the suffix introduces new nodes. Each level
        // has its own cursor, so siblings land together in rune order.
        for (size_t depth = mismatch.first - word.begin(); depth < word.size();
             ++depth) {
          if (next[depth] == kJiebaLevelOffsets[depth + 1]) {
            return Status::InternalError("Invalid embedded Jieba trie level");
          }
          nodes_[next[depth]++] = {next[depth + 1], word[depth], 0};
        }
        nodes_[next[word.size() - 1] - 1].entry = row + 1;
        previous = &word;
      }
    }
    if (!std::equal(next.begin(), next.end() - 1,
                    kJiebaLevelOffsets.begin() + 1)) {
      return Status::InternalError("Incomplete embedded Jieba trie");
    }

    // Most lookups start a new word. Index the root's wide fanout directly.
    roots_.resize(kJiebaRootSize);
    for (uint32_t i = 1; i < nodes_[1].first_child; ++i) {
      if (nodes_[i].rune >= roots_.size()) {
        return Status::InternalError("Invalid embedded Jieba trie root");
      }
      roots_[nodes_[i].rune] = i;
    }
    return Status::OK();
  }

  const cppjieba::DictUnit *Find(Iterator begin, Iterator end) const override {
    if (begin == end) {
      return nullptr;
    }
    uint32_t node = root_child(begin->rune);
    for (++begin; node && begin != end; ++begin) {
      node = child(node, begin->rune);
    }
    return entry(node);
  }

  void Find(Iterator begin, Iterator end, std::vector<cppjieba::Dag> &dags,
            size_t max_word_len) const override {
    const size_t size = end - begin;
    dags.resize(size);
    for (size_t i = 0; i < size; ++i) {
      auto &dag = dags[i];
      dag.runestr = begin[i];
      uint32_t node = root_child(begin[i].rune);
      // cppjieba always emits the single-rune edge, including unknown words.
      dag.nexts.push_back(std::make_pair(i, entry(node)));
      for (size_t j = i + 1; node && j < size && j - i < max_word_len; ++j) {
        node = child(node, begin[j].rune);
        if (const auto *word = entry(node)) {
          dag.nexts.push_back(std::make_pair(j, word));
        }
      }
    }
  }

 private:
  struct Node {
    uint32_t first_child;
    uint32_t rune;
    uint32_t entry;
  };

  uint32_t root_child(uint32_t rune) const {
    return rune < roots_.size() ? roots_[rune] : 0;
  }

  uint32_t child(uint32_t node, uint32_t rune) const {
    const auto begin = nodes_.begin() + nodes_[node].first_child;
    const auto end = nodes_.begin() + nodes_[node + 1].first_child;
    const auto it = std::lower_bound(
        begin, end, rune,
        [](const Node &n, uint32_t key) { return n.rune < key; });
    return it != end && it->rune == rune
               ? static_cast<uint32_t>(it - nodes_.begin())
               : 0;
  }

  const cppjieba::DictUnit *entry(uint32_t node) const {
    const uint32_t row = nodes_[node].entry;
    return row ? &entries_[row - 1] : nullptr;
  }

  // DictTrie owns the dictionary entries and the lookup together.
  const std::vector<cppjieba::DictUnit> &entries_;
  std::vector<Node> nodes_;
  std::vector<uint32_t> roots_;
};

Result<EmbeddedJiebaDict> load_compiled_dictionary() {
  auto data = decompress(kJiebaDictLz4, kJiebaDictLz4OriginalSize);
  if (!data) {
    return tl::make_unexpected(data.error());
  }
  // The generator writes columns without padding, independent of the target
  // ABI. DictUnit is materialized once for both dictionary lookup paths.
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
    auto lookup = std::make_unique<EmbeddedTrie>(*(*dict)->node_infos);
    auto status = lookup->build();
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    return cppjieba::DictTrie::CreateReadOnly(**dict, std::move(lookup));
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
