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
#include <cstddef>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>

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

}  // namespace

Result<EmbeddedJiebaDict> get_embedded_jieba_dict() {
  static const Result<EmbeddedJiebaDict> dict =
      []() -> Result<EmbeddedJiebaDict> {
    auto text = decompress(kJiebaDictLz4, kJiebaDictLz4OriginalSize);
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    MemoryInputBuffer buffer(*text);
    std::istream stream(&buffer);
    return std::make_shared<const cppjieba::DictTrie::DictCacheEntry>(
        cppjieba::DictTrie::LoadDict(stream));
  }();
  return dict;
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
