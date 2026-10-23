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

#include <memory>
#include <cppjieba/DictTrie.hpp>
#include <cppjieba/HMMModel.hpp>
#include <zvec/db/status.h>

namespace zvec::fts {

using EmbeddedJiebaDict =
    std::shared_ptr<const cppjieba::DictTrie::DictCacheEntry>;
using EmbeddedJiebaHmmModel = std::shared_ptr<const cppjieba::HMMModel>;

// Lazily decompress and parse each default resource once. The returned data
// is immutable and shared; the decompressed text is released after parsing.
Result<EmbeddedJiebaDict> get_embedded_jieba_dict();
Result<EmbeddedJiebaHmmModel> get_embedded_jieba_hmm_model();

}  // namespace zvec::fts
