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

#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace core {

const static std::string TAGLIST_HEADER_SEGMENT_NAME("local_taglists_header");
const static std::string TAGLIST_KEY_SEGMENT_NAME("local_taglists_key");
const static std::string TAGLIST_DATA_SEGMENT_NAME("local_taglists_data");

#pragma pack(4)
struct TagListHeader {
  uint64_t num_vecs;
  uint8_t meta_buf[252];
};
#pragma pack()

}  // namespace core
}  // namespace zvec