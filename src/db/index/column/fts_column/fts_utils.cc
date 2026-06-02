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

#include "fts_utils.h"
#include <zvec/ailego/logger/logger.h>

namespace zvec::fts {

bool parse_doc_term_key(const std::string &key, std::string *term_out,
                        uint32_t *doc_id_out) {
  // Key format: term + '\0' + doc_id(4B big-endian)
  // Minimum length: 1 byte term + 1 byte '\0' + 4 bytes doc_id = 6 bytes.
  if (key.size() < 6) {
    LOG_WARN("parse_doc_term_key: key too short. size[%zu]", key.size());
    return false;
  }
  const size_t separator_pos = key.size() - sizeof(uint32_t) - 1;
  if (key[separator_pos] != '\0') {
    LOG_WARN("parse_doc_term_key: missing separator. size[%zu]", key.size());
    return false;
  }
  *term_out = key.substr(0, separator_pos);
  *doc_id_out = decode_uint32_big_endian(key.data() + separator_pos + 1);
  return true;
}

}  // namespace zvec::fts
