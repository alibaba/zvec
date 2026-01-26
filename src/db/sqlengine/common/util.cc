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

#include "util.h"
#include <sys/time.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <ailego/encoding/json.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec::sqlengine {

// std::string
// Util::trim(const std::string str, char c) {
//    std::string tmp = str.substr(str.find_first_not_of(c));
//  return tmp.substr(0, tmp.find_last_not_of(c) + 1);
//}

// above implementation will trim more than one c at front or rear. This is not
// expected.
// below implementation remove both side only one matched char, strictly.
// str is supposed to match on both side at same time. remove both side one byte
// each as c = 0
std::string Util::trim_one_both_side(const std::string &str, unsigned char c) {
  int len = str.length();
  if (len < 2) {
    return str;
  }

  if (str.at(0) == c && str.at(len - 1) == c) {
    return str.substr(1, len - 2);
  }

  return str;
}

void Util::string_replace(const std::string &src, const std::string &dst,
                          std::string *str) {
  std::string::size_type pos = 0;
  std::string::size_type srclen = src.size();
  std::string::size_type dstlen = dst.size();

  while ((pos = str->find(src, pos)) != std::string::npos) {
    str->replace(pos, srclen, dst);
    pos += dstlen;
  }

  return;
}

// normalize sql for parse result after parse
std::string Util::normalize(const std::string &sql) {
  std::string new_sql = sql;
  // rule 1. replace \" with "
  Util::string_replace("\\\"", "\"", &new_sql);
  // rule 2. replace \' with ''
  Util::string_replace("\\\'", "\'", &new_sql);

  return new_sql;
}

std::shared_ptr<arrow::Schema> Util::append_field(
    const arrow::Schema &schema, const std::string &name,
    std::shared_ptr<arrow::DataType> type) {
  auto res = schema.AddField(schema.num_fields(), arrow::field(name, type));
  return res.MoveValueUnsafe();
}

std::shared_ptr<arrow::DataType> Util::sparse_type() {
  return arrow::struct_(arrow::FieldVector{
      arrow::field("index", arrow::binary()),
      arrow::field("value", arrow::binary()),
  });
}

}  // namespace zvec::sqlengine
