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

#include <algorithm>
#include <utility>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {
namespace ailego {

bool StringHelper::StartsWith(const std::string &ref,
                              const std::string &prefix) {
  return (ref.size() >= prefix.size()) &&
         (ref.compare(0, prefix.size(), prefix) == 0);
}

bool StringHelper::EndsWith(const std::string &ref, const std::string &suffix) {
  size_t s1 = ref.size();
  size_t s2 = suffix.size();
  return (s1 >= s2) && (ref.compare(s1 - s2, s2, suffix) == 0);
}

void StringHelper::LeftTrim(std::string &str) {
  str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) {
              return !std::isspace(ch);
            }));
}

void StringHelper::RightTrim(std::string &str) {
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](int ch) { return !std::isspace(ch); })
                .base(),
            str.end());
}

void StringHelper::Trim(std::string &str) {
  StringHelper::RightTrim(str);
  StringHelper::LeftTrim(str);
}

std::string StringHelper::CopyLeftTrim(std::string str) {
  StringHelper::LeftTrim(str);
  return str;
}

std::string StringHelper::CopyRightTrim(std::string str) {
  StringHelper::RightTrim(str);
  return str;
}

std::string StringHelper::CopyTrim(std::string str) {
  StringHelper::Trim(str);
  return str;
}

#if defined(_MSC_VER)
#define strncasecmp _strnicmp
#endif

bool StringHelper::CompareIgnoreCase(const std::string &a,
                                     const std::string &b) {
  if (a.size() != b.size()) {
    return false;
  }
  return (strncasecmp(a.data(), b.data(), a.size()) == 0);
}

void StringHelper::Append(std::string *str, const internal::Alphameric &a) {
  str->reserve(str->size() + a.size());
  str->append(a.data(), a.size());
}

void StringHelper::Append(std::string *str, const internal::Alphameric &a,
                          const internal::Alphameric &b) {
  str->reserve(str->size() + a.size() + b.size());
  str->append(a.data(), a.size());
  str->append(b.data(), b.size());
}

void StringHelper::Append(std::string *str, const internal::Alphameric &a,
                          const internal::Alphameric &b,
                          const internal::Alphameric &c) {
  str->reserve(str->size() + a.size() + b.size() + c.size());
  str->append(a.data(), a.size());
  str->append(b.data(), b.size());
  str->append(c.data(), c.size());
}

void StringHelper::Append(std::string *str, const internal::Alphameric &a,
                          const internal::Alphameric &b,
                          const internal::Alphameric &c,
                          const internal::Alphameric &d) {
  str->reserve(str->size() + a.size() + b.size() + c.size() + d.size());
  str->append(a.data(), a.size());
  str->append(b.data(), b.size());
  str->append(c.data(), c.size());
  str->append(d.data(), d.size());
}

void StringHelper::AppendViews(std::string *str,
                               std::initializer_list<StringView> views) {
  size_t new_size = str->size();
  for (auto &v : views) {
    new_size += v.size();
  }
  str->reserve(new_size);
  for (auto &v : views) {
    str->append(v.data(), v.size());
  }
}

}  // namespace ailego
}  // namespace zvec