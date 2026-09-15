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

#include <memory>
#include <string>
#include <arrow/api.h>
#include <arrow/record_batch.h>
#include "db/common/constants.h"

namespace zvec::sqlengine {

static const constexpr char *kFieldVector = "_zvec_vector";
static const constexpr char *kFieldSparseIndices = "_zvec_sindices";
static const constexpr char *kFieldSparseValues = "_zvec_svalues";
static const constexpr char *kFieldIsValid = "_zvec_is_valid";

static const inline std::string kCheckNotFiltered = "check_not_filtered";
static const inline std::string kFetchVector = "fetch_vector";
static const inline std::string kFetchSparseVector = "fetch_sparse_vector";
static const inline std::string kContainAll = "contain_all";
static const inline std::string kContainAny = "contain_any";

static const inline std::string kFuncArrayLength = "array_length";

#define enum_to_string(x) #x

class Util {
 public:
  static std::string trim_one_both_side(const std::string &str,
                                        unsigned char c);
  static void string_replace(const std::string &strsrc,
                             const std::string &strdst, std::string *str);
  static std::string normalize(const std::string &sql);

  static std::shared_ptr<arrow::Schema> append_field(
      const arrow::Schema &schema, const std::string &name,
      std::shared_ptr<arrow::DataType> type);

  static std::shared_ptr<arrow::DataType> sparse_type();
};

}  // namespace zvec::sqlengine
