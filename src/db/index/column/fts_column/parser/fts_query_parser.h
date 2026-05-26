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
#include <string>
#include "db/index/column/fts_column/fts_query_ast.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"

namespace zvec::fts {

/*! Default boolean operator applied to adjacent bare terms that are not
 *  separated by an explicit operator (AND / OR / + / -).
 *  This is equivalent to Lucene/Elasticsearch's `default_operator` semantics.
 */
enum class FtsDefaultOperator {
  OR,   // Adjacent bare terms are combined with OR (historical default).
  AND,  // Adjacent bare terms are combined with AND.
};

/*! FTS query parser
 *  Thread-compatible but not thread-safe: create one instance per parse call
 *  or protect with a mutex.
 */
class FtsQueryParser {
 public:
  FtsQueryParser() = default;

  /*! Parse an FTS query expression string into an AST.
   *  \param query        Query string, e.g. '+vector -slow "exact phrase" 中文
   *                      AND 分词'
   *  \param pipeline     Tokenizer pipeline used to tokenize phrase contents
   *                      and bare terms so that query-side segmentation
   *                      matches the doc-side index. Must be non-null.
   *  \param default_op   Default operator for adjacent bare terms with no
   *                      explicit operator. Defaults to OR for backward
   *                      compatibility. Does not change the semantics of
   *                      explicit AND / OR / + / - usages.
   *  \return Root AST node, or nullptr on parse failure. Call err_msg() to
   *          retrieve the error description.
   */
  FtsAstNodePtr parse(const std::string &query,
                      const TokenizerPipelinePtr &pipeline,
                      FtsDefaultOperator default_op = FtsDefaultOperator::OR);

  /*! Return the error message from the most recent failed parse() call. */
  const std::string &err_msg() const {
    return err_msg_;
  }

 private:
  std::string err_msg_;
};

}  // namespace zvec::fts
