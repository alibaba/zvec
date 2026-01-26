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

#include "zvec_parser.h"
#include <float.h>
#include <stdint.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <typeinfo>
#include <ailego/encoding/json.h>
#include <ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include "db/sqlengine/common/util.h"
#include "tree/ParseTree.h"
#include "zvec_cached_sql_parser.h"
#include "zvec_sql_parser.h"

using namespace antlr4;
using namespace tree;

namespace zvec::sqlengine {

const std::string &ZVecParser::err_msg() {
  return err_msg_;
}

std::string ZVecParser::trim(std::string &value) {
  value = Util::trim_one_both_side(value, '\'');
  value = Util::trim_one_both_side(value, '\"');
  // Util::string_replace(value, "\\", "");
  return value;
}

const std::string &ZVecParser::formatted_tree() {
  return formatted_tree_;
}

std::string ZVecParser::to_formatted_string_tree(void *tree, void *parser) {
  if (tree == NULL || parser == NULL) {
    return "";
  }

  ParseTree *parse_tree = reinterpret_cast<ParseTree *>(tree);
  ZVecParser *se_parser = reinterpret_cast<ZVecParser *>(parser);

  std::string tree_text = parse_tree->toStringTree(se_parser);

  int pos = 0, pos1 = 0, pos2 = 0, start = 0;
  int i = 0, num = 0;
  const std::string DELIMITER = "  ";
  const std::string LINE = "\n";
  int lastPos1 = 0;

  std::string out;

  while (true) {
    std::string formatted = "";

    pos1 = (int)tree_text.find_first_of('(', start);
    pos2 = (int)tree_text.find_first_of(')', start);

    if (pos1 == 0) {
      start = pos + 1;
      continue;
    }

    if (pos1 < 0 && pos2 < 0) {
      break;
    }

    if (pos1 >= 0 && pos1 < pos2) {
      if (lastPos1 == 1) {
        formatted += "(";
      }
      pos = pos1;
      formatted += tree_text.substr(start, (size_t)pos1 - start);
      num++;
    } else {
      if (lastPos1 == 1) {
        formatted += "(";
      }
      pos = pos2;
      formatted += tree_text.substr(start, (size_t)pos2 - start) + ")";
      num--;
    }

    formatted += LINE;
    for (i = 0; i < num; i++) {
      formatted += DELIMITER;
    }

    start = pos + 1;

    if (pos == pos1) {
      lastPos1 = 1;
    } else {
      lastPos1 = 0;
    }

    out += formatted;
  }

  return out;
}

void ZVecParser::save_to_file(const std::string &file_name,
                              const std::string &formatted) {
  std::ofstream outfile(file_name);
  outfile << formatted;
  outfile << std::endl;
  outfile.close();
}

ZVecParser::Ptr ZVecParser::create() {
  // TODO: support config
  // auto &config = zvec::Config::Instance();
  // int32_t cache_count = config.get_sql_info_cache_count();
  return create(100);
}

ZVecParser::Ptr ZVecParser::create(int cache_count) {
  // if not config, or if config between 0 and 100, upround to 100
  if (cache_count >= 0 && cache_count < DEFAULT_CACHE_COUNT) {
    cache_count = DEFAULT_CACHE_COUNT;
  }

  if (cache_count > 0) {
    LOG_DEBUG("ZVecCachedSQLParser enabled. effective cache_count %d",
              cache_count);
    return std::make_shared<ZVecCachedSQLParser>(cache_count);
  } else {
    LOG_DEBUG("ZVecCachedSQLParser disabled.");
    return std::make_shared<ZVecSQLParser>();
  }
}

Node::Ptr ZVecParser::parse_vector_text(std::string *vector_text) {
  return std::make_shared<VectorMatrixNode>(std::move(*vector_text), "", "",
                                            nullptr);
}

}  // namespace zvec::sqlengine
