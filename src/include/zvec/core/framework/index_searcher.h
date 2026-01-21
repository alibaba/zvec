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

#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_runner.h>
#include <zvec/core/framework/index_stats.h>

namespace zvec {
namespace core {

/*! Index Searcher
 */
class IndexSearcher : public IndexRunner {
 public:
  //! Index Searcher Pointer
  typedef std::shared_ptr<IndexSearcher> Pointer;

  //! Constructor
  IndexSearcher() = default;

  //! Destructor
  virtual ~IndexSearcher() = default;

  //! Initialize Searcher
  virtual int init(const ailego::Params & /*params*/) = 0;

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const = 0;

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const = 0;

  virtual int load(IndexStorage::Pointer /*container*/,
                   IndexMetric::Pointer /*metric*/) {
    return IndexError_NotImplemented;
  }
};

}  // namespace core
}  // namespace zvec
