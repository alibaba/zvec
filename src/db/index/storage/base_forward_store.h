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
#include <vector>
#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/table_builder.h>
#include <arrow/util/async_generator.h>
#include <zvec/db/status.h>

namespace cp = arrow::compute;

using Table = arrow::Table;
using RecordBatch = arrow::RecordBatch;
using RecordBatchReader = arrow::RecordBatchReader;
using RecordBatchBuilder = arrow::RecordBatchBuilder;
using TablePtr = std::shared_ptr<Table>;
using ExecBatchPtr = std::shared_ptr<arrow::compute::ExecBatch>;
using RecordBatchPtr = std::shared_ptr<RecordBatch>;
using RecordBatchReaderPtr = std::shared_ptr<RecordBatchReader>;
using RecordBatchBuilderPtr = std::shared_ptr<RecordBatchBuilder>;

namespace zvec {

class BaseForwardStore {
 public:
  using Ptr = std::shared_ptr<BaseForwardStore>;

  virtual Status Open() = 0;

  virtual TablePtr fetch(const std::vector<std::string> &columns,
                         const std::vector<int> &indices) = 0;

  virtual ExecBatchPtr fetch(const std::vector<std::string> &columns,
                             int index) = 0;

  virtual RecordBatchReaderPtr scan(
      const std::vector<std::string> &columns) = 0;

  virtual const std::shared_ptr<arrow::Schema> physic_schema() const = 0;

  virtual TablePtr get_table() = 0;
};

}  // namespace zvec