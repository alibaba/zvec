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
// arrow_ipc_writer.h
#pragma once

#include <memory>
#include <string>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/writer.h>
#include "db/index/common/index_filter.h"
#include "forward_writer.h"

namespace zvec {

class ArrowIpcWriter : public ForwardWriter {
 public:
  explicit ArrowIpcWriter(const std::string &filepath,
                          int64_t max_rows_per_batch = 0);
  ~ArrowIpcWriter() override;

  arrow::Status insert(std::shared_ptr<arrow::RecordBatchReader> reader,
                       const IndexFilter::Ptr &filter = nullptr) override;

  arrow::Status insert_batch(std::shared_ptr<arrow::RecordBatch> batch,
                             const IndexFilter::Ptr &filter = nullptr) override;

  arrow::Status finalize() override;

 private:
  arrow::Status write_batch(const arrow::RecordBatch &batch,
                            const IndexFilter::Ptr &filter);

 private:
  std::string filepath_;
  int64_t max_rows_per_batch_;

  std::shared_ptr<arrow::io::FileOutputStream> sink_;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer_;
  std::shared_ptr<arrow::Schema> schema_;
  bool finalized_;
};

}  // namespace zvec