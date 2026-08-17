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
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <parquet/arrow/writer.h>
#include "db/index/common/index_filter.h"
#include "forward_writer.h"

namespace zvec {

class ParquetWriter : public ForwardWriter {
 public:
  explicit ParquetWriter(const std::string &filepath,
                         int64_t max_rows_per_group = 0);

  ~ParquetWriter() override;

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
  int64_t max_rows_per_group_ = 0;

  std::shared_ptr<arrow::io::FileOutputStream> outfile_;
  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  bool finalized_ = false;
};

}  // namespace zvec