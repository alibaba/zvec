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
#include <parquet/arrow/writer.h>
#include <zvec/db/type.h>

namespace zvec {

class ChunkedFileWriter {
 public:
  using Ptr = std::unique_ptr<ChunkedFileWriter>;

  static std::unique_ptr<ChunkedFileWriter> Open(
      const std::string &file_path,
      const std::shared_ptr<arrow::Schema> &schema, FileFormat format);

  virtual arrow::Status Write(const arrow::RecordBatch &batch) = 0;

  virtual arrow::Status Write(const arrow::Table &table) = 0;

  virtual arrow::Status Close() = 0;

  virtual ~ChunkedFileWriter() = default;  // LCOV_EXCL_BR_LINE

 protected:
  explicit ChunkedFileWriter(std::shared_ptr<arrow::Schema> schema)
      : schema_(std::move(schema)) {}

  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace zvec
