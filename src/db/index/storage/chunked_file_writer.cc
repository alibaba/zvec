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

#include "chunked_file_writer.h"
#include <fstream>
#include <arrow/ipc/writer.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <zvec/ailego/logger/logger.h>


namespace zvec {

class IpcChunkedWriter : public ChunkedFileWriter {
 public:
  static arrow::Result<std::unique_ptr<IpcChunkedWriter>> Make(
      const std::string &path, const std::shared_ptr<arrow::Schema> &schema) {
    ARROW_ASSIGN_OR_RAISE(auto out_file,
                          arrow::io::FileOutputStream::Open(path));

    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    ARROW_ASSIGN_OR_RAISE(writer, arrow::ipc::MakeFileWriter(out_file, schema));

    return std::make_unique<IpcChunkedWriter>(schema, std::move(out_file),
                                              std::move(writer));
  }

  IpcChunkedWriter(std::shared_ptr<arrow::Schema> schema,
                   std::shared_ptr<arrow::io::FileOutputStream> out_file,
                   std::shared_ptr<arrow::ipc::RecordBatchWriter> writer)
      : ChunkedFileWriter(std::move(schema)),
        out_file_(std::move(out_file)),
        writer_(std::move(writer)) {}

  arrow::Status Write(const arrow::RecordBatch &batch) override {
    return writer_->WriteRecordBatch(batch);
  }

  arrow::Status Write(const arrow::Table &table) override {
    return writer_->WriteTable(table);
  }

  arrow::Status Close() override {
    ARROW_RETURN_NOT_OK(writer_->Close());
    return out_file_->Close();
  }

 private:
  std::shared_ptr<arrow::io::FileOutputStream> out_file_;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer_;
};


class ParquetChunkedWriter : public ChunkedFileWriter {
 public:
  static arrow::Result<std::unique_ptr<ParquetChunkedWriter>> Make(
      const std::string &path, const std::shared_ptr<arrow::Schema> &schema) {
    ARROW_ASSIGN_OR_RAISE(auto out_file,
                          arrow::io::FileOutputStream::Open(path));

    parquet::WriterProperties::Builder builder;
    auto properties = builder.build();

    std::shared_ptr<parquet::arrow::FileWriter> writer;
    ARROW_ASSIGN_OR_RAISE(writer, parquet::arrow::FileWriter::Open(
                                      *schema, arrow::default_memory_pool(),
                                      out_file, properties));

    return std::make_unique<ParquetChunkedWriter>(schema, std::move(out_file),
                                                  std::move(writer));
  }

  ParquetChunkedWriter(std::shared_ptr<arrow::Schema> schema,
                       std::shared_ptr<arrow::io::FileOutputStream> out_file,
                       std::shared_ptr<parquet::arrow::FileWriter> writer)
      : ChunkedFileWriter(std::move(schema)),
        out_file_(std::move(out_file)),
        writer_(std::move(writer)) {}

  arrow::Status Write(const arrow::RecordBatch &batch) override {
    return writer_->WriteRecordBatch(batch);
  }

  arrow::Status Write(const arrow::Table &table) override {
    return writer_->WriteTable(table);
  }

  arrow::Status Close() override {
    ARROW_RETURN_NOT_OK(writer_->Close());
    return out_file_->Close();
  }

 private:
  std::shared_ptr<arrow::io::FileOutputStream> out_file_;
  std::shared_ptr<parquet::arrow::FileWriter> writer_;
};


std::unique_ptr<ChunkedFileWriter> ChunkedFileWriter::Open(
    const std::string &file_path, const std::shared_ptr<arrow::Schema> &schema,
    FileFormat format) {
  switch (format) {
    case FileFormat::IPC: {
      auto result = IpcChunkedWriter::Make(file_path, schema);
      if (!result.ok()) {
        LOG_ERROR("Failed to open IPC writer: %s",
                  result.status().ToString().c_str());
        return nullptr;
      }
      return std::move(result).ValueUnsafe();
    }
    case FileFormat::PARQUET: {
      auto result = ParquetChunkedWriter::Make(file_path, schema);
      if (!result.ok()) {
        LOG_ERROR("Failed to open Parquet writer: %s",
                  result.status().ToString().c_str());
        return nullptr;
      }
      return std::move(result).ValueUnsafe();
    }
    default:
      LOG_ERROR("Unsupported format");
      return nullptr;
  }
}

}  // namespace zvec
