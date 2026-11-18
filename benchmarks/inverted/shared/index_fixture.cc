// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "index_fixture.h"
#include <stdexcept>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>

namespace zvec::benchmarks::inverted {

IndexFixture::IndexFixture(DataType data_type, const std::string &path,
                           bool read_only)
    : read_only_(read_only) {
  if (path.empty()) {
    throw std::invalid_argument("Index path must not be empty");
  }
  if (read_only) {
    if (!ailego::File::IsDirectory(path)) {
      throw std::runtime_error("Index directory does not exist: " + path);
    }
  } else {
    if (ailego::File::IsExist(path)) {
      throw std::runtime_error("Refusing to overwrite index path: " + path);
    }
    const auto parent = ailego::FileHelper::PathToUtf8(
        ailego::FileHelper::PathFromUtf8(path).parent_path());
    if (!parent.empty() && !ailego::File::MakePath(parent)) {
      throw std::runtime_error("Cannot create index parent directory: " +
                               parent);
    }
  }
  FieldSchema field{"value", data_type, true,
                    std::make_shared<InvertIndexParams>()};
  index_ = InvertedIndexer::CreateAndOpen("inverted_benchmark", path,
                                          !read_only, {field}, read_only);
  if (!index_) {
    throw std::runtime_error("Failed to open inverted index: " + path);
  }
  column_ = (*index_)["value"];
  if (!column_) {
    throw std::runtime_error("Failed to open benchmark column: " + path);
  }
  if (read_only && !column_->is_sealed()) {
    throw std::runtime_error("Benchmark index must be sealed: " + path);
  }
}

IndexFixture::~IndexFixture() {
  close();
}

InvertedColumnIndexer &IndexFixture::column() {
  if (!column_) {
    throw std::logic_error("Benchmark index is closed");
  }
  return *column_;
}

void IndexFixture::seal() {
  if (!index_ || !column_ || read_only_) {
    throw std::logic_error("Cannot seal a closed or read-only index");
  }
  auto status = index_->seal();
  if (!status.ok() || !column_->is_sealed()) {
    throw std::runtime_error("Failed to seal benchmark index: " +
                             status.message());
  }
}

void IndexFixture::close() {
  column_.reset();
  index_.reset();
}

}  // namespace zvec::benchmarks::inverted
