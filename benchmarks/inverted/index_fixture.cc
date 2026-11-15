// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "index_fixture.h"
#include <filesystem>
#include <stdexcept>

namespace zvec::benchmarks::inverted {

IndexFixture::IndexFixture(const TypeSuite &suite, const std::string &path,
                           bool read_only)
    : read_only_(read_only) {
  namespace fs = std::filesystem;
  if (path.empty()) {
    throw std::invalid_argument("Index path must not be empty");
  }
  if (read_only) {
    if (!fs::is_directory(path)) {
      throw std::runtime_error("Index directory does not exist: " + path);
    }
  } else {
    if (fs::exists(path)) {
      throw std::runtime_error("Refusing to overwrite index path: " + path);
    }
    const auto parent = fs::path(path).parent_path();
    if (!parent.empty()) {
      fs::create_directories(parent);
    }
  }
  FieldSchema field{"value", suite.data_type, true,
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
  Close();
}

InvertedColumnIndexer &IndexFixture::column() {
  if (!column_) {
    throw std::logic_error("Benchmark index is closed");
  }
  return *column_;
}

InvertedSearchResult::Ptr IndexFixture::Search(const QueryInput &query) const {
  if (!column_) {
    throw std::logic_error("Benchmark index is closed");
  }
  switch (query.op) {
    case QueryOp::kEq:
    case QueryOp::kNe:
    case QueryOp::kLt:
    case QueryOp::kGe: {
      if (query.values.size() != 1) {
        throw std::invalid_argument("Scalar query needs exactly one value");
      }
      CompareOp op = CompareOp::EQ;
      if (query.op == QueryOp::kNe) op = CompareOp::NE;
      if (query.op == QueryOp::kLt) op = CompareOp::LT;
      if (query.op == QueryOp::kGe) op = CompareOp::GE;
      return column_->search(query.values.front(), op);
    }
    case QueryOp::kAny:
    case QueryOp::kAll:
    case QueryOp::kNotAny:
    case QueryOp::kNotAll: {
      if (query.values.empty()) {
        throw std::invalid_argument(
            "Membership query needs at least one value");
      }
      CompareOp op = CompareOp::CONTAIN_ANY;
      if (query.op == QueryOp::kAll) op = CompareOp::CONTAIN_ALL;
      if (query.op == QueryOp::kNotAny) op = CompareOp::NOT_CONTAIN_ANY;
      if (query.op == QueryOp::kNotAll) op = CompareOp::NOT_CONTAIN_ALL;
      return column_->multi_search(query.values, op);
    }
    case QueryOp::kIsNull:
      return column_->search_null();
    case QueryOp::kIsNotNull:
      return column_->search_non_null();
    case QueryOp::kLengthEq:
      return column_->search_array_len(query.length, CompareOp::EQ);
  }
  throw std::invalid_argument("Unsupported query operation");
}

void IndexFixture::Seal() {
  if (!index_ || !column_ || read_only_) {
    throw std::logic_error("Cannot seal a closed or read-only index");
  }
  auto status = index_->seal();
  if (!status.ok() || !column_->is_sealed()) {
    throw std::runtime_error("Failed to seal benchmark index: " +
                             status.message());
  }
}

void IndexFixture::Close() {
  column_.reset();
  index_.reset();
}

}  // namespace zvec::benchmarks::inverted
