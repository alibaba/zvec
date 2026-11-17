// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <string>
#include <zvec/db/type.h>
#include "db/index/column/inverted_column/inverted_indexer.h"

namespace zvec::benchmarks::inverted {

class IndexFixture {
 public:
  IndexFixture(DataType data_type, const std::string &path, bool read_only);
  ~IndexFixture();
  IndexFixture(const IndexFixture &) = delete;
  IndexFixture &operator=(const IndexFixture &) = delete;

  InvertedColumnIndexer &column();
  void seal();
  void close();

 private:
  bool read_only_;
  InvertedIndexer::Ptr index_;
  InvertedColumnIndexer::Ptr column_;
};

}  // namespace zvec::benchmarks::inverted
