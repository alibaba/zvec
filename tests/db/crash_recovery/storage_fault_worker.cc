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

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include "checkpoint_recovery.h"

namespace zvec::checkpoint_test {
namespace {

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void Check(const Status &status) {
  Require(status.ok(), status.message());
}

void CheckWrite(const Result<WriteResults> &result, size_t count) {
  if (!result) {
    throw std::runtime_error(result.error().message());
  }
  Require(result->size() == count, "Incorrect write result count");
  for (const auto &status : result.value()) {
    Check(status);
  }
}

void InsertRange(Collection &collection, int begin, int end) {
  for (int id = begin; id < end; ++id) {
    std::vector<Doc> docs{MakeDoc(id, 0)};
    CheckWrite(collection.insert(docs), 1);
  }
}

void CheckQuery(Collection &collection, SearchQuery query,
                const std::set<std::string> &expected) {
  auto result = collection.query(query);
  if (!result) {
    throw std::runtime_error(result.error().message());
  }
  std::set<std::string> actual;
  for (const auto &doc : result.value()) {
    Require(doc != nullptr, "Null query result");
    Require(actual.insert(doc->pk()).second, "Duplicate query result");
  }
  Require(actual == expected, "Query results differ from recovered documents");
}

std::set<std::string> Verify(Collection &collection, bool fts, int minimum,
                             bool continued) {
  std::vector<std::string> keys;
  for (int id = 0; id <= 2 * kInitialDocs; ++id) {
    keys.push_back(PrimaryKey(id));
  }
  auto docs = collection.fetch(keys);
  if (!docs) {
    throw std::runtime_error(docs.error().message());
  }
  std::set<std::string> present;
  for (int id = 0; id <= 2 * kInitialDocs; ++id) {
    const auto it = docs->find(PrimaryKey(id));
    Require(it != docs->end(), "Missing fetch result for " + PrimaryKey(id));
    if (id < minimum || (continued && id == 2 * kInitialDocs)) {
      Require(it->second != nullptr, "Lost durable document " + PrimaryKey(id));
    }
    if (it->second) {
      const int generation = id == 2 * kInitialDocs ? 3 : 0;
      Require(*it->second == MakeDoc(id, generation),
              "Corrupt document " + PrimaryKey(id));
      present.insert(PrimaryKey(id));
    }
  }
  auto stats = collection.stats();
  if (!stats) {
    throw std::runtime_error(stats.error().message());
  }
  Require(stats->doc_count == present.size(), "Incorrect document count");
  SearchQuery query;
  query.topk_ = 2 * kInitialDocs + 2;
  query.target_.field_name_ = "vec";
  const std::vector<float> vector{0, 0, 1, 0};
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  CheckQuery(collection, query, present);
  auto original = present;
  original.erase(PrimaryKey(2 * kInitialDocs));
  query.filter_ = "generation = 0";
  CheckQuery(collection, query, original);
  if (fts) {
    for (const auto *field : {"text", "title"}) {
      SearchQuery text_query;
      text_query.topk_ = query.topk_;
      text_query.target_.field_name_ = field;
      FtsClause clause;
      clause.query_string_ = "version0";
      text_query.target_.clause_ = clause;
      CheckQuery(collection, text_query, original);
      clause.query_string_ = "shared";
      text_query.target_.clause_ = clause;
      CheckQuery(collection, text_query, present);
    }
  }
  return present;
}

// The caller exits with _Exit even on error. Destructors must not flush data
// after a failed operation and accidentally repair the state under test.
void Run(const std::string &path, const std::string &mode, bool fts,
         int minimum) {
  auto opened = mode == "prepare"
                    ? Collection::CreateAndOpen(path, MakeSchema(fts), {})
                    : Collection::Open(path, {});
  if (!opened) {
    throw std::runtime_error(opened.error().message());
  }
  // Keep ownership outside stack unwinding on failures.
  static Collection::Ptr collection;
  collection = std::move(opened.value());
  auto &col = *collection;
  if (mode == "prepare") {
    InsertRange(col, 0, kInitialDocs);
    Check(col.flush());
    Check(col.close());
  } else if (mode == "verify") {
    auto expected = Verify(col, fts, minimum, false);
    const int id = 2 * kInitialDocs;
    std::vector<Doc> docs{MakeDoc(id, 1)};
    CheckWrite(col.insert(docs), 1);
    docs = {MakeDoc(id, 2)};
    CheckWrite(col.update(docs), 1);
    CheckWrite(col.delete_({PrimaryKey(id)}), 1);
    auto deleted = col.fetch({PrimaryKey(id)});
    Require(deleted.has_value(), "Fetch after delete failed");
    Require(deleted->at(PrimaryKey(id)) == nullptr,
            "Deleted document survived");
    docs = {MakeDoc(id, 3)};
    CheckWrite(col.upsert(docs), 1);
    Check(col.flush());
    Check(col.optimize());
    expected.insert(PrimaryKey(id));
    Require(Verify(col, fts, minimum, true) == expected,
            "Recovered documents changed after optimize");
    Check(col.close());
    auto reopened = Collection::Open(path, {});
    if (!reopened) {
      throw std::runtime_error(reopened.error().message());
    }
    collection = std::move(reopened.value());
    Require(Verify(*collection, fts, minimum, true) == expected,
            "Recovered documents changed after reopen");
    Check(collection->close());
  } else {
    Require(mode == "insert" || mode == "flush" || mode == "optimize",
            "Unknown operation");
    if (mode != "insert") {
      InsertRange(col, kInitialDocs, 2 * kInitialDocs);
    }
    std::cout << "READY" << std::endl;
    std::raise(SIGSTOP);
    if (mode == "insert") {
      InsertRange(col, kInitialDocs, 2 * kInitialDocs);
    } else if (mode == "optimize") {
      Check(col.optimize());
    }
    Check(col.flush());
    std::cout << "DURABLE " << 2 * kInitialDocs << std::endl;
    // No close after the durability acknowledgement.
  }
}

}  // namespace
}  // namespace zvec::checkpoint_test

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: storage_fault_worker PATH MODE plain|fts MINIMUM"
              << std::endl;
    std::_Exit(2);
  }
  try {
    zvec::checkpoint_test::Run(argv[1], argv[2], std::string(argv[3]) == "fts",
                               std::stoi(argv[4]));
    std::_Exit(0);
  } catch (const std::exception &error) {
    std::cerr << "STORAGE_ERROR: " << error.what() << std::endl;
    std::_Exit(2);
  }
}
