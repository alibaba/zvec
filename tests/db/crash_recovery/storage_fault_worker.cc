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
#include "db/index/common/version_manager.h"
#include "recovery_verifier.h"

namespace zvec::checkpoint_test {
namespace {

void InsertRange(Collection &collection, int begin, int end) {
  for (int id = begin; id < end; ++id) {
    std::vector<Doc> docs{MakeDoc(id, 0)};
    CheckWrite(collection.insert(docs), 1);
  }
}

ExpectedDocs BaselineDocs() {
  auto expected = InitialDocs();
  for (int offset : {0, kInitialDocs / 2}) {
    expected[offset] = 1;
    expected.erase(offset + 1);
    expected[offset + 2] = 2;
  }
  return expected;
}

std::set<SegmentID> PersistedSegments(const std::string &path) {
  auto manager = VersionManager::Recovery(path);
  if (!manager) throw std::runtime_error(manager.error().message());
  std::set<SegmentID> ids;
  for (auto &meta :
       (*manager)->get_current_version().persisted_segment_metas()) {
    ids.insert(meta->id());
  }
  return ids;
}

ExpectedDocs RecoveredDocs(Collection &collection, bool fts, int minimum) {
  auto expected = BaselineDocs();
  std::vector<std::string> keys;
  for (int id = kInitialDocs; id < 2 * kInitialDocs; ++id)
    keys.push_back(PrimaryKey(id));
  auto docs = collection.fetch(keys);
  if (!docs) throw std::runtime_error(docs.error().message());
  for (int id = kInitialDocs; id < 2 * kInitialDocs; ++id) {
    auto doc = docs->at(PrimaryKey(id));
    Require(id >= minimum || doc != nullptr, "Lost acknowledged document");
    if (doc) expected[id] = 0;
  }
  VerifyState(collection, fts, expected, 2 * kInitialDocs);
  return expected;
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
    // Seal two small segments explicitly; a single segment only builds indexes.
    for (int begin : {0, kInitialDocs / 2}) {
      InsertRange(col, begin, begin + kInitialDocs / 2);
      auto iterator = col.create_iterator();
      if (!iterator) throw std::runtime_error(iterator.error().message());
    }
    for (int offset : {0, kInitialDocs / 2}) {
      std::vector<Doc> updated{MakeDoc(offset, 1)};
      CheckWrite(col.update(updated), 1);
      CheckWrite(col.delete_({PrimaryKey(offset + 1)}), 1);
      std::vector<Doc> upserted{MakeDoc(offset + 2, 2)};
      CheckWrite(col.upsert(upserted), 1);
    }
    Check(col.flush());
    Require(PersistedSegments(path).size() >= 2,
            "Missing compaction input segments");
    VerifyState(col, fts, BaselineDocs(), 2 * kInitialDocs);
    Check(col.close());
  } else if (mode == "verify") {
    auto expected = RecoveredDocs(col, fts, minimum);
    const int id = 2 * kInitialDocs;
    std::vector<Doc> docs{MakeDoc(id, 1)};
    CheckWrite(col.insert(docs), 1);
    expected[id] = 1;
    VerifyState(col, fts, expected, id);
    docs = {MakeDoc(id, 2)};
    CheckWrite(col.update(docs), 1);
    expected[id] = 2;
    VerifyState(col, fts, expected, id);
    CheckWrite(col.delete_({PrimaryKey(id)}), 1);
    expected.erase(id);
    VerifyState(col, fts, expected, id);
    docs = {MakeDoc(id, 3)};
    CheckWrite(col.upsert(docs), 1);
    expected[id] = 3;
    VerifyState(col, fts, expected, id);
    // Keep a recovered row updated across optimize and reopen, without
    // upserting it.
    docs = {MakeDoc(0, 3)};
    CheckWrite(col.update(docs), 1);
    expected[0] = 3;
    VerifyState(col, fts, expected, id);
    Check(col.flush());
    Check(col.optimize());
    VerifyState(col, fts, expected, id);
    Check(col.close());
    auto reopened = Collection::Open(path, {});
    if (!reopened) throw std::runtime_error(reopened.error().message());
    collection = std::move(reopened.value());
    VerifyState(*collection, fts, expected, id);
    Check(collection->close());
  } else {
    Require(mode == "insert" || mode == "flush" || mode == "optimize",
            "Unknown operation");
    if (mode != "insert") {
      InsertRange(col, kInitialDocs, 2 * kInitialDocs);
    }
    if (mode == "optimize") {
      // Finish sealing before arming faults so they exercise compaction itself.
      {
        auto iterator = col.create_iterator();
        if (!iterator) throw std::runtime_error(iterator.error().message());
      }
      Check(col.flush());
    }
    const auto input_segments = PersistedSegments(path);
    if (mode == "optimize") {
      Require(input_segments.size() >= 2,
              "Optimize requires multiple input segments");
      std::cout << "COMPACTION_INPUTS " << input_segments.size() << std::endl;
    }
    std::cout << "READY" << std::endl;
    std::raise(SIGSTOP);
    if (mode == "insert") {
      InsertRange(col, kInitialDocs, 2 * kInitialDocs);
    } else if (mode == "optimize") {
      Check(col.optimize());
      const auto output_segments = PersistedSegments(path);
      Require(output_segments.size() == 1 &&
                  !input_segments.count(*output_segments.begin()),
              "Optimize did not replace input segments with a merged segment");
      std::cout << "COMPACTION_COMPLETE" << std::endl;
    }
    Check(col.flush());
    std::cout << "DURABLE" << std::endl;
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
