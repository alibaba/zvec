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
#include "checkpoint_recovery.h"

namespace zvec::checkpoint_test {
namespace {

void Check(const Status &status, const char *operation) {
  if (!status.ok()) {
    std::cerr << operation << ": " << status.message() << std::endl;
    std::_Exit(2);
  }
}

void CheckWrite(const Result<WriteResults> &result, size_t count,
                const char *operation) {
  if (!result) {
    Check(result.error(), operation);
  }
  if (result->size() != count) {
    std::cerr << operation << ": unexpected result count" << std::endl;
    std::_Exit(2);
  }
  for (size_t i = 0; i < result->size(); ++i) {
    if (!(*result)[i].ok()) {
      std::cerr << operation << " document " << i << ": "
                << (*result)[i].message() << std::endl;
      std::_Exit(2);
    }
  }
}

[[noreturn]] void StopAtCheckpoint() {
  // The parent observes SIGSTOP before sending SIGKILL. No collection close,
  // destructor, or extra flush may run after the requested checkpoint.
  std::cerr << "Reached checkpoint" << std::endl;
  std::raise(SIGSTOP);
  std::_Exit(3);
}

}  // namespace
}  // namespace zvec::checkpoint_test

int main(int argc, char **argv) {
  using namespace zvec;
  using namespace zvec::checkpoint_test;
  if (argc != 4) {
    std::cerr << "Usage: checkpoint_recovery_worker PATH MODE FTS" << std::endl;
    return 2;
  }
  const std::string mode = argv[2];
  const bool fts = std::string(argv[3]) == "fts";
  const bool create = mode == "written" || mode == "flushed" ||
                      mode == "sealed" || mode == "optimized" ||
                      mode == "rollover";
  auto result = create ? Collection::CreateAndOpen(argv[1], MakeSchema(fts), {})
                       : Collection::Open(argv[1], {});
  if (!result) {
    Check(result.error(), create ? "create" : "recover");
  }
  auto collection = result.value();
  if (create) {
    const int count = mode == "rollover" ? kSegmentDocs + 1 : kInitialDocs;
    // Separate calls exercise the automatic segment switch at its boundary.
    for (int id = 0; id < count; ++id) {
      std::vector<Doc> docs{MakeDoc(id, 0)};
      CheckWrite(collection->insert(docs), docs.size(), "insert");
    }
    if (mode == "flushed") {
      Check(collection->flush(), "flush");
    } else if (mode == "sealed") {
      // Creating an iterator publishes a sealed segment through the public API.
      // Keep the iterator and collection alive until the parent kills us.
      auto iterator = collection->create_iterator();
      if (!iterator) {
        Check(iterator.error(), "create_iterator");
      }
      StopAtCheckpoint();
    } else if (mode == "optimized") {
      Check(collection->optimize(), "optimize");
    }
  } else if (mode == "mutated" || mode == "mutated_flushed") {
    std::vector<Doc> updates{MakeDoc(0, 1)};
    CheckWrite(collection->update(updates), 1, "update");
    CheckWrite(collection->delete_({PrimaryKey(1), PrimaryKey(2)}), 2,
               "delete");
    std::vector<Doc> upserts{MakeDoc(2, 2), MakeDoc(kInitialDocs, 2)};
    CheckWrite(collection->upsert(upserts), upserts.size(), "upsert");
    if (mode == "mutated_flushed") {
      Check(collection->flush(), "flush mutations");
    }
  } else if (mode == "deleted") {
    std::vector<std::string> keys;
    for (int id = 0; id < kInitialDocs; ++id) {
      keys.push_back(PrimaryKey(id));
    }
    CheckWrite(collection->delete_(keys), keys.size(), "delete all");
    Check(collection->flush(), "flush deletes");
    Check(collection->optimize(), "optimize empty collection");
  } else if (mode != "recovered") {
    std::cerr << "Unknown checkpoint: " << mode << std::endl;
    return 2;
  }
  StopAtCheckpoint();
}
