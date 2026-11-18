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

#include <functional>
#include <string>
#include <string_view>
#include <zvec/db/status.h>

namespace zvec {

// Internal publication boundary, called under the database writer lock.
// Hooks run before each operation and are instance-local fault injection.
// No callback may start another publication to the same destination.
class DurableFile {
 public:
  enum class Stage { kWrite, kFileSync, kRename, kDirectorySync };
  using Hook = std::function<Status(Stage, const std::string &)>;
  struct Outcome {
    Status status;
    // A failure after rename is uncertain: retain both generations and stop
    // dependent writes until recovery confirms the authoritative record.
    bool renamed{false};
  };

  // Writes destination + ".pending", syncs, atomically replaces destination,
  // then syncs its directory. A stale private .pending is safe to overwrite.
  static Outcome Replace(const std::string &destination,
                         std::string_view contents, const Hook &hook = {});

  // Publishes an already completed, closed file in the SAME directory.
  // The caller owns unique naming and cleanup; this never deletes a source.
  static Outcome PublishPrepared(const std::string &prepared,
                                 const std::string &destination,
                                 const Hook &hook = {});

  static Status Sync(const std::string &path);

  // Used to finish a full-storage checkpoint before its manifest commit.
  // Platforms without a verified directory-sync primitive fail closed here.
  static Status SyncDirectory(const std::string &path);
};

}  // namespace zvec
