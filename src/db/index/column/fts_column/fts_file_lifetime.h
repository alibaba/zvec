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

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <zvec/db/status.h>

namespace zvec::fts {

class FtsRetirementQueue;

// Pin this owner inside the storage/context, after arranging destruction so
// native file handles close BEFORE its last reference is released.
class FtsFileOwner {
 public:
  using Pointer = std::shared_ptr<FtsFileOwner>;
  ~FtsFileOwner();
  FtsFileOwner(const FtsFileOwner &) = delete;
  FtsFileOwner &operator=(const FtsFileOwner &) = delete;

  // Preserve both committed files and candidates whose commit is uncertain.
  void preserve() noexcept {
    remove_.store(false, std::memory_order_release);
  }
  // Only call after a new authoritative version no longer references this
  // generation. Actual removal waits for every reader/range/snapshot pin.
  void retire() noexcept {
    remove_.store(true, std::memory_order_release);
  }

 private:
  friend class FtsRetirementQueue;
  struct Entry {
    std::string path;
    bool directory;
  };
  FtsFileOwner(std::shared_ptr<FtsRetirementQueue> queue, std::string path,
               bool directory, bool candidate);
  std::shared_ptr<FtsRetirementQueue> queue_;
  // A child file/context also pins its registered enclosing directory. This
  // prevents whole-segment retirement from bypassing query-owned files.
  Pointer parent_;
  // Preallocate the queue node before commit; retirement only splices it.
  std::list<Entry> entry_;
  std::atomic<bool> remove_;
};

class FtsRetirementQueue
    : public std::enable_shared_from_this<FtsRetirementQueue> {
 public:
  using Pointer = std::shared_ptr<FtsRetirementQueue>;
  using Hook = std::function<Status(const std::string &)>;
  static Pointer Create(Hook before_remove = {});
  // Reopens in one process share pins, including readers that outlive the
  // collection handle. Cross-process exclusion remains the collection lock.
  static Pointer ForDirectory(const std::string &path);

  // Caller supplies a path belonging solely to this generation. Candidates
  // are removed on last release unless preserve() is called after commit (or
  // an uncertain commit). Existing committed files start preserved.
  // Reopens of a live generation share the same owner and retirement state.
  // Caller must serialize opening against retirement and never reopen a path
  // after its last retired owner has been released; paths are never reused.
  // Register an enclosing directory before tracking any of its children.
  FtsFileOwner::Pointer track(std::string path, bool directory, bool candidate);
  size_t retry();
  size_t pending() const;

 private:
  friend class FtsFileOwner;
  explicit FtsRetirementQueue(Hook hook) : hook_(std::move(hook)) {}
  void enqueue(std::list<FtsFileOwner::Entry> *entry);
  void release(FtsFileOwner *owner);
  struct Tracked {
    std::weak_ptr<FtsFileOwner> owner;
    FtsFileOwner *identity;
  };
  Hook hook_;
  mutable std::mutex mutex_;
  std::list<FtsFileOwner::Entry> pending_;
  std::unordered_map<std::string, Tracked> owners_;
};

}  // namespace zvec::fts
