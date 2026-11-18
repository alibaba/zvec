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

#include "fts_file_lifetime.h"
#include <filesystem>
#include <zvec/ailego/utility/file_helper.h>

namespace zvec::fts {

FtsFileOwner::FtsFileOwner(std::shared_ptr<FtsRetirementQueue> queue,
                           std::string path, bool directory, bool candidate)
    : queue_(std::move(queue)),
      entry_{{std::move(path), directory}},
      remove_(candidate) {}

FtsFileOwner::~FtsFileOwner() {
  queue_->release(this);
  if (remove_.load(std::memory_order_acquire)) {
    queue_->retry();
  }
}

FtsRetirementQueue::Pointer FtsRetirementQueue::Create(Hook before_remove) {
  return Pointer(new FtsRetirementQueue(std::move(before_remove)));
}

FtsRetirementQueue::Pointer FtsRetirementQueue::ForDirectory(
    const std::string &path) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::weak_ptr<FtsRetirementQueue>>
      queues;
  std::error_code error;
  auto absolute = std::filesystem::weakly_canonical(
      ailego::FileHelper::PathFromUtf8(path), error);
  const auto key = error ? path : absolute.u8string();
  std::lock_guard lock(mutex);
  auto existing = queues.find(key);
  if (existing != queues.end()) {
    if (auto queue = existing->second.lock()) {
      return queue;
    }
  }
  for (auto it = queues.begin(); it != queues.end();) {
    if (it->second.expired()) {
      it = queues.erase(it);
    } else {
      ++it;
    }
  }
  auto queue = Create();
  queues[key] = queue;
  return queue;
}

FtsFileOwner::Pointer FtsRetirementQueue::track(std::string path,
                                                bool directory,
                                                bool candidate) {
  std::error_code error;
  auto absolute = std::filesystem::weakly_canonical(
      ailego::FileHelper::PathFromUtf8(path), error);
  path =
      error
          ? ailego::FileHelper::PathFromUtf8(path).lexically_normal().u8string()
          : absolute.u8string();
  std::lock_guard lock(mutex_);
  auto existing = owners_.find(path);
  if (existing != owners_.end()) {
    if (auto owner = existing->second.owner.lock()) {
      return owner;
    }
  }
  auto owner = FtsFileOwner::Pointer(
      new FtsFileOwner(shared_from_this(), path, directory, candidate));
  auto parent_path = ailego::FileHelper::PathFromUtf8(path).parent_path();
  while (!parent_path.empty()) {
    auto parent = owners_.find(parent_path.u8string());
    if (parent != owners_.end()) {
      auto pin = parent->second.owner.lock();
      if (pin && pin->entry_.front().directory) {
        owner->parent_ = std::move(pin);
        break;
      }
    }
    const auto ancestor = parent_path.parent_path();
    if (ancestor == parent_path) {
      break;
    }
    parent_path = ancestor;
  }
  owners_[path] = {owner, owner.get()};
  return owner;
}

void FtsRetirementQueue::release(FtsFileOwner *owner) {
  std::lock_guard lock(mutex_);
  auto tracked = owners_.find(owner->entry_.front().path);
  if (tracked != owners_.end() && tracked->second.identity == owner) {
    owners_.erase(tracked);
  }
  if (owner->remove_.load(std::memory_order_acquire)) {
    pending_.splice(pending_.end(), owner->entry_);
  }
}

void FtsRetirementQueue::enqueue(std::list<FtsFileOwner::Entry> *entry) {
  std::lock_guard lock(mutex_);
  pending_.splice(pending_.end(), *entry);
}

size_t FtsRetirementQueue::pending() const {
  std::lock_guard lock(mutex_);
  return pending_.size();
}

size_t FtsRetirementQueue::retry() {
  std::list<FtsFileOwner::Entry> batch;
  {
    std::lock_guard lock(mutex_);
    batch.splice(batch.end(), pending_);
  }
  size_t removed = 0;
  for (auto it = batch.begin(); it != batch.end();) {
    std::error_code error;
    if (hook_ && !hook_(it->path).ok()) {
      ++it;
      continue;
    }
    const auto path = ailego::FileHelper::PathFromUtf8(it->path);
    if (it->directory) {
      std::filesystem::remove_all(path, error);
    } else {
      std::filesystem::remove(path, error);
    }
    if (error) {
      ++it;
    } else {
      it = batch.erase(it);
      ++removed;
    }
  }
  enqueue(&batch);
  return removed;
}

}  // namespace zvec::fts
