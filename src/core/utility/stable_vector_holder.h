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

#include <cstdint>
#include <memory>
#include <vector>

namespace zvec {
namespace core {

// Vectors and keys are in holder iteration order. Owners keep the backing
// allocations/mappings alive independently of the holder and its iterators.
// The caller must keep the source immutable (including no explicit close)
// while using the view. All vector addresses remain valid across reads and
// may be read concurrently until the view is released.
struct StableVectorView {
  std::vector<uint64_t> keys{};
  std::vector<const void *> vectors{};
  std::vector<std::shared_ptr<const void>> owners{};
};

// Optional internal capability; does not change the public IndexHolder ABI.
// Unlike OrdinalAccessHolder, this never exposes reusable scratch or evictable
// pages. Errors leave the output unchanged. Only NotImplemented permits the
// caller to fall back to copying; read errors must be propagated.
class StableVectorHolder {
 public:
  virtual ~StableVectorHolder() = default;
  virtual int get_stable_vectors(StableVectorView *view) = 0;
};

}  // namespace core
}  // namespace zvec
