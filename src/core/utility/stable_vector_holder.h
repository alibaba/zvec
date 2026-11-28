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

#include <memory>

namespace zvec {
namespace core {

// Optional internal capability; does not change the public IndexHolder ABI.
// On success, lifetime is non-null and keeps all data returned by the holder's
// existing iterators alive across reads and iterator/provider destruction.
// Addresses may be read concurrently until lifetime is released. The caller
// must keep the source immutable, including no explicit close, during that
// time. Acquiring a lifetime reads no vectors and does not evaluate filters.
// Errors leave the output unchanged. Only NotImplemented allows copying as a
// fallback; other errors must propagate.
class StableVectorHolder {
 public:
  virtual ~StableVectorHolder() = default;
  virtual int retain_stable_data(std::shared_ptr<const void> *lifetime) = 0;
};

}  // namespace core
}  // namespace zvec
