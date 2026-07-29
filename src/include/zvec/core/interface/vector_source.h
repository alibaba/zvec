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
#include <zvec/export.h>

namespace zvec {
namespace core {

class ZVEC_CORE_API VectorSource {
 public:
  VectorSource();
  virtual ~VectorSource();

  virtual const void *get_vector(uint32_t node_id) const = 0;

  virtual void get_vectors(const uint32_t *ids, uint32_t count,
                           const void **out) const;
};

}  // namespace core
}  // namespace zvec
