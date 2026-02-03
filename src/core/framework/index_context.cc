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

#include <atomic>
#include <random>
#include <zvec/core/framework/index_context.h>

namespace zvec {
namespace core {

uint32_t IndexContext::GenerateMagic(void) {
  static std::atomic_uint32_t magic_number{std::random_device()()};
  return magic_number.fetch_add(1);
}

}  // namespace core
}  // namespace zvec
