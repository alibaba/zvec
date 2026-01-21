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

#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Memory Helper
 */
struct MemoryHelper {
  //! Retrieve the page size of memory
  static size_t PageSize(void);

  //! Retrieve the huge page size of memory
  static size_t HugePageSize(void);

  //! Retrieve the VSZ and RSS of self process in bytes
  static bool SelfUsage(size_t *vsz, size_t *rss);

  //! Retrieve the RSS of self process in bytes
  static size_t SelfRSS(void);

  //! Retrieve the peak RSS of self process in bytes
  static size_t SelfPeakRSS(void);

  //! Retrieve the total size of physical memory (RAM) in bytes
  static size_t TotalRamSize(void);

  //! Retrieve the available size of physical memory (RAM) in bytes
  static size_t AvailableRamSize(void);

  //! Retrieve the used size of physical memory (RAM) in bytes
  static size_t UsedRamSize(void);

  //! Retrieve the total size of physical memory (RAM) in bytes in container
  static size_t ContainerAwareTotalRamSize(void);
};

}  // namespace ailego
}  // namespace zvec
