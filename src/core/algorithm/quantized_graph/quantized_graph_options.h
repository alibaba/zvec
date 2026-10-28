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
namespace zvec::core {
struct QuantizedGraphSearchOptions {
  // Defaults select the existing metric-specific preset. Beam zero disables
  // routing while retaining the same model/mean for a controlled comparison.
  float margin = -1;
  float bucket_width = 0;
  // Hard cap on expanded nodes, including filtered nodes; 0 uses the preset.
  uint32_t max_expansions = 0;
  uint32_t route_beam = 1;
};
}  // namespace zvec::core
