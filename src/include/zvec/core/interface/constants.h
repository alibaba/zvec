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

namespace zvec::core_interface {

constexpr static uint32_t kDefaultHnswEfConstruction = 500;
constexpr static uint32_t kDefaultHnswNeighborCnt = 50;

constexpr static uint32_t kDefaultHnswEfSearch = 300;


}  // namespace zvec::core_interface