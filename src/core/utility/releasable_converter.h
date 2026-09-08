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

namespace zvec {
namespace core {

// Optional internal capability; does not change the public IndexConverter ABI.
// Release only the converter's result holder (and its input ownership chain),
// preserving trained parameters, metadata, statistics and any rotation state.
// Previously returned holders remain valid while retained by their callers.
class ReleasableConverter {
 public:
  virtual ~ReleasableConverter() = default;
  virtual void release_result() = 0;
};

}  // namespace core
}  // namespace zvec
