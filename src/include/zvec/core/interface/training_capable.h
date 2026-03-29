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

#include <zvec/core/interface/training_session.h>

namespace zvec {
namespace core_interface {

/**
 * @brief Training capability interface for indexes that support post-build training.
 *
 * This interface follows the Capability Pattern, allowing indexes to optionally
 * provide training functionality without polluting the base Index class.
 */
class ITrainingCapable {
 public:
  virtual ~ITrainingCapable() = default;

  virtual ITrainingSession::Pointer CreateTrainingSession() = 0;
};

}  // namespace core_interface
}  // namespace zvec
