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
#include <string>

namespace zvec {
namespace core {

/*! Index Module
 */
class IndexModule {
 public:
  //! Index Module Pointer
  typedef std::shared_ptr<IndexModule> Pointer;

  //! Destructor
  virtual ~IndexModule(void) {}

  //! Retrieve debug information
  virtual std::string debug_string(void) const {
    return std::string();
  }

  //! Retrieve name of module
  const std::string &name(void) const {
    return name_;
  }

  //! Retrieve revision of module
  uint32_t revision(void) const {
    return revision_;
  }

 protected:
  friend struct IndexFactory;

  //! Set name of module
  void set_name(const std::string &str) {
    name_ = str;
  }

  //! Set revision of module
  void set_revision(uint32_t val) {
    revision_ = val;
  }

 private:
  //! Members
  uint32_t revision_{0u};
  std::string name_{};
};

}  // namespace core
}  // namespace zvec
