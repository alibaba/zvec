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
#include <zvec/core/framework/index_holder.h>
#include "utility/ordinal_access_holder.h"

namespace zvec::core_interface {

// Build-only input for the direct add() API. Unlike Collection merge, this
// path has no existing provider to borrow. Store its sole owned vector copy
// in BufferStorage, with compact slots independent of the external doc ID.
// Mutation and prepare() are serialized by the owning index; readers may run
// concurrently once prepared, but input must not change during a build.
class BufferedInput final : public core::IndexHolder,
                            public core::OrdinalAccessHolder,
                            public std::enable_shared_from_this<BufferedInput> {
 public:
  BufferedInput(core::IndexQueryMeta meta, std::string prefix);
  ~BufferedInput() override;
  int add(uint32_t key, const void *data);
  int fetch(uint32_t key, std::string *out) const;
  int prepare();
  int flush();
  void clear();

  size_t count() const override;
  size_t dimension() const override;
  core::IndexMeta::DataType data_type() const override;
  size_t element_size() const override;
  bool multipass() const override {
    return true;
  }
  core::IndexHolder::Iterator::Pointer create_iterator() override;
  int create_ordinal_reader(
      core::OrdinalAccessHolder::Reader::Pointer *out) override;

 private:
  class Reader;
  class Iterator;
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace zvec::core_interface
