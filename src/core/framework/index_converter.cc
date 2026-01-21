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

#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_helper.h>

namespace zvec {
namespace core {

int IndexConverter::TrainAndTransform(const IndexConverter::Pointer &converter,
                                      IndexHolder::Pointer holder) {
  auto two_pass_holder = IndexHelper::MakeTwoPassHolder(std::move(holder));
  int ret = converter->train(two_pass_holder);
  if (ret == 0) {
    ret = converter->transform(std::move(two_pass_holder));
  }
  return ret;
}

int IndexConverter::TrainTransformAndDump(
    const IndexConverter::Pointer &converter, IndexHolder::Pointer holder,
    const IndexDumper::Pointer &dumper) {
  int ret = IndexConverter::TrainAndTransform(converter, std::move(holder));
  if (ret == 0) {
    ret = converter->dump(dumper);
  }
  return ret;
}

}  // namespace core
}  // namespace zvec
