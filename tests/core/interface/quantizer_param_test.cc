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

#include <gtest/gtest.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {
namespace {

TEST(QuantizerParam, SerializesCanonicalUniformNames) {
  QuantizerParam uint7(QuantizerType::kUniformUint7);
  QuantizerParam uint8(QuantizerType::kUniformUint8);
  EXPECT_NE(std::string::npos, uint7.SerializeToJson().find("kUniformUint7"));
  EXPECT_NE(std::string::npos, uint8.SerializeToJson().find("kUniformUint8"));
}

}  // namespace
}  // namespace zvec::core_interface
