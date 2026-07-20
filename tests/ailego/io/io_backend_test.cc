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

#include <ailego/io/io_backend_def.h>
#include <gtest/gtest.h>
#include <string>
#include <zvec/ailego/io/io_backend.h>

namespace zvec {
namespace ailego {

TEST(IOBackendTest, StableBackendValues) {
  EXPECT_EQ(static_cast<uint32_t>(IOBackendType::kPread), 0U);
  EXPECT_EQ(static_cast<uint32_t>(IOBackendType::kLibAio), 1U);
  EXPECT_EQ(static_cast<uint32_t>(IOBackendType::kPosixAio), 2U);
}

TEST(IOBackendTest, BackendNames) {
  EXPECT_STREQ(IOBackendTypeName(IOBackendType::kPread), "pread");
  EXPECT_STREQ(IOBackendTypeName(IOBackendType::kLibAio), "libaio");
  EXPECT_STREQ(IOBackendTypeName(IOBackendType::kPosixAio), "posix_aio");
}

#if defined(__APPLE__) || defined(__MACH__)
TEST(IOBackendTest, MacOSUsesPosixAio) {
  EXPECT_EQ(current_io_backend_type(), IOBackendType::kPosixAio);
  EXPECT_NE(current_io_backend_description().find("kqueue"), std::string::npos);
}
#endif

}  // namespace ailego
}  // namespace zvec
