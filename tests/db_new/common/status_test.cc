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

#include "zvec/db/status.h"

#include <sstream>
#include <string>

#include <gtest/gtest.h>

using namespace zvec;

TEST(StatusTest, OkStatusHasStableCodeAndEmptyMessage) {
  const Status status;

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(StatusCode::OK, status.code());
  EXPECT_TRUE(status.message().empty());
  EXPECT_STREQ("", status.c_str());
  EXPECT_STREQ("OK", GetDefaultMessage(status.code()));
}

TEST(StatusTest, FactoryMethodsPreserveCodeAndMessage) {
  const auto invalid = Status::InvalidArgument("field ", "age", " expected ",
                                               42);
  const auto missing = Status::NotFound("collection ", "users");
  const auto duplicate = Status::AlreadyExists("pk ", "001");

  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT, invalid.code());
  EXPECT_EQ("field age expected 42", invalid.message());

  EXPECT_EQ(StatusCode::NOT_FOUND, missing.code());
  EXPECT_EQ("collection users", missing.message());

  EXPECT_EQ(StatusCode::ALREADY_EXISTS, duplicate.code());
  EXPECT_EQ("pk 001", duplicate.message());
}

TEST(StatusTest, EqualityUsesMessageForFailuresOnly) {
  EXPECT_EQ(Status::OK(), Status());
  EXPECT_EQ(Status(), Status(StatusCode::OK, "ignored for ok"));

  EXPECT_EQ(Status::NotFound("same"), Status::NotFound("same"));
  EXPECT_NE(Status::NotFound("left"), Status::NotFound("right"));
  EXPECT_NE(Status::NotFound("same"), Status::InvalidArgument("same"));
}

TEST(StatusTest, StreamOutputIsReadable) {
  std::ostringstream ok_stream;
  ok_stream << Status::OK();
  EXPECT_EQ("OK", ok_stream.str());

  std::ostringstream error_stream;
  error_stream << Status::PermissionDenied("readonly");

  EXPECT_NE(std::string::npos, error_stream.str().find("Permission denied"));
  EXPECT_NE(std::string::npos, error_stream.str().find("readonly"));
}
