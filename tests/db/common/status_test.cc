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
#include <gtest/gtest.h>

using namespace zvec;

TEST(StatusTest, DefaultConstructor) {
  Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::OK);
  EXPECT_EQ(status.message(), "");
}

TEST(StatusTest, ConstructorWithCodeAndMessage) {
  std::string msg = "Test error message";
  Status status(StatusCode::INVALID_ARGUMENT, msg);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), msg);
}

TEST(StatusTest, ConstructorWithRvalueMessage) {
  std::string msg = "Test error message";
  Status status(StatusCode::NOT_FOUND, std::move(msg));

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::NOT_FOUND);
  EXPECT_EQ(status.message(), "Test error message");
}

TEST(StatusTest, CopyConstructor) {
  Status original(StatusCode::INTERNAL_ERROR, "Copy test");
  Status copy(original);

  EXPECT_FALSE(copy.ok());
  EXPECT_EQ(copy.code(), StatusCode::INTERNAL_ERROR);
  EXPECT_EQ(copy.message(), "Copy test");
  EXPECT_EQ(original.code(), copy.code());
  EXPECT_EQ(original.message(), copy.message());
}

TEST(StatusTest, CopyAssignment) {
  Status original(StatusCode::PERMISSION_DENIED, "Assignment test");
  Status assigned;
  assigned = original;

  EXPECT_FALSE(assigned.ok());
  EXPECT_EQ(assigned.code(), StatusCode::PERMISSION_DENIED);
  EXPECT_EQ(assigned.message(), "Assignment test");
}

TEST(StatusTest, MoveConstructor) {
  Status original(StatusCode::RESOURCE_EXHAUSTED, "Move test");
  Status moved(std::move(original));

  EXPECT_FALSE(moved.ok());
  EXPECT_EQ(moved.code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(moved.message(), "Move test");
}

TEST(StatusTest, MoveAssignment) {
  Status original(StatusCode::UNAVAILABLE, "Move assignment test");
  Status moved;
  moved = std::move(original);

  EXPECT_FALSE(moved.ok());
  EXPECT_EQ(moved.code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ(moved.message(), "Move assignment test");
}

TEST(StatusTest, ComparisonOperators) {
  Status status1(StatusCode::INVALID_ARGUMENT, "Error 1");
  Status status2(StatusCode::INVALID_ARGUMENT, "Error 1");
  Status status3(StatusCode::NOT_FOUND, "Error 2");
  Status ok1;
  Status ok2;

  EXPECT_TRUE(status1 == status2);
  EXPECT_FALSE(status1 == status3);
  EXPECT_TRUE(ok1 == ok2);
  EXPECT_FALSE(status1 == ok1);

  EXPECT_FALSE(status1 != status2);
  EXPECT_TRUE(status1 != status3);
  EXPECT_FALSE(ok1 != ok2);
  EXPECT_TRUE(status1 != ok1);
}

TEST(StatusTest, FactoryMethods) {
  auto invalid_arg = Status::InvalidArgument("Invalid arg: ", 42);
  EXPECT_FALSE(invalid_arg.ok());
  EXPECT_EQ(invalid_arg.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_FALSE(invalid_arg.message().empty());

  auto not_found = Status::NotFound("Not found: ", "key");
  EXPECT_FALSE(not_found.ok());
  EXPECT_EQ(not_found.code(), StatusCode::NOT_FOUND);
  EXPECT_FALSE(not_found.message().empty());

  auto already_exists = Status::AlreadyExists("Already exists: ", "item");
  EXPECT_FALSE(already_exists.ok());
  EXPECT_EQ(already_exists.code(), StatusCode::ALREADY_EXISTS);
  EXPECT_FALSE(already_exists.message().empty());

  auto internal_error = Status::InternalError("Internal error: ", "details");
  EXPECT_FALSE(internal_error.ok());
  EXPECT_EQ(internal_error.code(), StatusCode::INTERNAL_ERROR);
  EXPECT_FALSE(internal_error.message().empty());

  auto permission_denied =
      Status::PermissionDenied("Permission denied for: ", "resource");
  EXPECT_FALSE(permission_denied.ok());
  EXPECT_EQ(permission_denied.code(), StatusCode::PERMISSION_DENIED);
  EXPECT_FALSE(permission_denied.message().empty());
}

TEST(StatusTest, OKFactory) {
  auto ok = Status::OK();
  EXPECT_TRUE(ok.ok());
  EXPECT_EQ(ok.code(), StatusCode::OK);
  EXPECT_EQ(ok.message(), "");
}

TEST(StatusTest, CStringConversion) {
  Status status(StatusCode::UNKNOWN, "C string test");
  EXPECT_STREQ(status.c_str(), "C string test");

  Status ok_status;
  EXPECT_STREQ(ok_status.c_str(), "");
}

TEST(StatusTest, OutputStreamOperator) {
  Status status(StatusCode::INVALID_ARGUMENT, "Stream test");
  std::ostringstream oss;
  oss << status;
  EXPECT_FALSE(oss.str().empty());
  EXPECT_NE(oss.str().find(GetDefaultMessage(StatusCode::INVALID_ARGUMENT)),
            std::string::npos);
  EXPECT_NE(oss.str().find("Stream test"), std::string::npos);

  Status ok_status;
  std::ostringstream oss2;
  oss2 << ok_status;
  EXPECT_FALSE(oss2.str().empty());
  EXPECT_NE(oss2.str().find("OK"), std::string::npos);
}