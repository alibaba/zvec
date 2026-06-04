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

#include <gtest/gtest.h>
#include "zvec/db/status.h"

#define ZVEC_EXPECT_OK(expr)                                                 \
  do {                                                                       \
    const auto _zvec_status = (expr);                                        \
    EXPECT_TRUE(_zvec_status.ok()) << _zvec_status;                          \
  } while (false)

#define ZVEC_ASSERT_OK(expr)                                                 \
  do {                                                                       \
    const auto _zvec_status = (expr);                                        \
    ASSERT_TRUE(_zvec_status.ok()) << _zvec_status;                          \
  } while (false)

#define ZVEC_EXPECT_STATUS(expr, expected_code)                              \
  do {                                                                       \
    const auto _zvec_status = (expr);                                        \
    EXPECT_EQ((expected_code), _zvec_status.code()) << _zvec_status;         \
  } while (false)

#define ZVEC_ASSERT_STATUS(expr, expected_code)                              \
  do {                                                                       \
    const auto _zvec_status = (expr);                                        \
    ASSERT_EQ((expected_code), _zvec_status.code()) << _zvec_status;         \
  } while (false)
