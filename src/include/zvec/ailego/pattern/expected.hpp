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

// Compatibility shim.  The vendored tl::expected moved to
// <zvec/detail/expected.hpp> so the stable API can reach it without a
// default package installing anything under zvec/ailego/.  Engine code
// keeps this path; new code should include the detail header directly.

#pragma once

#include <zvec/detail/expected.hpp>
