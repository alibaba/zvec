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

// Compatibility shim.  Float16 and FloatHelper moved to
// <zvec/detail/float_helper.h> so the stable API can reach them without a
// default package installing anything under zvec/ailego/.  The namespace
// and the class names are unchanged, so this is purely a path move.
//
// The header this stands in for also pulled in
// <zvec/ailego/internal/platform.h>, and engine code all over src/ reaches the
// AILEGO_* feature macros and the helpers next to them (ailego_assert,
// static_cast_from_float_to_uint8, ...) through that path -- for example
// cube.h -> type_helper.h -> here.  Keep providing it: this shim is engine-only
// and never installed, so the stable closure is unaffected, while
// <zvec/detail/float_helper.h> stays free of engine includes for the published
// API's sake and re-derives the one macro it needs on its own.

#pragma once

#include <zvec/ailego/internal/platform.h>
#include <zvec/detail/float_helper.h>
