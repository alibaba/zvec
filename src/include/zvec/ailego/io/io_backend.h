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

// I/O backend type enum.
//
// This is the public, dependency-free part of the I/O backend abstraction.
// It defines the IOBackendType enum and the convenience helpers
// current_io_backend_type() / current_io_backend_description() so that
// public headers can reference IOBackendType without pulling in the
// internal IOBackend singleton or libaio_loader.

#pragma once

#include <string>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

// Supported I/O backend types.
enum class IOBackendType {
  kPread,   // Synchronous pread() — no async I/O
  kLibAio,  // libaio loaded at runtime via dlopen()
};

// Returns the currently active I/O backend type.
// Triggers backend initialization on first call (libaio > pread).
IOBackendType current_io_backend_type();

// Returns a human-readable description of the currently active I/O backend.
// When only pread is available, includes installation guidance for libaio.
std::string current_io_backend_description();

}  // namespace ailego
}  // namespace zvec
