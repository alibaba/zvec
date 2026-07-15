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

// I/O backend type enum and helpers.
//
// This is the public, dependency-free part of the I/O backend abstraction.
// It defines the IOBackendType enum and human-readable name/description
// helpers so that public headers can reference IOBackendType without pulling
// in the internal IOBackend singleton or libaio_loader.

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

// Returns a human-readable name for the given backend type.
inline const char *IOBackendTypeName(IOBackendType type) {
  switch (type) {
    case IOBackendType::kLibAio:
      return "libaio";
    case IOBackendType::kPread:
      return "pread";
  }
  return "unknown";
}

// Returns a human-readable description for the given backend type.
// When the backend is kPread, includes installation guidance for libaio.
inline const char *IOBackendDescription(IOBackendType type) {
  switch (type) {
    case IOBackendType::kLibAio:
      return "libaio async I/O backend loaded at runtime via dlopen().";
    case IOBackendType::kPread:
      return "No async I/O backend available. Install libaio (e.g. "
             "'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) "
             "and retry. DiskAnn will fall back to synchronous pread() \u2014 "
             "performance will be degraded.";
  }
  return "Unknown I/O backend.";
}

// Returns the currently active I/O backend type.
// Triggers backend initialization on first call (libaio > pread).
IOBackendType current_io_backend_type();

// Returns a human-readable description of the currently active I/O backend.
// When only pread is available, includes installation guidance for libaio.
std::string current_io_backend_description();

}  // namespace ailego
}  // namespace zvec
