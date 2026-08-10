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

// Implementation of the convenience helpers declared in the public header
// zvec/ailego/io/io_backend.h.
//
// This translation unit is the single place that bridges the dependency-free
// public header to the internal IOBackend singleton, so that public headers
// can expose current_io_backend_type() / current_io_backend_description()
// without pulling in libaio_loader or io_backend_def.h.

#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/io/io_backend.h>

namespace zvec {
namespace ailego {

IOBackendType current_io_backend_type() {
  return IOBackend::Instance().available();
}

std::string current_io_backend_description() {
  auto type = IOBackend::Instance().available();
  return IOBackendDescription(type);
}

}  // namespace ailego
}  // namespace zvec
