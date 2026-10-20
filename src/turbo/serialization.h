// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <functional>
#include <string>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/core/framework/index_error.h>

namespace zvec::turbo {
using SerializationSink = std::function<int(const void *, size_t)>;

inline int WriteSerialized(const SerializationSink &sink, const void *data,
                           size_t bytes) {
  if (!sink) return core::IndexError_InvalidArgument;
  const auto *p = static_cast<const char *>(data);
  while (bytes) {
    const size_t n = std::min<size_t>(bytes, 64 * 1024);
    const int ret = sink(p, n);
    if (ret != 0) return ret;
    p += n;
    bytes -= n;
  }
  return 0;
}

// The optional reservation belongs to the output owner, declared before its
// string. Legacy caller-owned strings are admission-checked during the call;
// internal persistent outputs retain the reservation for their whole lifetime.
template <typename Emit>
int SerializeString(size_t bytes, const Emit &emit, std::string *out,
                    ailego::MemoryReservation *owner = nullptr) try {
  if (!out) return core::IndexError_InvalidArgument;
  ailego::MemoryReservation reservation;
  reservation.add_array(bytes, 2);
  reservation.add(32);
  std::string candidate(bytes, '\0');
  size_t offset = 0;
  int ret = emit([&](const void *data, size_t n) -> int {
    if (n > bytes - offset) return core::IndexError_InvalidLength;
    if (n) std::memcpy(candidate.data() + offset, data, n);
    offset += n;
    return 0;
  });
  if (ret != 0) return ret;
  if (offset != bytes) return core::IndexError_InvalidLength;
  out->swap(candidate);
  std::string().swap(candidate);  // Free the old output before its charge.
  if (owner) owner->swap(reservation);
  return 0;
} catch (const std::bad_alloc &) {
  return core::IndexError_NoMemory;
}
}  // namespace zvec::turbo
