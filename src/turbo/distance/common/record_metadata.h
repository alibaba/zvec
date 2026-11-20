// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo::record {

// Record codes and metadata are byte-packed, including in persisted indexes.
// Fixed-size copies avoid alignment and aliasing assumptions without padding
// the format or changing the arithmetic in the distance kernels.
class Metadata {
 public:
  explicit Metadata(const void *data)
      : data_(static_cast<const uint8_t *>(data)) {}

  float operator[](size_t index) const {
    float value;
    std::memcpy(&value, data_ + index * sizeof(float), sizeof(value));
    return value;
  }

  // INT8 records store an integer code sum after four FP32 fields.
  int32_t code_sum() const {
    int32_t value;
    std::memcpy(&value, data_ + 4 * sizeof(float), sizeof(value));
    return value;
  }

 private:
  const uint8_t *data_;
};

}  // namespace zvec::turbo::record
