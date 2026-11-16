// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
namespace zvec::core {
// Bounded byte reader shared by external files and embedded index segments.
struct QuantizedGraphRouterReader {
  std::string_view data;
  size_t offset = 0;
  bool operator()(void *out, size_t bytes) {
    if (bytes > data.size() - offset) return false;
    std::memcpy(out, data.data() + offset, bytes);
    offset += bytes;
    return true;
  }
};
inline constexpr const char *kQuantizedGraphRouterSegment =
    "hnsw.quantized_graph.router";
inline constexpr size_t kMaxQuantizedGraphRouterBytes = size_t{1} << 30;
inline bool ReadQuantizedGraphRouterFile(const char *path, std::string &data) {
  data.clear();
  if (!path) return false;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto size = in.tellg();
  if (!in || size <= 0 || uint64_t(size) > kMaxQuantizedGraphRouterBytes)
    return false;
  data.resize(size_t(size));
  in.seekg(0);
  in.read(data.data(), data.size());
  if (!in) {
    data.clear();
    return false;
  }
  return true;
}
}  // namespace zvec::core
