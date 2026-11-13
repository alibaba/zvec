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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <arrow/status.h>

namespace zvec::sqlengine::scalar_jit {

struct ExpressionLimits {
  size_t nodes = 1024;
  size_t depth = 64;
  size_t key_bytes = 64 * 1024;
  size_t input_bytes = 1024 * 1024;
  size_t in_values = 4096;
};

struct RuntimeConfig {
  bool enabled = true;
  bool dump_ir = false;
  bool profile = false;
  size_t max_compilations = 2;
  size_t max_live_programs = 256;
  ExpressionLimits limits;
  std::string error;
};

struct MemoryUsage {
  // Object-file size is a code-size proxy, NOT executable pages or LLVM RSS.
  uint64_t object_bytes = 0;
  uint64_t ir_bytes = 0;
  uint64_t owned_bytes = 0;
};

struct RuntimeSnapshot {
  bool built = false;
  bool enabled = false;
  bool initialized = false;
  std::string reason;
  std::string initialization_error;
  RuntimeConfig config;
  std::map<std::string, uint64_t> counters;
};

// The PID is captured at library load, before any lazy mutex or LLVM state.
// All cold entry points must check this before accessing inherited globals.
bool InOriginalProcess();
const RuntimeConfig &GetRuntimeConfig();
std::string UnavailableReason();
RuntimeSnapshot GetRuntimeSnapshot();
arrow::Status Unavailable(const char *reason);
std::string FailureReason(const arrow::Status &status,
                          const char *fallback = "compile_failed");
uint64_t SteadyNanos();
void RecordCounter(const char *name, uint64_t amount = 1) noexcept;
void RecordTiming(const char *stage, uint64_t nanos) noexcept;
void RecordSelection(const std::string &reason) noexcept;
void RecordInitialization(const arrow::Status &status);
// Internal injectable initialization seam; no production fault switches.
arrow::Status InitializeBackend(
    const std::function<arrow::Status()> &initialize);

// Process-wide admission also covers direct, uncached compiler entry points.
class CompilePermit {
 public:
  CompilePermit() = default;
  CompilePermit(const CompilePermit &) = delete;
  CompilePermit &operator=(const CompilePermit &) = delete;
  ~CompilePermit();
  arrow::Status acquire();

 private:
  bool held_ = false;
};

// Ownership-based, process-wide accounting; never used inside evaluation.
class MemoryTicket {
 public:
  MemoryTicket() = default;
  MemoryTicket(const MemoryTicket &) = delete;
  MemoryTicket &operator=(const MemoryTicket &) = delete;
  ~MemoryTicket();
  void set(const char *kind, MemoryUsage bytes, uint64_t snapshots = 0);

 private:
  const char *kind_ = nullptr;
  MemoryUsage bytes_;
  uint64_t snapshots_ = 0;
};

class StageTimer {
 public:
  explicit StageTimer(const char *stage)
      : stage_(stage), start_(SteadyNanos()) {}
  ~StageTimer() {
    RecordTiming(stage_, SteadyNanos() - start_);
  }

 private:
  const char *stage_;
  uint64_t start_;
};

// Only construct on the explicitly selected profiling path. Unsampled calls
// touch thread-local state only. Each backend samples independently.
class ExecutionSample {
 public:
  explicit ExecutionSample(bool jit);
  ~ExecutionSample();

 private:
  bool jit_;
  bool sampled_;
  uint64_t start_ = 0;
};

}  // namespace zvec::sqlengine::scalar_jit
