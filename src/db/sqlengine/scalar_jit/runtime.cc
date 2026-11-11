// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <ailego/internal/cpu_features.h>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include <zvec/ailego/logger/logger.h>
#ifdef ZVEC_ENABLE_SCALAR_JIT
#include "db/sqlengine/planner/scalar_doc_filter.h"
#endif

namespace zvec::sqlengine::scalar_jit {
namespace {
#ifdef _WIN32
const auto origin_pid = _getpid();
#else
const auto origin_pid = getpid();
#endif

class ReasonDetail : public arrow::StatusDetail {
 public:
  explicit ReasonDetail(std::string reason) : reason_(std::move(reason)) {}
  const char *type_id() const override {
    return "zvec.scalar_jit.reason.v1";
  }
  std::string ToString() const override {
    return reason_;
  }

 private:
  std::string reason_;
};

RuntimeConfig ReadConfig() {
  RuntimeConfig config;
  const char *enabled = std::getenv("ZVEC_SCALAR_JIT");
  config.enabled = !enabled || std::strcmp(enabled, "0") != 0;
  auto flag = [](const char *name) {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
  };
  config.dump_ir = flag("ZVEC_SCALAR_JIT_DUMP_IR");
  config.profile = flag("ZVEC_SCALAR_JIT_PROFILE");
  auto positive = [&](const char *name, size_t *destination) {
    const char *value = std::getenv(name);
    if (!value) return;
    size_t number = 0;
    bool valid = *value != '\0';
    for (const char *p = value; *p && valid; ++p) {
      if (*p < '0' || *p > '9' ||
          number > (std::numeric_limits<size_t>::max() - (*p - '0')) / 10) {
        valid = false;
      } else {
        number = number * 10 + (*p - '0');
      }
    }
    if (!valid || !number)
      config.error = "invalid_configuration";
    else
      *destination = number;
  };
  positive("ZVEC_SCALAR_JIT_MAX_COMPILATIONS", &config.max_compilations);
  positive("ZVEC_SCALAR_JIT_MAX_LIVE_PROGRAMS", &config.max_live_programs);
  return config;
}

struct State {
  const RuntimeConfig config = ReadConfig();
  std::mutex mutex;
  std::mutex initialization_mutex;
  uint64_t active = 0;
  struct MemoryTotals {
    uint64_t count = 0;
    MemoryUsage bytes;
  };
  MemoryTotals memory[4];  // program, binding, descriptor, helper
  uint64_t snapshots = 0;
  bool initialized = false;
  bool initialization_failed = false;
  arrow::StatusCode initialization_error = arrow::StatusCode::OK;
  std::map<std::string, uint64_t> counters;
};

State &ProcessState() {
  // Deliberate process lifetime: query threads must be joined before shutdown.
  // No static destructor can race LLVM, diagnostic samples, or Python teardown.
  static auto *state = new State;
  return *state;
}

void ChangeMemory(State &state, const char *kind, MemoryUsage bytes,
                  uint64_t snapshots, bool add) {
  const size_t index = std::strcmp(kind, "program") == 0      ? 0
                       : std::strcmp(kind, "binding") == 0    ? 1
                       : std::strcmp(kind, "descriptor") == 0 ? 2
                                                              : 3;
  auto &total = state.memory[index];
  auto change = [add](uint64_t &counter, uint64_t amount) {
    if (add)
      counter += amount;
    else
      counter -= amount;
  };
  change(total.count, 1);
  change(total.bytes.object_bytes, bytes.object_bytes);
  change(total.bytes.ir_bytes, bytes.ir_bytes);
  change(total.bytes.owned_bytes, bytes.owned_bytes);
  change(state.snapshots, snapshots);
}
}  // namespace

bool InOriginalProcess() {
#ifdef _WIN32
  return origin_pid == _getpid();
#else
  return origin_pid == getpid();
#endif
}

const RuntimeConfig &GetRuntimeConfig() {
  // Child callers must not enter a possibly inherited function-static guard.
  // This thread-local fallback never touches the inherited runtime mutex.
  if (!InOriginalProcess()) {
    thread_local RuntimeConfig child_config;
    child_config.enabled = false;
    return child_config;
  }
  return ProcessState().config;
}

std::string UnavailableReason() {
#ifndef ZVEC_ENABLE_SCALAR_JIT
  return "build_disabled";
#else
  if (!InOriginalProcess()) return "forked_process";
  auto &state = ProcessState();
  if (!state.config.error.empty()) return state.config.error;
  if (!state.config.enabled) return "runtime_disabled";
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.initialization_failed ? "initialization_failed" : "";
#endif
}

RuntimeSnapshot GetRuntimeSnapshot() {
  RuntimeSnapshot result;
#ifdef ZVEC_ENABLE_SCALAR_JIT
  result.built = true;
#endif
  result.reason = UnavailableReason();
  result.enabled = result.reason.empty();
  if (!InOriginalProcess()) return result;
  auto &state = ProcessState();
  std::unique_lock<std::mutex> lock(state.mutex);
  result.config = state.config;
  result.initialized = state.initialized;
  if (state.initialization_error != arrow::StatusCode::OK)
    result.initialization_error =
        arrow::Status(state.initialization_error, "").CodeAsString();
  result.counters = state.counters;
  result.counters["active_compilations"] = state.active;
  result.counters["shared_snapshot_references"] = state.snapshots;
  const char *kinds[] = {"program", "binding", "descriptor", "helper"};
  for (size_t i = 0; i < 4; ++i) {
    const auto prefix = std::string("live_") + kinds[i];
    result.counters[prefix + "s"] = state.memory[i].count;
    result.counters[prefix + "_object_bytes"] =
        state.memory[i].bytes.object_bytes;
    result.counters[prefix + "_ir_bytes"] = state.memory[i].bytes.ir_bytes;
    result.counters[prefix + "_owned_bytes"] =
        state.memory[i].bytes.owned_bytes;
  }
  lock.unlock();
#ifdef ZVEC_ENABLE_SCALAR_JIT
  const auto cache = ScalarDocFilter::CacheStats();
#define CACHE_COUNTER(name) result.counters["cache_" #name] = cache.name
  CACHE_COUNTER(hits);
  CACHE_COUNTER(misses);
  CACHE_COUNTER(in_flight_bypasses);
  CACHE_COUNTER(admission_rejections);
  CACHE_COUNTER(compilations);
  CACHE_COUNTER(failed_compilations);
  CACHE_COUNTER(backoff_hits);
  CACHE_COUNTER(evictions);
  CACHE_COUNTER(entries);
  CACHE_COUNTER(negative_entries);
#undef CACHE_COUNTER
  result.counters["resident_object_bytes"] = cache.resident_memory.object_bytes;
  result.counters["resident_ir_bytes"] = cache.resident_memory.ir_bytes;
  result.counters["resident_owned_bytes"] = cache.resident_memory.owned_bytes;
#endif
  return result;
}

arrow::Status Unavailable(const char *reason) {
  return arrow::Status::NotImplemented("Scalar JIT unavailable")
      .WithDetail(std::make_shared<ReasonDetail>(reason));
}

std::string FailureReason(const arrow::Status &status, const char *fallback) {
  if (status.detail() &&
      std::strcmp(status.detail()->type_id(), "zvec.scalar_jit.reason.v1") == 0)
    return status.detail()->ToString();
  return fallback;
}

uint64_t SteadyNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void RecordCounter(const char *name, uint64_t amount) noexcept try {
  if (!InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.counters[name] += amount;
} catch (...) {
  // Diagnostics must never make an otherwise recoverable query fail.
}

void RecordTiming(const char *stage, uint64_t nanos) noexcept try {
  if (!InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.counters[std::string(stage) + "_count"];
  state.counters[std::string(stage) + "_ns"] += nanos;
} catch (...) {
  // Best-effort sampled telemetry; ownership/admission counters are separate.
}

void RecordSelection(const std::string &reason) noexcept try {
  if (!InOriginalProcess()) return;
  if (GetRuntimeConfig().profile) {
    const auto cpu = ailego::internal::CpuFeatures::UsableX86();
    LOG_DEBUG("Scalar JIT CPU sse42=%d avx=%d avx2=%d avx512f=%d", cpu.sse42,
              cpu.avx, cpu.avx2, cpu.avx512f);
  }
  if (reason.empty())
    RecordCounter("jit_selections");
  else {
    RecordCounter("arrow_selections");
    RecordCounter(("fallback_" + reason).c_str());
  }
} catch (...) {
  // Best-effort diagnostics must not interrupt Arrow fallback.
}

void RecordInitialization(const arrow::Status &status) {
  if (!InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (status.ok()) {
    state.initialized = true;
  } else {
    state.initialization_error = status.code();
    if (!status.IsOutOfMemory()) state.initialization_failed = true;
    LOG_DEBUG("Scalar JIT initialization failed status=%s",
              status.CodeAsString().c_str());
  }
}

arrow::Status InitializeBackend(
    const std::function<arrow::Status()> &initialize) {
  const auto reason = UnavailableReason();
  if (!reason.empty()) return Unavailable(reason.c_str());
  auto &state = ProcessState();
  std::unique_lock<std::mutex> lock(state.initialization_mutex,
                                    std::try_to_lock);
  if (!lock.owns_lock()) return Unavailable("compilation_busy");
  const auto latest = UnavailableReason();
  if (!latest.empty()) return Unavailable(latest.c_str());
  StageTimer timer("initialization");
  arrow::Status status;
  try {
    status = initialize();
  } catch (const std::bad_alloc &) {
    status = arrow::Status::OutOfMemory("Scalar initialization allocation");
  } catch (...) {
    status = arrow::Status::UnknownError("Scalar initialization exception");
  }
  RecordInitialization(status);
  if (!status.ok() && !status.IsOutOfMemory())
    return Unavailable("initialization_failed");
  return status;
}

arrow::Status CompilePermit::Acquire() {
  if (held_)
    return arrow::Status::Invalid("Compilation permit already acquired");
  const auto reason = UnavailableReason();
  if (!reason.empty()) return Unavailable(reason.c_str());
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &active = state.active;
  if (active >= state.config.max_compilations)
    return Unavailable("compilation_busy");
  const auto live = state.memory[0].count;
  if (live >= state.config.max_live_programs ||
      active >= state.config.max_live_programs - live)
    return Unavailable("live_program_limit");
  ++active;
  held_ = true;
  return arrow::Status::OK();
}

CompilePermit::~CompilePermit() {
  if (!held_ || !InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  --state.active;
}

void MemoryTicket::Set(const char *kind, MemoryUsage bytes,
                       uint64_t snapshots) {
  if (!InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (kind_) ChangeMemory(state, kind_, bytes_, snapshots_, false);
  kind_ = kind;
  bytes_ = bytes;
  snapshots_ = snapshots;
  ChangeMemory(state, kind_, bytes_, snapshots_, true);
}

MemoryTicket::~MemoryTicket() {
  if (!kind_ || !InOriginalProcess()) return;
  auto &state = ProcessState();
  std::lock_guard<std::mutex> lock(state.mutex);
  ChangeMemory(state, kind_, bytes_, snapshots_, false);
}

ExecutionSample::ExecutionSample(bool jit) : jit_(jit) {
  thread_local uint64_t sequence[2] = {0, 0};
  sampled_ = (sequence[jit ? 1 : 0]++ % 1024) == 0;
  if (sampled_) start_ = SteadyNanos();
}

ExecutionSample::~ExecutionSample() {
  if (!sampled_) return;
  const auto elapsed = SteadyNanos() - start_;
  RecordTiming(jit_ ? "jit_execution_sample" : "arrow_execution_sample",
               elapsed);
  // Sampled DEBUG evidence is also available to installed SDK users.
  LOG_DEBUG("Scalar JIT execution sample backend=%s ns=%llu",
            jit_ ? "jit" : "arrow", static_cast<unsigned long long>(elapsed));
}

}  // namespace zvec::sqlengine::scalar_jit
