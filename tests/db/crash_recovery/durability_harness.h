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

// Shared harness for durability tests that run storage_fault_worker in a
// child process, optionally with storage_shim.c preloaded.

#pragma once

#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "recovery_verifier.h"
#include "utility.h"

extern char **environ;

namespace zvec::checkpoint_test {

namespace fs = std::filesystem;

#if defined(__APPLE__)
constexpr const char *kPreloadVariable = "DYLD_INSERT_LIBRARIES";
#else
constexpr const char *kPreloadVariable = "LD_PRELOAD";
#endif

inline std::string ReadFile(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

inline void WriteFile(const fs::path &path, const std::string &data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(data.data(), static_cast<std::streamsize>(data.size()));
  ASSERT_TRUE(stream.good()) << path;
}

// The shim is found next to the test binaries (TEST_BINARY_DIR/lib, as the
// nightly job unpacks them) before falling back to the build-time path.
inline std::string LocateShim() {
  std::vector<fs::path> candidates;
  if (const char *dir = std::getenv("TEST_BINARY_DIR"))
    candidates.push_back(fs::path(dir) / "lib" / STORAGE_SHIM_NAME);
  candidates.push_back(fs::path("lib") / STORAGE_SHIM_NAME);
  candidates.push_back(STORAGE_SHIM_PATH);
  for (const auto &candidate : candidates) {
    if (fs::exists(candidate)) return fs::canonical(candidate).string();
  }
  return STORAGE_SHIM_PATH;
}

struct WorkerResult {
  int status = 0;
  // Set when the harness itself failed (fork, waitpid, timeout, log file);
  // the worker's status is then meaningless.
  std::string harness_error;
  std::string output;

  bool exited(int code) const {
    return harness_error.empty() && WIFEXITED(status) &&
           WEXITSTATUS(status) == code;
  }
  // Whole-line match: "FLUSH_FAILED" must not match "LATER_FLUSH_FAILED".
  bool has_line(const std::string &line) const {
    std::istringstream lines(output);
    std::string current;
    while (std::getline(lines, current)) {
      if (current == line) return true;
    }
    return false;
  }
  bool has_line_prefix(const std::string &prefix) const {
    std::istringstream lines(output);
    std::string current;
    while (std::getline(lines, current)) {
      if (current.rfind(prefix, 0) == 0) return true;
    }
    return false;
  }
  std::string describe() const {
    std::ostringstream text;
    if (!harness_error.empty()) {
      text << "harness error: " << harness_error;
    } else if (WIFEXITED(status)) {
      text << "exit " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      text << "signal " << WTERMSIG(status);
    } else {
      text << "status " << status;
    }
    return text.str() + "\n" + output;
  }
};

// Runs storage_fault_worker with extra environment variables. A worker that
// stops itself (SIGSTOP after READY) is handed to `on_stop` and resumed.
inline WorkerResult RunWorker(
    const std::vector<std::string> &arguments,
    const std::map<std::string, std::string> &extra_env,
    const fs::path &log_path, const std::function<void()> &on_stop = nullptr) {
  WorkerResult result;
  static const std::string worker = LocateBinary("storage_fault_worker");
  std::vector<std::string> args{worker};
  args.insert(args.end(), arguments.begin(), arguments.end());
  std::vector<std::string> env;
  for (char **entry = environ; *entry; ++entry) {
    const std::string value = *entry;
    if (!extra_env.count(value.substr(0, value.find('='))))
      env.push_back(value);
  }
  for (const auto &[key, value] : extra_env) env.push_back(key + "=" + value);
  std::vector<char *> argv, envp;
  for (auto &value : args) argv.push_back(value.data());
  argv.push_back(nullptr);
  for (auto &value : env) envp.push_back(value.data());
  envp.push_back(nullptr);

  const int log_fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (log_fd < 0) {
    result.harness_error =
        "cannot open log " + log_path.string() + ": " + std::strerror(errno);
    return result;
  }
  const pid_t child = fork();
  if (child < 0) {
    result.harness_error = std::string("fork failed: ") + std::strerror(errno);
    close(log_fd);
    return result;
  }
  if (child == 0) {
    // Only async-signal-safe calls are allowed between fork and exec.
    if (dup2(log_fd, STDOUT_FILENO) < 0 || dup2(log_fd, STDERR_FILENO) < 0) {
      _exit(126);
    }
    close(log_fd);
    execve(argv[0], argv.data(), envp.data());
    _exit(127);
  }
  close(log_fd);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(300);
  int status = 0;
  // Reaps the child after the harness gave up on it.
  auto kill_and_reap = [&] {
    kill(child, SIGKILL);
    int ignored = 0;
    while (waitpid(child, &ignored, 0) < 0 && errno == EINTR) {
    }
  };
  for (;;) {
    const pid_t waited = waitpid(child, &status, WUNTRACED | WNOHANG);
    if (waited == child) {
      if (WIFSTOPPED(status)) {
        if (on_stop) on_stop();
        if (kill(child, SIGCONT) != 0) {
          result.harness_error =
              std::string("SIGCONT failed: ") + std::strerror(errno);
          kill_and_reap();
          break;
        }
        continue;
      }
      break;
    }
    if (waited < 0) {
      if (errno == EINTR) continue;
      result.harness_error =
          std::string("waitpid failed: ") + std::strerror(errno);
      kill_and_reap();
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      result.harness_error = "worker timed out after 300 s";
      kill_and_reap();
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  result.status = status;
  result.output = ReadFile(log_path);
  return result;
}


class DurabilityTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (fs::current_path() / "zvec_durability_XXXXXX").string();
    char *directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    directory_ = fs::canonical(directory);
  }

  void TearDown() override {
    // Failed cases keep their images and logs for inspection.
    if (!directory_.empty() && !HasFailure()) {
      std::error_code error;
      fs::remove_all(directory_, error);
    }
  }

  // Runs the worker with storage_shim.c preloaded and proves the shim was
  // loaded: a missing shim must fail the test instead of passing vacuously.
  WorkerResult RunShimmed(const std::vector<std::string> &arguments,
                          std::map<std::string, std::string> env,
                          const std::string &name,
                          const std::function<void()> &on_stop = nullptr) {
    const auto handshake = directory_ / (name + ".shim");
    const auto shim = LocateShim();
    env[kPreloadVariable] = shim;
    env["ZVEC_SHIM_HANDSHAKE"] = handshake.string();
    auto result =
        RunWorker(arguments, env, directory_ / (name + ".log"), on_stop);
    EXPECT_TRUE(fs::exists(handshake))
        << "storage shim was not loaded into the worker: " << shim;
    return result;
  }

  WorkerResult Prepare(const fs::path &collection, const char *kind) {
    return RunWorker({collection.string(), "prepare", kind, "64"}, {},
                     directory_ / "prepare.log");
  }

  fs::path directory_;
};

}  // namespace zvec::checkpoint_test
