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

#include "zvec/db/config.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/global_resource.h"
#include "zvec/db/status.h"

using namespace zvec;

#if GTEST_HAS_DEATH_TEST
namespace {

std::weak_ptr<ailego::Logger> &ShutdownLoggerProbe() {
  static std::weak_ptr<ailego::Logger> logger;
  return logger;
}

int shutdown_logger_destructions = 0;

class ShutdownProbeLogger : public ailego::Logger {
 public:
  ~ShutdownProbeLogger() override {
    ++shutdown_logger_destructions;
  }
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *, va_list) override {}
};

void CheckLoggerShutdown() {
  // Keeping a weak reference alive prevents the control block from being
  // freed. A shutdown hook accessing an already-destroyed broker shared_ptr
  // then exposes its second release instead of relying on allocator reuse.
  const auto use_count = ShutdownLoggerProbe().use_count();
  if (shutdown_logger_destructions != 1 || use_count != 0) {
    std::fprintf(stderr, "logger destructions=%d, remaining owners=%ld\n",
                 shutdown_logger_destructions, use_count);
    std::fflush(stderr);
    std::_Exit(1);
  }
  std::fputs("logger shutdown released exactly once\n", stderr);
  std::fflush(stderr);
  std::_Exit(0);
}

void ExitAfterConfigInitialization() {
  auto &probe = ShutdownLoggerProbe();
  if (std::atexit(CheckLoggerShutdown) != 0) {
    std::_Exit(2);
  }
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = 128ULL * 1024 * 1024;
  config.query_thread_count = 1;
  config.optimize_thread_count = 1;
  if (!GlobalConfig::Instance().initialize(config).ok()) {
    std::_Exit(3);
  }
  auto logger = std::make_shared<ShutdownProbeLogger>();
  probe = logger;
  ailego::LoggerBroker::Register(std::move(logger));
  std::exit(0);
}

void CheckFailedConfigInitializationKeepsLoggingAvailable() {
  constexpr uint64_t kExistingPoolBytes = 85ULL * 1024 * 1024;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(kExistingPoolBytes));

  auto original_logger = std::make_shared<ShutdownProbeLogger>();
  ailego::LoggerBroker::Register(original_logger);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_ERROR);

  auto &config = GlobalConfig::Instance();
  const auto original_memory_limit = config.memory_limit_bytes();
  const auto original_query_threads = config.query_thread_count();
  const auto original_log_level = config.log_level();
  GlobalConfig::ConfigData requested;
  requested.memory_limit_bytes = 128ULL * 1024 * 1024;
  requested.query_thread_count = original_query_threads == 1 ? 2 : 1;
  requested.optimize_thread_count = 1;
  requested.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::kDebug);

  const auto status = config.initialize(requested);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(original_memory_limit, config.memory_limit_bytes());
  EXPECT_EQ(original_query_threads, config.query_thread_count());
  EXPECT_EQ(original_log_level, config.log_level());
  EXPECT_EQ(kExistingPoolBytes, pool.capacity());
  // Logging is initialized before resources and remains available for
  // diagnosing a later failure. Configuration is not published.
  EXPECT_TRUE(
      ailego::LoggerBroker::IsLevelEnabled(ailego::Logger::LEVEL_DEBUG));
  auto initialized_logger = ailego::LoggerBroker::Register(original_logger);
  ASSERT_NE(nullptr, initialized_logger);
  EXPECT_NE(original_logger, initialized_logger);
  ailego::LoggerBroker::Register(initialized_logger);

  // A compatible retry must still return the terminal initialization error,
  // not reconfigure logging or publish a new configuration.
  requested.memory_limit_bytes = 100ULL * 1024 * 1024;
  requested.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::kFatal);
  const auto repeated_status = config.initialize(requested);
  EXPECT_EQ(status.code(), repeated_status.code());
  EXPECT_EQ(status.message(), repeated_status.message());
  EXPECT_EQ(original_memory_limit, config.memory_limit_bytes());
  EXPECT_EQ(original_query_threads, config.query_thread_count());
  EXPECT_EQ(original_log_level, config.log_level());
  EXPECT_EQ(kExistingPoolBytes, pool.capacity());
  EXPECT_TRUE(
      ailego::LoggerBroker::IsLevelEnabled(ailego::Logger::LEVEL_DEBUG));
  EXPECT_EQ(initialized_logger,
            ailego::LoggerBroker::Register(initialized_logger));
}

}  // namespace
#endif  // GTEST_HAS_DEATH_TEST

TEST(ConfigDeathTest,
     FailedResourceInitializationKeepsLoggingWithoutPublishing) {
#if GTEST_HAS_DEATH_TEST
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        CheckFailedConfigInitializationKeepsLoggingAvailable();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP()
      << "Process-isolated exit tests are not supported on this platform";
#endif
}

TEST(ConfigDeathTest, LoggingShutdownPrecedesBrokerStaticDestruction) {
#if GTEST_HAS_DEATH_TEST
  // Re-exec so no earlier test can initialize the broker and mask the
  // first-initialization ordering of its destructor and the exit hook.
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(ExitAfterConfigInitialization(), ::testing::ExitedWithCode(0),
              "logger shutdown released exactly once");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP()
      << "Process-isolated exit tests are not supported on this platform";
#endif
}

class ConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset GlobalConfig for each test
    // Note: Since GlobalConfig is a singleton and uses atomic flag,
    // we cannot easily reset it. In a real test environment, you might
    // need to use a testing framework that supports fixture reset or
    // modify the GlobalConfig to support reset for testing purposes.
  }
};

TEST_F(ConfigTest, ThreadConfigDataDefaults) {
  GlobalConfig::ConfigData config;

  ASSERT_GT(config.query_thread_count, 0u);
  ASSERT_EQ(config.query_thread_count, config.optimize_thread_count);
  ASSERT_FALSE(config.query_thread_binding);
  ASSERT_FALSE(config.optimize_thread_binding);
}

TEST_F(ConfigTest, InitializeWithDefaultConfig) {
  GlobalConfig::ConfigData config;

  // Test initialization with default config
  auto status = GlobalConfig::Instance().initialize(config);
  ASSERT_TRUE(status.ok()) << "Initialization failed: " << status.message();

  // Verify default values
  ASSERT_GT(GlobalConfig::Instance().memory_limit_bytes(), 0);
  ASSERT_EQ(GlobalConfig::Instance().log_level(),
            GlobalConfig::LogLevel::kWarn);
  ASSERT_EQ(GlobalConfig::Instance().log_type(), "ConsoleLogger");
  ASSERT_GT(GlobalConfig::Instance().query_thread_count(), 0);
  ASSERT_FALSE(GlobalConfig::Instance().query_thread_binding());
  ASSERT_EQ(GlobalConfig::Instance().invert_to_forward_scan_ratio(), 0.9f);
  ASSERT_EQ(GlobalConfig::Instance().brute_force_by_keys_ratio(), 0.1f);
  ASSERT_EQ(GlobalConfig::Instance().fts_brute_force_by_keys_ratio(), 0.05f);
  ASSERT_GT(GlobalConfig::Instance().optimize_thread_count(), 0);
  ASSERT_FALSE(GlobalConfig::Instance().optimize_thread_binding());
}

TEST_F(ConfigTest, InitializeWithCustomConsoleLogConfig) {
  GlobalConfig::ConfigData config;
  config.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::kDebug);
  config.memory_limit_bytes = 1024 * 1024 * 1024;  // 1GB
  config.query_thread_count = 4;
  config.optimize_thread_count = 2;

  auto status = GlobalConfig::Instance().initialize(config);
  // First initialization should succeed
  if (status.code() == StatusCode::INVALID_ARGUMENT &&
      status.message().find("already initialized") != std::string::npos) {
    // If already initialized, skip this test
    GTEST_SKIP() << "GlobalConfig already initialized";
  }
}

TEST_F(ConfigTest, InitializeWithCustomFileLogConfig) {
  GlobalConfig::ConfigData config;
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>(
      GlobalConfig::LogLevel::kInfo, "/tmp/logs", "test.log", 1024, 14);
  config.log_config = file_config;
  config.memory_limit_bytes = 2 * 1024 * 1024 * 1024ULL;  // 2GB
  config.query_thread_count = 8;
  config.optimize_thread_count = 4;

  auto status = GlobalConfig::Instance().initialize(config);
  // First initialization should succeed
  if (status.code() == StatusCode::INVALID_ARGUMENT &&
      status.message().find("already initialized") != std::string::npos) {
    // If already initialized, skip this test
    GTEST_SKIP() << "GlobalConfig already initialized";
  }
}

TEST_F(ConfigTest, DoubleInitializationSilentlyFails) {
  GlobalConfig::ConfigData config;

  auto status1 = GlobalConfig::Instance().initialize(config);
  // If first initialization failed due to already being initialized
  if (status1.code() == StatusCode::INVALID_ARGUMENT &&
      status1.message().find("already initialized") != std::string::npos) {
    // Try again with a fresh config
    auto status2 = GlobalConfig::Instance().initialize(config);
    ASSERT_FALSE(status2.ok());
    ASSERT_EQ(status2.code(), StatusCode::INVALID_ARGUMENT);
    ASSERT_NE(status2.message().find("already initialized"), std::string::npos);
  } else {
    // First initialization succeeded, second should fail
    ASSERT_TRUE(status1.ok());

    // The second initialization is allowed but becomes a no-op
    auto status2 = GlobalConfig::Instance().initialize(config);
    ASSERT_TRUE(status2.ok());
  }
}

TEST_F(ConfigTest, ValidateConfigWithInvalidMemoryLimit) {
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = 0;  // Invalid value

  GlobalConfig
      config_instance;  // Create a local instance for testing validation
  auto status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("memory_limit_bytes must be greater than"),
            std::string::npos);
}

TEST_F(ConfigTest, InvalidInitializeCanBeCorrected) {
  GlobalConfig config_instance;
  GlobalConfig::ConfigData invalid;
  invalid.memory_limit_bytes = 0;
  auto invalid_status = config_instance.initialize(invalid);
  ASSERT_FALSE(invalid_status.ok());
  ASSERT_EQ(StatusCode::INVALID_ARGUMENT, invalid_status.code());

  GlobalConfig::ConfigData valid;
  auto valid_status = config_instance.initialize(valid);
  ASSERT_TRUE(valid_status.ok()) << valid_status.message();
  EXPECT_EQ(valid.memory_limit_bytes, config_instance.memory_limit_bytes());
}

TEST_F(ConfigTest, FailedResourceInitializationDoesNotPublishConfig) {
  ASSERT_EQ(0, GlobalResource::Instance().initialize());

  GlobalConfig config_instance;
  const uint32_t original_query_threads = config_instance.query_thread_count();
  GlobalConfig::ConfigData requested;
  const auto &published = GlobalConfig::Instance();
  requested.memory_limit_bytes = published.memory_limit_bytes();
  requested.query_thread_count = published.query_thread_count() + 1;
  if (requested.query_thread_count == original_query_threads) {
    ++requested.query_thread_count;
  }
  requested.query_thread_binding = published.query_thread_binding();
  requested.optimize_thread_count = published.optimize_thread_count();
  requested.optimize_thread_binding = published.optimize_thread_binding();

  const auto status = config_instance.initialize(requested);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(original_query_threads, config_instance.query_thread_count());
}

TEST_F(ConfigTest, ValidateConfigWithInvalidQueryThreadCount) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 0;  // Invalid value

  GlobalConfig config_instance;
  auto status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("query_thread_count must be greater than 0"),
            std::string::npos);
}

TEST_F(ConfigTest, ValidateConfigWithInvalidRatios) {
  GlobalConfig::ConfigData config;

  // Test invalid invert_to_forward_scan_ratio
  config.invert_to_forward_scan_ratio = -0.1f;
  GlobalConfig config_instance;
  auto status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find(
                "invert_to_forward_scan_ratio must be between 0 and 1"),
            std::string::npos);

  // Test invalid brute_force_by_keys_ratio
  config.invert_to_forward_scan_ratio = 0.9f;  // Reset to valid value
  config.brute_force_by_keys_ratio = 1.5f;     // Invalid value
  status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find(
                "brute_force_by_keys_ratio must be between 0 and 1"),
            std::string::npos);

  // Test invalid fts_brute_force_by_keys_ratio
  config.brute_force_by_keys_ratio = 0.1f;       // Reset to valid value
  config.fts_brute_force_by_keys_ratio = -0.5f;  // Invalid value
  status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find(
                "fts_brute_force_by_keys_ratio must be between 0 and 1"),
            std::string::npos);
}

TEST_F(ConfigTest, ValidateConfigWithInvalidFileLogSettings) {
  GlobalConfig::ConfigData config;

  // Test with empty log directory
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>();
  file_config->dir = "";
  config.log_config = file_config;

  GlobalConfig config_instance;
  auto status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_dir cannot be empty"),
            std::string::npos);

  // Test with empty basename
  file_config->dir = "/tmp/logs";
  file_config->basename = "";
  status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_file basename cannot be empty"),
            std::string::npos);

  // Test with invalid file size
  file_config->basename = "test.log";
  file_config->file_size = 0;
  status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log file_size must be greater than"),
            std::string::npos);

  // Test with invalid overdue days
  file_config->file_size = 1024;
  file_config->overdue_days = 0;
  status = config_instance.validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_overdue_days must be greater than 0"),
            std::string::npos);
}

TEST_F(ConfigTest, LogLevelEnumValues) {
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::kDebug), 0);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::kInfo), 1);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::kWarn), 2);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::kError), 3);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::kFatal), 4);
}

TEST_F(ConfigTest, LogConfigPolymorphism) {
  auto console_config = std::make_shared<GlobalConfig::ConsoleLogConfig>();
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>();

  ASSERT_EQ(console_config->get_logger_type(), CONSOLE_LOG_TYPE_NAME);
  ASSERT_EQ(file_config->get_logger_type(), FILE_LOG_TYPE_NAME);
}

TEST_F(ConfigTest, InitializePublishesAnImmutableLogConfigSnapshot) {
  GlobalConfig config_instance;
  const GlobalConfig::LogConfig &original_log = config_instance.log_config();
  const auto original_level = original_log.level;

  GlobalConfig::ConfigData requested;
  const auto &process_config = GlobalConfig::Instance();
  requested.memory_limit_bytes = process_config.memory_limit_bytes();
  requested.query_thread_count = process_config.query_thread_count();
  requested.query_thread_binding = process_config.query_thread_binding();
  requested.optimize_thread_count = process_config.optimize_thread_count();
  requested.optimize_thread_binding = process_config.optimize_thread_binding();
  auto caller_owned_log = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::kInfo);
  requested.log_config = caller_owned_log;

  const auto status = config_instance.initialize(requested);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(GlobalConfig::LogLevel::kInfo, config_instance.log_level());

  caller_owned_log->level = GlobalConfig::LogLevel::kFatal;
  EXPECT_EQ(GlobalConfig::LogLevel::kInfo, config_instance.log_level());
  EXPECT_EQ(original_level, original_log.level);
}

// jieba_dict_dir is the only ConfigData field that can be written outside
// of initialize() — language SDKs call set_default_jieba_dict_dir() at
// module-load to register the dict path they bundled. The setter is
// independent of the initialize() one-shot lifecycle.
TEST_F(ConfigTest, JiebaDictDirSetterIsIndependentOfInitialize) {
  auto saved = GlobalConfig::Instance().jieba_dict_dir();

  // Setter works regardless of whether initialize was called.
  GlobalConfig::Instance().set_default_jieba_dict_dir("/tmp/zvec/dict-A");
  ASSERT_EQ(GlobalConfig::Instance().jieba_dict_dir(), "/tmp/zvec/dict-A");

  // Last writer wins.
  GlobalConfig::Instance().set_default_jieba_dict_dir("/tmp/zvec/dict-B");
  ASSERT_EQ(GlobalConfig::Instance().jieba_dict_dir(), "/tmp/zvec/dict-B");

  // Empty clears.
  GlobalConfig::Instance().set_default_jieba_dict_dir("");
  ASSERT_EQ(GlobalConfig::Instance().jieba_dict_dir(), "");

  GlobalConfig::Instance().set_default_jieba_dict_dir(saved);
}
