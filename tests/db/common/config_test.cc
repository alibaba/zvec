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
#include <cstdint>
#include <gtest/gtest.h>
#include "db/common/constants.h"
#include "zvec/db/status.h"

using namespace zvec;

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

TEST_F(ConfigTest, InitializeWithDefaultConfig) {
  GlobalConfig::ConfigData config;

  // Test initialization with default config
  auto status = GlobalConfig::Instance().Initialize(config);
  ASSERT_TRUE(status.ok()) << "Initialization failed: " << status.message();

  // Verify default values
  ASSERT_GT(GlobalConfig::Instance().memory_limit_bytes(), 0);
  ASSERT_EQ(GlobalConfig::Instance().log_level(), GlobalConfig::LogLevel::WARN);
  ASSERT_EQ(GlobalConfig::Instance().log_type(), "ConsoleLogger");
  ASSERT_GT(GlobalConfig::Instance().query_thread_count(), 0);
  ASSERT_EQ(GlobalConfig::Instance().invert_to_forward_scan_ratio(), 0.9f);
  ASSERT_EQ(GlobalConfig::Instance().brute_force_by_keys_ratio(), 0.1f);
  ASSERT_GT(GlobalConfig::Instance().optimize_thread_count(), 0);
}

TEST_F(ConfigTest, InitializeWithCustomConsoleLogConfig) {
  GlobalConfig::ConfigData config;
  config.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      GlobalConfig::LogLevel::DEBUG);
  config.memory_limit_bytes = 1024 * 1024 * 1024;  // 1GB
  config.query_thread_count = 4;
  config.optimize_thread_count = 2;

  auto status = GlobalConfig::Instance().Initialize(config);
  // First initialization should succeed
  if (status.code() == StatusCode::INVALID_ARGUMENT &&
      status.message().find("already initialized") != std::string::npos) {
    // If already initialized, skip this test
    GTEST_SKIP() << "GlobalConfig already initialized";
  }

  ASSERT_TRUE(status.ok()) << "Initialization failed: " << status.message();

  // Verify custom values
  ASSERT_EQ(GlobalConfig::Instance().memory_limit_bytes(), 1024 * 1024 * 1024);
  ASSERT_EQ(GlobalConfig::Instance().log_level(),
            GlobalConfig::LogLevel::DEBUG);
  ASSERT_EQ(GlobalConfig::Instance().log_type(), "ConsoleLogger");
  ASSERT_EQ(GlobalConfig::Instance().query_thread_count(), 4);
  ASSERT_EQ(GlobalConfig::Instance().optimize_thread_count(), 2);
}

TEST_F(ConfigTest, InitializeWithCustomFileLogConfig) {
  GlobalConfig::ConfigData config;
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>(
      GlobalConfig::LogLevel::INFO, "/tmp/logs", "test.log", 1024, 14);
  config.log_config = file_config;
  config.memory_limit_bytes = 2 * 1024 * 1024 * 1024ULL;  // 2GB
  config.query_thread_count = 8;
  config.optimize_thread_count = 4;

  auto status = GlobalConfig::Instance().Initialize(config);
  // First initialization should succeed
  if (status.code() == StatusCode::INVALID_ARGUMENT &&
      status.message().find("already initialized") != std::string::npos) {
    // If already initialized, skip this test
    GTEST_SKIP() << "GlobalConfig already initialized";
  }

  ASSERT_TRUE(status.ok()) << "Initialization failed: " << status.message();

  // Verify custom values
  ASSERT_EQ(GlobalConfig::Instance().memory_limit_bytes(),
            2 * 1024 * 1024 * 1024ULL);
  ASSERT_EQ(GlobalConfig::Instance().log_level(), GlobalConfig::LogLevel::INFO);
  ASSERT_EQ(GlobalConfig::Instance().log_type(), "AppendLogger");
  ASSERT_EQ(GlobalConfig::Instance().log_dir(), "/tmp/logs");
  ASSERT_EQ(GlobalConfig::Instance().log_file_basename(), "test.log");
  ASSERT_EQ(GlobalConfig::Instance().log_file_size(), 1024);
  ASSERT_EQ(GlobalConfig::Instance().log_overdue_days(), 14);
  ASSERT_EQ(GlobalConfig::Instance().query_thread_count(), 8);
  ASSERT_EQ(GlobalConfig::Instance().optimize_thread_count(), 4);
}

TEST_F(ConfigTest, DoubleInitializationFails) {
  GlobalConfig::ConfigData config;

  auto status1 = GlobalConfig::Instance().Initialize(config);
  // If first initialization failed due to already being initialized
  if (status1.code() == StatusCode::INVALID_ARGUMENT &&
      status1.message().find("already initialized") != std::string::npos) {
    // Try again with a fresh config
    auto status2 = GlobalConfig::Instance().Initialize(config);
    ASSERT_FALSE(status2.ok());
    ASSERT_EQ(status2.code(), StatusCode::INVALID_ARGUMENT);
    ASSERT_NE(status2.message().find("already initialized"), std::string::npos);
  } else {
    // First initialization succeeded, second should fail
    ASSERT_TRUE(status1.ok());

    auto status2 = GlobalConfig::Instance().Initialize(config);
    ASSERT_FALSE(status2.ok());
    ASSERT_EQ(status2.code(), StatusCode::INVALID_ARGUMENT);
    ASSERT_NE(status2.message().find("already initialized"), std::string::npos);
  }
}

TEST_F(ConfigTest, ValidateConfigWithInvalidMemoryLimit) {
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = 0;  // Invalid value

  GlobalConfig
      config_instance;  // Create a local instance for testing validation
  auto status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("memory_limit_bytes must be greater than"),
            std::string::npos);
}

TEST_F(ConfigTest, ValidateConfigWithInvalidQueryThreadCount) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 0;  // Invalid value

  GlobalConfig config_instance;
  auto status = config_instance.Validate(config);
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
  auto status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find(
                "invert_to_forward_scan_ratio must be between 0 and 1"),
            std::string::npos);

  // Test invalid brute_force_by_keys_ratio
  config.invert_to_forward_scan_ratio = 0.9f;  // Reset to valid value
  config.brute_force_by_keys_ratio = 1.5f;     // Invalid value
  status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find(
                "brute_force_by_keys_ratio must be between 0 and 1"),
            std::string::npos);
}

TEST_F(ConfigTest, ValidateConfigWithInvalidFileLogSettings) {
  GlobalConfig::ConfigData config;

  // Test with empty log directory
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>();
  file_config->dir = "";
  config.log_config = file_config;

  GlobalConfig config_instance;
  auto status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_dir cannot be empty"),
            std::string::npos);

  // Test with empty basename
  file_config->dir = "/tmp/logs";
  file_config->basename = "";
  status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_file basename cannot be empty"),
            std::string::npos);

  // Test with invalid file size
  file_config->basename = "test.log";
  file_config->file_size = 0;
  status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log file_size must be greater than"),
            std::string::npos);

  // Test with invalid overdue days
  file_config->file_size = 1024;
  file_config->overdue_days = 0;
  status = config_instance.Validate(config);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_NE(status.message().find("log_overdue_days must be greater than 0"),
            std::string::npos);
}

TEST_F(ConfigTest, LogLevelEnumValues) {
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::DEBUG), 0);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::INFO), 1);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::WARN), 2);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::ERROR), 3);
  ASSERT_EQ(static_cast<int>(GlobalConfig::LogLevel::FATAL), 4);
}

TEST_F(ConfigTest, LogConfigPolymorphism) {
  auto console_config = std::make_shared<GlobalConfig::ConsoleLogConfig>();
  auto file_config = std::make_shared<GlobalConfig::FileLogConfig>();

  ASSERT_EQ(console_config->GetLoggerType(), CONSOLE_LOG_TYPE_NAME);
  ASSERT_EQ(file_config->GetLoggerType(), FILE_LOG_TYPE_NAME);
}