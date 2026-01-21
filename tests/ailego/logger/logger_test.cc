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

#include <gtest/gtest.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

using namespace zvec;
using namespace zvec::ailego;

static void DoLogging() {
  static int log_count = 0;
  LOG_INFO("DoLogging: %d", ++log_count);
}

static void DoErrLogging() {
  static int err_log_count = 0;
  LOG_ERROR("DoErrLogging: %d", ++err_log_count);
}

TEST(IndexLogger, General) {
  ASSERT_TRUE(ailego::Factory<Logger>::Has("ConsoleLogger"));

  for (int i = 0; i < 10; ++i) {
    LoggerBroker::SetLevel(i);
    LOG_DEBUG("level: %d, %s", i, "LOG_DEBUG");
    LOG_INFO("level: %d, %s", i, "LOG_INFO");
    LOG_WARN("level: %d, %s", i, "LOG_WARN");
    LOG_ERROR("level: %d, %s", i, "LOG_ERROR");
    LOG_FATAL("level: %d, %s", i, "LOG_FATAL");
  }

  LoggerBroker::SetLevel(0);
  LOG_DEBUG("%s", std::string("LOG_DEBUG").c_str());
  LOG_INFO("%s", std::string("LOG_INFO").c_str());
  LOG_WARN("%s", std::string("LOG_WARN").c_str());
  LOG_ERROR("%s", std::string("LOG_ERROR").c_str());
  LOG_FATAL("%s", std::string("LOG_FATAL").c_str());

  ThreadPool pool;
  for (uint32_t i = 0; i < 20; ++i) {
    pool.enqueue(Closure::New(DoLogging));
  }
  for (uint32_t i = 0; i < 20; ++i) {
    pool.enqueue(Closure::New(DoErrLogging));
  }
  pool.wake_all();
  pool.wait_finish();

  LoggerBroker::Unregister();
  LOG_DEBUG("%s", "LOG_DEBUG");
  LOG_INFO("%s", "LOG_INFO");
  LOG_WARN("%s", "LOG_WARN");
  LOG_ERROR("%s", "LOG_ERROR");
  LOG_FATAL("%s", "LOG_FATAL");
}