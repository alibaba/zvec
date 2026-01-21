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

#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(TimeHelper, Monotime) {
  std::cout << "NanoSeconds: " << ailego::Monotime::NanoSeconds() << std::endl;
  std::cout << "MicroSeconds: " << ailego::Monotime::MicroSeconds()
            << std::endl;
  std::cout << "MilliSeconds: " << ailego::Monotime::MilliSeconds()
            << std::endl;
  std::cout << "Seconds: " << ailego::Monotime::Seconds() << std::endl;
}

TEST(TimeHelper, Realtime) {
  std::cout << "NanoSeconds: " << ailego::Realtime::NanoSeconds() << std::endl;
  std::cout << "MicroSeconds: " << ailego::Realtime::MicroSeconds()
            << std::endl;
  std::cout << "MilliSeconds: " << ailego::Realtime::MilliSeconds()
            << std::endl;
  std::cout << "Seconds: " << ailego::Realtime::Seconds() << std::endl;

  uint64_t now = ailego::Realtime::Seconds();
  std::cout << "Localtime: " << ailego::Realtime::Localtime(now) << std::endl;
  std::cout << "Gmtime: " << ailego::Realtime::Gmtime(now) << std::endl;
  std::cout << "Localtime: " << ailego::Realtime::Localtime() << std::endl;
  std::cout << "Gmtime: " << ailego::Realtime::Gmtime() << std::endl;
}

TEST(TimeHelper, ElapsedTime) {
  ailego::ElapsedTime stamp;
  std::cout << "elapsed: " << stamp.nano_seconds() << " ns" << std::endl;
  std::cout << "elapsed: " << stamp.micro_seconds() << " us" << std::endl;
  std::cout << "elapsed: " << stamp.milli_seconds() << " ms" << std::endl;
  std::cout << "elapsed: " << stamp.seconds() << " s" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(101));

  stamp.reset();
  std::cout << "elapsed: " << stamp.nano_seconds() << " ns" << std::endl;
  std::cout << "elapsed: " << stamp.micro_seconds() << " us" << std::endl;
  std::cout << "elapsed: " << stamp.milli_seconds() << " ms" << std::endl;
  std::cout << "elapsed: " << stamp.seconds() << " s" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(101));

  stamp.reset();
  std::cout << "elapsed: " << stamp.nano_seconds() << " ns" << std::endl;
  std::cout << "elapsed: " << stamp.micro_seconds() << " us" << std::endl;
  std::cout << "elapsed: " << stamp.milli_seconds() << " ms" << std::endl;
  std::cout << "elapsed: " << stamp.seconds() << " s" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(101));

  stamp.reset();
  std::cout << "elapsed: " << stamp.nano_seconds() << " ns" << std::endl;
  std::cout << "elapsed: " << stamp.micro_seconds() << " us" << std::endl;
  std::cout << "elapsed: " << stamp.milli_seconds() << " ms" << std::endl;
  std::cout << "elapsed: " << stamp.seconds() << " s" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(101));

  std::cout << "elapsed: " << stamp.nano_seconds() << " ns" << std::endl;
  std::cout << "elapsed: " << stamp.micro_seconds() << " us" << std::endl;
  std::cout << "elapsed: " << stamp.milli_seconds() << " ms" << std::endl;
  std::cout << "elapsed: " << stamp.seconds() << " s" << std::endl;
}
