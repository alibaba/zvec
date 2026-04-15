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


#ifndef _WIN32
#include <unistd.h>
#endif
#include <filesystem>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include "zvec/ailego/logger/logger.h"


struct Config {
  std::string path;
};


bool ParseArgs(int argc, char **argv, Config &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--path" && i + 1 < argc) {
      config.path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      return false;
    }
  }

  // Validate required arguments
  if (config.path.empty()) {
    return false;
  }

  return true;
}


void PrintUsage(const char *program) {
  std::cout << "Usage: " << program << " --path <collection_path>" << std::endl;
  std::cout << std::endl;
  std::cout << "Arguments:" << std::endl;
  std::cout << "  --path         Path to the collection (required)"
            << std::endl;
}


int main(int argc, char **argv) {
  Config config;

  // Parse arguments
  if (!ParseArgs(argc, argv, config)) {
    PrintUsage(argv[0]);
#ifdef __ANDROID__
    _exit(1);
#endif
    return 1;
  }

  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "[collection_optimizer] Current Working Directory: "
              << cwd.string() << std::endl;
  } catch (const std::filesystem::filesystem_error &e) {
    std::cout << "[collection_optimizer] Failed to get the current working "
                 "directory: "
              << e.what() << std::endl;
  }

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Path:        " << config.path << std::endl;
  std::cout << std::endl;

  // Scope 'result' so its shared_ptr is released before we call _exit().
  zvec::Collection::Ptr collection;
  {
    auto result =
        zvec::Collection::Open(config.path, zvec::CollectionOptions{false, true});
    if (!result) {
      LOG_ERROR("Failed to open collection[%s]: %s", config.path.c_str(),
                result.error().c_str());
#ifdef __ANDROID__
      _exit(1);
#endif
      return -1;
    }
    collection = result.value();
  }
  std::cout << "Collection[" << config.path.c_str() << "] opened successfully"
            << std::endl;

  // Print initial stats
  std::cout << "Initial stats: " << collection->Stats()->to_string_formatted()
            << std::endl;

  auto s = collection->Optimize(zvec::OptimizeOptions{2});
  if (s.ok()) {
    std::cout << "Optimize completed successfully" << std::endl;
    // Print final stats
    std::cout << "Final stats: " << collection->Stats()->to_string_formatted()
              << std::endl;
#ifdef __ANDROID__
    // On Android with c++_static STL, static destructors of glog/gflags
    // crash during process teardown.  Use _exit() to skip them.
    collection->Flush();
    collection.reset();
    sync();
    std::cout.flush();
    std::cerr.flush();
    fflush(stdout);
    fflush(stderr);
    _exit(0);
#endif
    return 0;
  } else {
    std::cout << "Optimize failed: " << s.message() << std::endl;
#ifdef __ANDROID__
    collection.reset();
    sync();
    std::cout.flush();
    std::cerr.flush();
    fflush(stdout);
    fflush(stderr);
    _exit(1);
#endif
    return 1;
  }
}
