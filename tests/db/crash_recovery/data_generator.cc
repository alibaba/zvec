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


#include <unistd.h>
#include <filesystem>
#include <thread>
#include <zvec/db/collection.h>
#include "zvec/ailego/logger/logger.h"
#include "utility.h"


constexpr int kBatchSize = 20;
constexpr int kBatchDelayMs = 10;


struct Config {
  std::string path;
  int start_id = 0;
  int end_id = 0;
  std::string operation;  // "insert", "upsert", "update", "delete"
  int version = 999999;
};


bool ParseArgs(int argc, char **argv, Config &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--path" && i + 1 < argc) {
      config.path = argv[++i];
    } else if (arg == "--start" && i + 1 < argc) {
      config.start_id = std::stoi(argv[++i]);
    } else if (arg == "--end" && i + 1 < argc) {
      config.end_id = std::stoi(argv[++i]);
    } else if (arg == "--op" && i + 1 < argc) {
      config.operation = argv[++i];
    } else if (arg == "--version" && i + 1 < argc) {
      config.version = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      return false;
    }
  }

  // Validate required arguments
  if (config.path.empty() || config.operation.empty() ||
      config.start_id >= config.end_id || config.version == 999999) {
    return false;
  }

  // Validate operation
  if (config.operation != "insert" && config.operation != "upsert" &&
      config.operation != "update" && config.operation != "delete") {
    std::cerr << "Error: Invalid operation '" << config.operation
              << "'. Must be 'insert', 'upsert', 'update', or 'delete'."
              << std::endl;
    return false;
  }

  return true;
}


void PrintUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --path <collection_path> --start <start_id> --end <end_id> "
               "--op <operation>"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Arguments:" << std::endl;
  std::cout << "  --path      Path to the collection (required)" << std::endl;
  std::cout << "  --start     Starting document ID (inclusive, required)"
            << std::endl;
  std::cout << "  --end       Ending document ID (exclusive, required)"
            << std::endl;
  std::cout
      << "  --op        Operation: insert, upsert, update, or delete (required)"
      << std::endl;
  std::cout << "  --version   Operation: version (required)" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  # Insert 1000 documents (pk_0 to pk_999)" << std::endl;
  std::cout << "  " << program
            << " --path ./test_db --start 0 --end 1000 --op insert --version 0"
            << std::endl;
  std::cout << std::endl;
  std::cout << "  # Update documents 1000-1999" << std::endl;
  std::cout
      << "  " << program
      << " --path ./test_db --start 1000 --end 2000 --op update --version 1"
      << std::endl;
  std::cout << std::endl;
  std::cout << "  # Upsert documents 0-499" << std::endl;
  std::cout << "  " << program
            << " --path ./test_db --start 0 --end 500 --op upsert --version 2"
            << std::endl;
}


int main(int argc, char **argv) {
  Config config;

  // Parse arguments
  if (!ParseArgs(argc, argv, config)) {
    PrintUsage(argv[0]);
    return 1;
  }

  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "[data_generator] Current Working Directory: " << cwd.string()
              << std::endl;
  } catch (const std::filesystem::filesystem_error &e) {
    std::cout
        << "[data_generator] Failed to get the current working directory: "
        << e.what() << std::endl;
  }

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Path:      " << config.path << std::endl;
  std::cout << "  Range:     [" << config.start_id << ", " << config.end_id
            << ")" << std::endl;
  std::cout << "  Operation: " << config.operation << std::endl;
  std::cout << "  BatchSize: " << kBatchSize << std::endl;
  std::cout << "  BatchDelay: " << kBatchDelayMs << "ms" << std::endl;
  std::cout << std::endl;

  auto result = zvec::Collection::Open(
      config.path, zvec::CollectionOptions{false, true, 4 * 1024 * 1024});
  if (!result) {
    LOG_ERROR("Failed to open collection[%s]: %s", config.path.c_str(),
              result.error().c_str());
    return -1;
  }

  auto collection = result.value();
  LOG_INFO("Collection[%s] opened successfully", config.path.c_str());

  // Process documents in batches
  int total_docs = config.end_id - config.start_id;
  int processed = 0;
  int batch_num = 0;
  int next_progress_threshold = total_docs / 10;  // 10% increments
  int progress_percent = 0;

  while (config.start_id < config.end_id) {
    int batch_end = std::min(config.start_id + kBatchSize, config.end_id);
    int batch_count = batch_end - config.start_id;

    std::vector<zvec::Doc> docs;
    docs.reserve(batch_count);
    for (int i = config.start_id; i < batch_end; i++) {
      docs.push_back(zvec::CreateTestDoc(i, config.version));
    }

    zvec::Result<zvec::WriteResults> results;
    if (config.operation == "insert") {
      results = collection->Insert(docs);
    } else if (config.operation == "upsert") {
      results = collection->Upsert(docs);
    } else if (config.operation == "update") {
      results = collection->Update(docs);
    } else if (config.operation == "delete") {
      std::vector<std::string> pks{};
      for (const auto &doc : docs) {
        pks.emplace_back(doc.pk());
      }
      results = collection->Delete(pks);
    }
    if (!results) {
      LOG_ERROR("Failed to perform operation[%s], reason: %s",
                config.operation.c_str(), results.error().message().c_str());
      return 1;
    }
    for (auto &s : results.value()) {
      if (!s.ok()) {
        LOG_ERROR("Failed to perform operation[%s], reason: %s",
                  config.operation.c_str(), s.message().c_str());
        return 1;
      }
    }

    processed += batch_count;
    config.start_id = batch_end;
    batch_num++;

    // Print progress every 10%
    if (processed >= next_progress_threshold) {
      progress_percent++;
      LOG_INFO("Progress: %d (%d/%d documents)", progress_percent * 10,
               processed, total_docs);
      next_progress_threshold = (progress_percent + 1) * total_docs / 10;
    }

    // Sleep between batches
    if (config.start_id < config.end_id) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kBatchDelayMs));
    }
  }

  std::cout << std::endl;
  std::cout << "Success! Processed " << processed << " documents in "
            << batch_num << " batches." << std::endl;

  return 0;
}
