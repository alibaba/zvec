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
#include <iostream>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_helper.h"
#include "zvec/core/framework/index_segment_storage.h"

using namespace zvec;
using namespace zvec::core;

TEST(FileDumper, General) {
  std::string file_path = "file_dumper_test_file";

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);

  IndexMeta meta1;
  meta1.set_trainer("index_trainer", 0, ailego::Params());
  ASSERT_EQ(0, dumper->create(file_path));
  EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper.get()));

  for (size_t i = 0; i < 10; ++i) {
    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello.size(), dumper->write(hello.data(), hello.size()));
    EXPECT_EQ(0, dumper->append(std::to_string(i), hello.size(), 0, 0));
  }
  ASSERT_EQ(0, dumper->close());

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("proxima.mmap_file.container.memory_locking", true);
  params.set("proxima.mmap_file.container.memory_warmup", true);
  params.set("proxima.mmap_file.container.checksum_validation", true);
  ASSERT_EQ(0, container->init(params));

  IndexMeta meta2;
  EXPECT_EQ("", meta2.trainer_name());
  ASSERT_EQ(0, container->open(file_path, false));
  EXPECT_EQ(0, IndexHelper::DeserializeFromStorage(container.get(), &meta2));
  EXPECT_EQ("index_trainer", meta2.trainer_name());

  for (size_t i = 0; i < 10; ++i) {
    auto seg = container->get(std::to_string(i));
    IndexStorage::MemoryBlock data;
    EXPECT_EQ(seg->data_size(), seg->read(0, data, seg->data_size()));

    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello, std::string((const char *)data.data(), seg->data_size()));
  }
}

TEST(IndexSegmentDumper, General) {
  std::string file_path = "index_segment_dumper_test_file";

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(file_path));

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "AAAAA");

    IndexMeta meta1;
    meta1.set_trainer("index_trainer", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 0; i < 10; ++i) {
      std::string hello = "A: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "BBBBB");

    IndexMeta meta1;
    meta1.set_builder("index_builder", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 100; i < 110; ++i) {
      std::string hello = "B: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "CCCCC");

    IndexMeta meta1;
    meta1.set_converter("index_converter", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 1000; i < 1010; ++i) {
      std::string hello = "C: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }
  ASSERT_EQ(0, dumper->close());

  ///// Read data with container

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("proxima.mmap_file.container.memory_locking", true);
  params.set("proxima.mmap_file.container.memory_warmup", true);
  params.set("proxima.mmap_file.container.checksum_validation", true);
  ASSERT_EQ(0, container->init(params));
  ASSERT_EQ(0, container->open(file_path, false));

  // Exercise nested unpacking with independently owned, file-backed reads.
  auto file_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(file_storage);
  ASSERT_EQ(0, file_storage->init(ailego::Params()));
  ASSERT_EQ(0, file_storage->open(file_path, false));
  auto parent = file_storage->get("AAAAA", 1);
  ASSERT_TRUE(parent);
  IndexSegmentStorage nested(parent);
  ASSERT_EQ(0, nested.open("", false));
  auto segment = nested.get("3", 0);
  ASSERT_TRUE(segment);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(segment->data_size(),
            segment->read(0, block, segment->data_size()));
  EXPECT_EQ("A: Hello world!!! #3",
            std::string(static_cast<const char *>(block.data()),
                        segment->data_size()));
}


TEST(FileReadStorage, MemoryBlockReadsRetainIndependentBuffers) {
  const std::string file_path = "file_read_storage_memory_block_test";
  const std::string first(64, 'A');
  const std::string second(8192, 'B');
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(file_path));
  ASSERT_EQ(first.size(), dumper->write(first.data(), first.size()));
  ASSERT_EQ(second.size(), dumper->write(second.data(), second.size()));
  ASSERT_EQ(0, dumper->append("payload", first.size() + second.size(), 0, 0));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(file_path, false));
  auto segment = storage->get("payload", 1);
  ASSERT_TRUE(segment);
  IndexStorage::MemoryBlock first_block;
  IndexStorage::MemoryBlock second_block;
  ASSERT_EQ(first.size(), segment->read(0, first_block, first.size()));
  ASSERT_EQ(second.size(),
            segment->read(first.size(), second_block, second.size()));
  EXPECT_EQ(first, std::string(static_cast<const char *>(first_block.data()),
                               first.size()));
  EXPECT_EQ(second, std::string(static_cast<const char *>(second_block.data()),
                                second.size()));
  ASSERT_EQ(0u, segment->read(segment->capacity() + 1, second_block, 1));
  EXPECT_EQ(nullptr, second_block.data());
  ASSERT_EQ(0, storage->close());
  segment.reset();
  EXPECT_EQ(first, std::string(static_cast<const char *>(first_block.data()),
                               first.size()));
  std::remove(file_path.c_str());
}
