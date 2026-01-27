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
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>

using namespace zvec;
using namespace zvec::core;

TEST(BufferStorage, General) {
  ailego::BufferManager::Instance().init(10 * 1024 * 1024, 1);
  std::string file_path = "buffer_storage_test_file";
  ailego::File::Delete(file_path);

  auto write_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(write_storage);
  std::cout << file_path << std::endl;
  EXPECT_NE(0, write_storage->open(file_path, false));

  ailego::Params params;
  EXPECT_EQ(0, write_storage->init(params));
  std::cout << file_path << std::endl;
  EXPECT_EQ(0, write_storage->open(file_path, true));

  IndexMeta meta;
  meta.set_trainer("trainer", 111, ailego::Params());
  meta.set_searcher("searcher", 222, ailego::Params());
  meta.set_builder("builder", 333, ailego::Params());

  EXPECT_EQ(0, IndexHelper::SerializeToStorage(meta, write_storage.get()));
  EXPECT_EQ(0, write_storage->append("AAAA", 1234));
  EXPECT_EQ(0, write_storage->append("BBBB", 1234));
  auto aaaa = write_storage->get("AAAA");
  ASSERT_TRUE(aaaa);
  auto aaaa1 = aaaa->clone();
  ASSERT_TRUE(aaaa1);
  std::string hello = "Hello world!!!";
  EXPECT_EQ(hello.size(), aaaa1->write(0, hello.data(), hello.size()));
  EXPECT_EQ(0, write_storage->close());

  // Reopen it
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  EXPECT_EQ(0, read_storage->open(file_path, false));

  IndexMeta meta2;
  EXPECT_EQ(0, IndexHelper::DeserializeFromStorage(read_storage.get(), &meta2));
  EXPECT_EQ("trainer", meta2.trainer_name());
  EXPECT_EQ("searcher", meta2.searcher_name());
  EXPECT_EQ("builder", meta2.builder_name());
  auto aaaa2 = read_storage->get("AAAA");
  ASSERT_TRUE(aaaa2);
  const void *data;
  EXPECT_EQ(hello.size(), aaaa2->read(0, &data, hello.size()));
  auto aaaa3 = aaaa2->clone();
  ASSERT_TRUE(aaaa3);
  EXPECT_EQ(hello.size(), aaaa3->read(0, &data, hello.size()));
  EXPECT_EQ(hello, std::string((const char *)data, hello.size()));
}
