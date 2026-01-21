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

using namespace zvec;
using namespace zvec::core;

static int GenRandInt(int m, int n) {
  static std::mt19937 gen((std::random_device())());
  return std::uniform_int_distribution<int>(m, n)(gen);
}

static void AddRandomPadding(const std::string &in, const std::string &out,
                             size_t header_padding_size,
                             size_t footer_padding_size) {
  ailego::File out_file;
  out_file.create(out, 0);
  for (size_t i = 0; i < header_padding_size; ++i) {
    uint8_t r = GenRandInt(0, 255);
    out_file.write(&r, 1);
  }

  ailego::File in_file;
  ASSERT_TRUE(in_file.open(in, true));
  std::string buf(in_file.size(), '\0');
  ASSERT_EQ(buf.size(), in_file.read(&buf[0], buf.size()));
  out_file.write(buf.data(), buf.size());

  for (size_t i = 0; i < footer_padding_size; ++i) {
    uint8_t r = GenRandInt(0, 255);
    out_file.write(&r, 1);
  }
}

TEST(MMapFileReadStorage, General) {
  std::string file_path = "mmap_file_container_test_file";
  std::string file_path_padding = "mmap_file_container_test_file_padding";

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);

  IndexMeta meta1;
  meta1.set_trainer("index_trainer", 0, ailego::Params());
  ASSERT_EQ(0, dumper->create(file_path));
  EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper.get()));

  for (size_t i = 0; i < 21; ++i) {
    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello.size(), dumper->write(hello.data(), hello.size()));
    EXPECT_EQ(0, dumper->append(std::to_string(i), hello.size(), 0, 0));
  }
  ASSERT_EQ(0, dumper->close());
  size_t header_paddings = GenRandInt(0, 1024);
  size_t footer_paddings = GenRandInt(0, 1024);
  AddRandomPadding(file_path, file_path_padding, header_paddings,
                   footer_paddings);
  ailego::File file;
  file.open(file_path_padding, true);
  int64_t header_offset =
      GenRandInt(0, 1) ? header_paddings : header_paddings - file.size();
  int64_t footer_offset =
      (GenRandInt(0, 1) ? file.size() : 0) - footer_paddings;

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("proxima.mmap_file.container.memory_locking", true);
  params.set("proxima.mmap_file.container.memory_warmup", true);
  params.set("proxima.mmap_file.container.checksum_validation", true);
  params.set("proxima.mmap_file.container.header_offset", header_offset);
  params.set("proxima.mmap_file.container.footer_offset", footer_offset);
  ASSERT_EQ(0, container->init(params));

  IndexMeta meta2;
  EXPECT_EQ(0u, container->get_all().size());
  EXPECT_EQ("", meta2.trainer_name());
  EXPECT_EQ("", meta2.searcher_name());
  ASSERT_EQ(0, container->open(file_path_padding, false));
  EXPECT_EQ(0, IndexHelper::DeserializeFromStorage(container.get(), &meta2));
  EXPECT_EQ(23u, container->get_all().size());
  EXPECT_EQ("index_trainer", meta2.trainer_name());
  EXPECT_EQ("", meta2.searcher_name());

  for (size_t i = 0; i < 21; ++i) {
    auto seg = container->get(std::to_string(i));
    auto seg1 = seg->clone();

    const void *data = nullptr;
    EXPECT_EQ(seg1->data_size(), seg1->read(0, &data, seg1->data_size()));
    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello, std::string((const char *)data, seg1->data_size()));
  }
  container->cleanup();
}
