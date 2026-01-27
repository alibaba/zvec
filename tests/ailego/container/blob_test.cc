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

#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/ailego/container/blob.h>

using namespace zvec;

TEST(BlobWrap, Constructor) {
  ailego::BlobWrap blob1;
  EXPECT_EQ(0u, blob1.size());
  EXPECT_FALSE(blob1.buffer());
  EXPECT_FALSE(blob1.is_valid());

  std::string buf2;
  ailego::BlobWrap blob2(buf2);
  EXPECT_EQ(0u, blob2.size());
  EXPECT_TRUE(blob2.buffer());
  EXPECT_FALSE(blob2.is_valid());

  buf2.append("good...");
  EXPECT_EQ(0u, blob2.size());
  EXPECT_TRUE(blob2.buffer());

  ailego::BlobWrap blob3(blob2);
  EXPECT_EQ(0u, blob3.size());
  EXPECT_TRUE(blob3.buffer());

  std::string buf4("........");
  ailego::BlobWrap blob4(buf4);
  EXPECT_NE(0u, blob4.size());
  EXPECT_TRUE(blob4.buffer());
  EXPECT_TRUE(blob4.is_valid());

  ailego::BlobWrap blob5(std::move(blob4));
  EXPECT_EQ(0u, blob4.size());
  EXPECT_FALSE(blob4.buffer());
  EXPECT_NE(0u, blob5.size());
  EXPECT_TRUE(blob5.buffer());

  blob4 = blob5;
  EXPECT_NE(0u, blob4.size());
  EXPECT_TRUE(blob4.buffer());
  EXPECT_NE(0u, blob5.size());
  EXPECT_TRUE(blob5.buffer());

  blob1 = std::move(blob5);
  EXPECT_NE(0u, blob1.size());
  EXPECT_TRUE(blob1.buffer());
  EXPECT_EQ(0u, blob5.size());
  EXPECT_FALSE(blob5.buffer());
}

TEST(BlobWrap, General) {
  ailego::BlobWrap blob1;
  std::string buf1("11111111111");

  blob1.mount(buf1);
  EXPECT_TRUE(blob1.buffer());

  blob1.umount();
  EXPECT_FALSE(blob1.buffer());

  std::string buf2("22222222222222222");
  const ailego::BlobWrap blob2(buf2);
  EXPECT_TRUE(
      std::is_const<
          typename std::remove_pointer<decltype(blob2.buffer())>::type>::value);

  ailego::BlobWrap blob3;
  std::string buf3("3333");
  blob3.mount(const_cast<char *>(buf3.data()), buf3.size());
  blob3.copy(blob2);
  EXPECT_FALSE(
      std::is_const<
          typename std::remove_pointer<decltype(blob3.buffer())>::type>::value);

  std::string buf4("444444444444444444444");
  ailego::BlobWrap blob4;
  blob4.mount(buf4);
  blob4.copy(buf1.data(), buf1.size());

  std::string buf5("55555");
  ailego::BlobWrap blob5(buf5);
  blob5.zero();
  blob4.copy(buf4);
}
