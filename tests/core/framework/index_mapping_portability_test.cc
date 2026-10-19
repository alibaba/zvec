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

#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_mapping.h>

namespace zvec {
namespace core {

TEST(IndexMappingPortability, ForeignPageAppendMetadataChainAndReopen) {
  for (size_t writer_page : {4096u, 65536u}) {
    const std::string path =
        "index_mapping_portability_" + std::to_string(writer_page) + ".tmp";
    ailego::File file;
    ASSERT_TRUE(file.create(path, writer_page * 2));
    IndexFormat::MetaHeader header;
    IndexFormat::SetupMetaHeader(
        &header, writer_page - sizeof(IndexFormat::MetaFooter), writer_page);
    ASSERT_EQ(sizeof(header), file.write(0, &header, sizeof(header)));
    // Leave room for one entry only, to force a new metadata section on append.
    std::vector<char> table(64, 0);
    IndexFormat::SegmentMeta meta{};
    meta.segment_id_offset = 58;
    meta.data_size = sizeof(uint64_t);
    meta.padding_size = writer_page - meta.data_size;
    std::memcpy(table.data(), &meta, sizeof(meta));
    std::memcpy(table.data() + meta.segment_id_offset, "first", 6);
    ASSERT_EQ(table.size(), file.write(header.meta_footer_offset - table.size(),
                                       table.data(), table.size()));
    IndexFormat::MetaFooter footer;
    IndexFormat::SetupMetaFooter(&footer);
    footer.segment_count = 1;
    footer.segments_meta_size = table.size();
    footer.segments_meta_crc =
        ailego::Crc32c::Hash(table.data(), table.size(), 0);
    footer.content_size = writer_page;
    footer.total_size = writer_page * 2;
    IndexFormat::UpdateMetaFooter(&footer, 0);
    ASSERT_EQ(sizeof(footer),
              file.write(header.meta_footer_offset, &footer, sizeof(footer)));
    const uint64_t original = 1234567;
    ASSERT_EQ(sizeof(original),
              file.write(writer_page, &original, sizeof(original)));
    file.close();

    IndexMapping mapping;
    ASSERT_EQ(0, mapping.open(path, false, false));
    auto *first = mapping.map("first", false, false);
    ASSERT_NE(nullptr, first);
    EXPECT_EQ(original, *static_cast<uint64_t *>(first->data()));
    ASSERT_EQ(0, mapping.append("second", 4096));
    auto *second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, second);
    *static_cast<uint64_t *>(second->data()) = 7654321;
    second->meta()->data_size = sizeof(uint64_t);
    second->meta()->padding_size -= sizeof(uint64_t);
    second->set_dirty();
    mapping.refresh(1);
    ASSERT_EQ(0, mapping.flush());
    mapping.close();

    ASSERT_EQ(0, mapping.open(path, false, false));
    first = mapping.map("first", false, false);
    second = mapping.map("second", false, false);
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);
    EXPECT_EQ(original, *static_cast<uint64_t *>(first->data()));
    EXPECT_EQ(7654321u, *static_cast<uint64_t *>(second->data()));
    mapping.close();
    ASSERT_TRUE(ailego::File::Delete(path));
  }
}

}  // namespace core
}  // namespace zvec
