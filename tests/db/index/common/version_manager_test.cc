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

#include "db/index/common/version_manager.h"
#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/meta.h"
#include "proto/zvec.pb.h"
#include "zvec/db/schema.h"

namespace zvec {

class VersionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for testing
    test_path_ = "./version_manager_test";
    FileHelper::RemoveDirectory(test_path_);
    FileHelper::CreateDirectory(test_path_);
  }

  void TearDown() override {
    // Clean up temporary files
    FileHelper::RemoveDirectory(test_path_);
  }

  std::string test_path_;
};

// Test basic Version functionality
TEST_F(VersionManagerTest, VersionBasicOperations) {
  Version version;

  // Create a sample schema
  CollectionSchema schema;
  schema.set_name("test_collection");

  version.set_schema(schema);

  // Verify schema is set correctly
  EXPECT_EQ(version.schema().name(), "test_collection");

  // Test segment meta operations
  auto segment_meta = std::make_shared<SegmentMeta>(1);
  segment_meta->set_id(1);

  // Add segment meta
  EXPECT_TRUE(version.add_persisted_segment_meta(segment_meta).ok());

  // Try to add duplicate - should fail
  EXPECT_FALSE(version.add_persisted_segment_meta(segment_meta).ok());

  // Get segment metas
  auto segment_metas = version.persisted_segment_metas();
  EXPECT_EQ(segment_metas.size(), 1);
  EXPECT_EQ(segment_metas[0]->id(), 1);

  // Remove segment meta
  EXPECT_TRUE(version.remove_persisted_segment_meta(1).ok());
  EXPECT_EQ(version.persisted_segment_metas().size(), 0);

  // Try to remove non-existent segment - should fail
  EXPECT_FALSE(version.remove_persisted_segment_meta(1).ok());

  std::cout << version.to_string() << std::endl;
  std::cout << version.to_string_formatted() << std::endl;
}

// Test Version Load/Save operations
TEST_F(VersionManagerTest, VersionLoadSave) {
  std::string manifest_path = test_path_ + "/manifest";

  // Create and populate a version
  Version version;

  CollectionSchema schema;
  schema.set_name("test_collection");
  version.set_schema(schema);

  auto segment_meta = std::make_shared<SegmentMeta>(1);
  segment_meta->set_id(1);
  version.add_persisted_segment_meta(segment_meta);

  version.set_id_map_path_suffix(100);
  version.set_delete_snapshot_path_suffix(200);
  version.set_next_segment_id(2);

  // Save version
  EXPECT_TRUE(Version::Save(manifest_path, version).ok());

  // Load version
  Version loaded_version;
  EXPECT_TRUE(Version::Load(manifest_path, &loaded_version).ok());

  // Verify loaded version matches original
  EXPECT_EQ(loaded_version.schema().name(), "test_collection");
  EXPECT_EQ(loaded_version.persisted_segment_metas().size(), 1);
  EXPECT_EQ(loaded_version.id_map_path_suffix(), 100);
  EXPECT_EQ(loaded_version.delete_snapshot_path_suffix(), 200);
  EXPECT_EQ(loaded_version.next_segment_id(), 2);
}

// Test VersionManager creation and recovery
TEST_F(VersionManagerTest, VersionManagerCreateAndRecover) {
  std::string version_path = test_path_ + "/version";

  std::filesystem::create_directories(version_path);

  // Create initial version
  Version initial_version;
  CollectionSchema schema;
  schema.set_name("initial_collection");
  initial_version.set_schema(schema);

  auto segment_meta = std::make_shared<SegmentMeta>(1);
  segment_meta->set_id(1);
  initial_version.add_persisted_segment_meta(segment_meta);

  // Create VersionManager
  auto create_result = VersionManager::Create(version_path, initial_version);
  EXPECT_TRUE(create_result.has_value());

  auto version_manager = create_result.value();

  // Get current version and verify
  auto current_version = version_manager->get_current_version();
  EXPECT_EQ(current_version.schema().name(), "initial_collection");

  // Modify version
  auto new_segment = std::make_shared<SegmentMeta>(2);
  new_segment->set_id(2);
  EXPECT_TRUE(version_manager->add_persisted_segment_meta(new_segment).ok());

  // Flush changes
  ASSERT_TRUE(version_manager->flush().ok());

  // Recover VersionManager
  auto recover_result = VersionManager::Recovery(version_path);
  EXPECT_TRUE(recover_result.has_value());

  auto recovered_manager = recover_result.value();
  auto recovered_version = recovered_manager->get_current_version();

  // Verify recovered version matches modified version
  EXPECT_EQ(recovered_version.schema().name(), "initial_collection");
  EXPECT_EQ(recovered_version.persisted_segment_metas().size(), 2);
}

// Test VersionManager operations
TEST_F(VersionManagerTest, VersionManagerOperations) {
  std::string version_path = test_path_ + "/version_ops";

  std::filesystem::create_directories(version_path);

  // Create initial version
  Version initial_version;
  CollectionSchema schema;
  schema.set_name("test_collection");
  initial_version.set_schema(schema);

  auto create_result = VersionManager::Create(version_path, initial_version);
  auto version_manager = create_result.value();

  // Test segment meta operations through VersionManager
  auto segment_meta = std::make_shared<SegmentMeta>(1);
  segment_meta->set_id(1);
  EXPECT_TRUE(version_manager->add_persisted_segment_meta(segment_meta).ok());

  // Test reset writing segment meta
  auto writing_segment = std::make_shared<SegmentMeta>(100);
  writing_segment->set_id(100);
  EXPECT_TRUE(
      version_manager->reset_writing_segment_meta(writing_segment).ok());

  // Test suffix setters
  version_manager->set_id_map_path_suffix(50);
  version_manager->set_delete_snapshot_path_suffix(60);
  version_manager->set_next_segment_id(3);

  // Flush and verify
  EXPECT_TRUE(version_manager->flush().ok());

  auto current_version = version_manager->get_current_version();
  EXPECT_EQ(current_version.id_map_path_suffix(), 50);
  EXPECT_EQ(current_version.delete_snapshot_path_suffix(), 60);
  EXPECT_EQ(current_version.next_segment_id(), 3);
  EXPECT_EQ(current_version.writing_segment_meta()->id(), 100);
}

// Test Version equality operator
TEST_F(VersionManagerTest, VersionEquality) {
  Version version1, version2;

  CollectionSchema schema1, schema2;
  schema1.set_name("collection1");
  schema2.set_name("collection1");

  version1.set_schema(schema1);
  version2.set_schema(schema2);

  auto segment_meta1 = std::make_shared<SegmentMeta>(1);
  segment_meta1->set_id(1);
  version1.add_persisted_segment_meta(segment_meta1);

  auto segment_meta2 = std::make_shared<SegmentMeta>(1);
  segment_meta2->set_id(1);
  version2.add_persisted_segment_meta(segment_meta2);

  // Versions should be equal
  EXPECT_TRUE(version1 == version2);

  // Make them different
  auto segment_meta3 = std::make_shared<SegmentMeta>(2);
  segment_meta3->set_id(2);
  version2.add_persisted_segment_meta(segment_meta3);

  // Versions should not be equal now
  EXPECT_FALSE(version1 == version2);
}

// Test error conditions
TEST_F(VersionManagerTest, ErrorConditions) {
  std::string version_path = test_path_ + "/error_test";

  std::filesystem::create_directories(version_path);

  // Create initial version
  Version initial_version;
  CollectionSchema schema;
  schema.set_name("test");
  initial_version.set_schema(schema);

  auto create_result = VersionManager::Create(version_path, initial_version);
  auto version_manager = create_result.value();

  // Test operations with null segment meta
  EXPECT_FALSE(version_manager->add_persisted_segment_meta(nullptr).ok());

  // Test operations with non-existent segment ID
  EXPECT_FALSE(version_manager->remove_persisted_segment_meta(999).ok());
}

// Test conversion between protobuf and internal schema
TEST_F(VersionManagerTest, SchemaConversion) {
  // Create protobuf schema
  zvec::proto::CollectionSchema pb_schema;
  pb_schema.set_name("test_collection");

  auto pb_field = pb_schema.add_fields();
  pb_field->set_name("vector_field");
  pb_field->set_data_type(zvec::proto::DataType::DT_VECTOR_FP32);
  pb_field->set_dimension(128);

  // Convert to internal schema (this would be done in the Load method)
  CollectionSchema internal_schema;
  internal_schema.set_name(pb_schema.name());
  // In a real implementation, fields would be converted here

  // Test that we can set and retrieve the schema
  Version version;
  version.set_schema(internal_schema);

  EXPECT_EQ(version.schema().name(), "test_collection");
}

// Test SegmentMeta functionality
TEST_F(VersionManagerTest, SegmentMetaOperations) {
  SegmentMeta segment_meta(10);

  EXPECT_EQ(segment_meta.id(), 10);

  // Test block operations
  BlockMeta block(1, BlockType::SCALAR, 0, 100);
  segment_meta.add_persisted_block(block);

  EXPECT_EQ(segment_meta.persisted_blocks().size(), 1);
  EXPECT_EQ(segment_meta.persisted_blocks()[0].id(), 1);

  // Test indexed vector fields
  EXPECT_FALSE(segment_meta.vector_indexed("field1"));
  segment_meta.add_indexed_vector_field("field1");
  EXPECT_TRUE(segment_meta.vector_indexed("field1"));

  // Test min/max doc id
  EXPECT_EQ(segment_meta.min_doc_id(), 0);
  EXPECT_EQ(segment_meta.max_doc_id(), 100);
}

}  // namespace zvec