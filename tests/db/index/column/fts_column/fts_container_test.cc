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

#include "db/index/column/fts_column/fts_container.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "db/index/column/fts_column/fts_dictionary.h"
#include "db/index/column/fts_column/fts_field.h"
#include "db/index/column/fts_column/fts_file_lifetime.h"
#include "db/index/column/fts_column/fts_positions.h"
#include "db/index/column/fts_column/fts_stream.h"

namespace zvec::fts {
namespace {

TEST(FtsSpoolTest, PreservesBytesAcrossSequentialAndMixedIo) {
  auto spool = FtsSpool::Create();
  ASSERT_TRUE(spool.has_value());
  std::string expected;
  for (size_t i = 0; i < 3000; ++i) {
    const auto bytes = std::to_string(i) + '\0';
    ASSERT_TRUE((*spool)->append(bytes).ok());
    expected += bytes;
  }
  auto prefix = (*spool)->read(0, 4096);
  ASSERT_TRUE(prefix.has_value());
  EXPECT_EQ(expected.substr(0, 4096), *prefix);
  // Read-to-write at exactly the current offset still needs a stdio barrier.
  ASSERT_TRUE((*spool)->overwrite(4096, "changed").ok());
  expected.replace(4096, 7, "changed");
  auto next = (*spool)->read(4103, 33);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(expected.substr(4103, 33), *next);
  auto suffix = (*spool)->read(4136, expected.size() - 4136);
  ASSERT_TRUE(suffix.has_value());
  EXPECT_EQ(expected.substr(4136), *suffix);
  ASSERT_TRUE((*spool)->append("tail").ok());
  expected += "tail";
  auto all = (*spool)->read(0, expected.size());
  ASSERT_TRUE(all.has_value());
  EXPECT_EQ(expected, *all);
  EXPECT_EQ(expected.size(), (*spool)->size());
  EXPECT_FALSE((*spool)->overwrite(expected.size(), "outside").ok());
  auto unchanged = (*spool)->read(0, expected.size());
  ASSERT_TRUE(unchanged.has_value());
  EXPECT_EQ(expected, *unchanged);
}

class ShortWriteDumper : public core::IndexDumper {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  int append(const std::string &, size_t, size_t, uint32_t) override {
    return 0;
  }
  size_t write(const void *, size_t length) override {
    return length == 0 ? 0 : length - 1;
  }
  uint32_t magic() const override {
    return 0;
  }
};

TEST(FtsContainerWriterTest, AShortWritePermanentlyFailsTheCandidate) {
  auto writer =
      FtsContainerWriter::Create(std::make_shared<ShortWriteDumper>());
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->begin_section("field.0.terms").ok());
  EXPECT_EQ((*writer)->append("term").code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ((*writer)->append("retry").code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ((*writer)->end_section().code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ((*writer)->finish({}).code(), StatusCode::UNAVAILABLE);
}

TEST(FtsContainerWriterTest, RefusesToOverflowOuterContainerSize) {
  class FullDumper : public ShortWriteDumper {
   public:
    size_t size() const override {
      return std::numeric_limits<size_t>::max();
    }
  };
  auto writer = FtsContainerWriter::Create(std::make_shared<FullDumper>());
  ASSERT_FALSE(writer);
  EXPECT_EQ(writer.error().code(), StatusCode::RESOURCE_EXHAUSTED);
}

TEST(FtsContainerWriterTest, RejectsReservedAndEmbeddedNulSectionNames) {
  for (const auto &name :
       {std::string("IndexVersion"), std::string("field.0.terms\0extra", 19)}) {
    auto writer =
        FtsContainerWriter::Create(std::make_shared<ShortWriteDumper>());
    ASSERT_TRUE(writer);
    EXPECT_EQ((*writer)->begin_section(name).code(),
              StatusCode::INVALID_ARGUMENT);
  }
}

class FtsContainerTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(16 * 1024 * 1024));
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = "fts_container_test_" + std::to_string(stamp) + ".zfts";
    payload_.resize(700001);
    for (size_t i = 0; i < payload_.size(); ++i) {
      payload_[i] = static_cast<char>(i % 251);
    }
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(dumper->create(path_), 0);
    auto writer = FtsContainerWriter::Create(dumper, 1024);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE((*writer)->begin_section("field.0.postings").ok());
    for (size_t i = 0; i < payload_.size(); i += 333) {
      ASSERT_TRUE(
          (*writer)->append(std::string_view(payload_).substr(i, 333)).ok());
    }
    ASSERT_TRUE((*writer)->end_section().ok());
    ASSERT_TRUE((*writer)->begin_section("field.0.positions").ok());
    ASSERT_TRUE((*writer)->end_section().ok());
    format::Metadata meta;
    meta.base_id[0] = 1;
    meta.base_generation = 9;
    meta.layout_granule = 1024;
    ASSERT_TRUE((*writer)->finish(meta).ok());
    EXPECT_FALSE((*writer)->begin_section("after.finish").ok());
  }

  void TearDown() override {
    std::remove(path_.c_str());
  }

  void corrupt_section(const std::string &name) {
    auto storage = core::IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(storage, nullptr);
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path_, false));
    auto section = storage->get(name);
    ASSERT_NE(section, nullptr);
    const size_t offset = section->data_offset();
    section.reset();
    ASSERT_EQ(0, storage->close());
    std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekg(static_cast<std::streamoff>(offset));
    char byte = 0;
    file.read(&byte, 1);
    ASSERT_TRUE(file.good());
    byte ^= 1;
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(&byte, 1);
    ASSERT_TRUE(file.good());
  }

  void write_dictionary(const std::vector<std::string> &terms,
                        bool wrong_fingerprint = false, uint32_t doc = 0) {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path_));
    auto container = FtsContainerWriter::Create(dumper);
    auto dictionary = FtsDictionaryWriter::Create();
    ASSERT_TRUE(container);
    ASSERT_TRUE(dictionary);
    format::TermMetadata term;
    term.document_frequency = term.total_frequency = term.inline_tf = 1;
    term.inline_doc = doc;
    for (const auto &text : terms) {
      ASSERT_TRUE((*dictionary)->add(text, term).ok());
    }
    auto summary = (*dictionary)->finish(container->get(), 0);
    ASSERT_TRUE(summary);
    EXPECT_FALSE((*dictionary)->add("after.finish", term).ok());
    format::Metadata metadata;
    metadata.base_id[0] = 1;
    metadata.row_count = 1;
    metadata.fields.emplace_back();
    auto &field = metadata.fields[0];
    field.id[0] = 2;
    field.incarnation[0] = 3;
    field.name = "text";
    field.present_count = field.scoring_document_count = 1;
    field.total_tokens = terms.size();
    field.term_count = summary->term_count;
    field.dictionary_fingerprint = summary->fingerprint;
    if (wrong_fingerprint) {
      field.dictionary_fingerprint[0] ^= 1;
    }
    ASSERT_TRUE((*container)->finish(metadata).ok());
  }

  std::string path_;
  std::string payload_;
};

TEST_F(FtsContainerTest, RetiredFileSurvivesUntilLastRangeReleasesStorage) {
  auto queue = FtsRetirementQueue::Create();
  for (bool mmap : {false, true}) {
    const std::string generation = path_ + (mmap ? ".mmap" : ".buffer");
    ASSERT_TRUE(std::filesystem::copy_file(path_, generation));
    auto owner = queue->track(generation, false, false);
    FtsReadRange pinned;
    {
      auto opened =
          FtsContainerReader::Open(generation, mmap, 1024 * 1024, owner);
      ASSERT_TRUE(opened.has_value());
      auto range = (*opened)->read("field.0.postings", 13, 23);
      ASSERT_TRUE(range.has_value());
      pinned = std::move(*range);
      owner->retire();
      owner.reset();
    }
    EXPECT_TRUE(std::filesystem::exists(generation));
    EXPECT_EQ(payload_.substr(13, 23), pinned.view());
    pinned = FtsReadRange{};
    EXPECT_FALSE(std::filesystem::exists(generation));
    EXPECT_EQ(0u, queue->pending());
  }
}

TEST_F(FtsContainerTest, RetirementFailureIsRetriedWithoutChangingGeneration) {
  bool fail = true;
  auto queue = FtsRetirementQueue::Create([&](const std::string &) {
    return fail ? Status::InternalError("Injected occupied-file failure")
                : Status::OK();
  });
  const std::string generation = path_ + ".retired";
  ASSERT_TRUE(std::filesystem::copy_file(path_, generation));
  auto owner = queue->track(generation, false, true);
  owner->preserve();  // Committed (or uncertain): last release must not delete.
  owner.reset();
  EXPECT_TRUE(std::filesystem::exists(generation));
  EXPECT_EQ(0u, queue->pending());
  owner = queue->track(generation, false, false);
  owner->retire();
  owner.reset();
  EXPECT_TRUE(std::filesystem::exists(generation));
  EXPECT_EQ(1u, queue->pending());
  fail = false;
  EXPECT_EQ(1u, queue->retry());
  EXPECT_EQ(0u, queue->pending());
  EXPECT_FALSE(std::filesystem::exists(generation));
}

TEST_F(FtsContainerTest, BothBackendsReadCrossPageAndEmptyRanges) {
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(reader);
    EXPECT_EQ((*reader)->metadata().base_generation, 9u);
    EXPECT_TRUE((*reader)->verify_all().ok());
    for (size_t offset : {0u, 1u, 1023u, 1024u, 16381u, 699993u}) {
      const size_t length = std::min<size_t>(4098, payload_.size() - offset);
      auto range = (*reader)->read("field.0.postings", offset, length);
      ASSERT_TRUE(range);
      EXPECT_EQ(range->view(),
                std::string_view(payload_).substr(offset, length));
    }
    auto empty = (*reader)->read("field.0.positions", 0, 0);
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->view().empty());
    EXPECT_EQ((*reader)->read("missing", 0, 0).error().code(),
              StatusCode::NOT_FOUND);
    EXPECT_FALSE((*reader)->read("field.0.postings", UINT64_MAX - 2, 8));
  }
}

TEST_F(FtsContainerTest, RangeKeepsGenerationAliveAfterReaderRelease) {
  for (bool mmap : {false, true}) {
    auto opened = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(opened);
    auto reader = std::move(*opened);
    auto range = reader->read("field.0.postings", 27, 13);
    ASSERT_TRUE(range);
    reader.reset();
    EXPECT_EQ(range->view(), std::string_view(payload_).substr(27, 13));
  }
}

TEST_F(FtsContainerTest, SectionWindowsReuseNativePagesAcrossChecksumChunks) {
  auto storage = core::IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path_, false));
  const auto section = storage->get("field.0.postings");
  ASSERT_NE(section, nullptr);
  const uint64_t physical = section->data_offset();
  const uint64_t page = ailego::kVectorPageSize;
  // Find a checksum boundary whose adjacent bytes occupy the same native
  // page. Buffer reads must retain one window, including bypass-only pools;
  // mmap retains its existing checksum-chunk window boundaries.
  uint64_t boundary = 1024;
  while ((physical + boundary - 1) / page != (physical + boundary) / page) {
    boundary += 1024;
  }
  ASSERT_LT(boundary, payload_.size());
  for (size_t budget : {size_t{0}, size_t{16 * 1024 * 1024}}) {
    ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(budget));
    for (bool mmap : {false, true}) {
      auto owner = FtsContainerReader::Open(path_, mmap);
      ASSERT_TRUE(owner);
      auto reader = FtsSectionReader::Open(*owner, "field.0.postings");
      ASSERT_TRUE(reader);
      const auto before = (*owner)->read_stats().range_reads;
      for (uint64_t offset : {boundary - 1, boundary}) {
        auto bytes = (*reader)->read(offset, 1);
        ASSERT_TRUE(bytes);
        EXPECT_EQ(payload_.substr(offset, 1), *bytes);
      }
      EXPECT_EQ(before + (mmap ? 2 : 1), (*owner)->read_stats().range_reads);
      // Also exercise a native-page crossing, the maximum request, and the
      // unaligned final window without reading outside the section.
      const uint64_t crossing = 2 * page - physical % page - 1;
      for (const auto &range :
           {format::ByteRange{crossing, 2}, format::ByteRange{0, 65536},
            format::ByteRange{payload_.size() - 3, 3}}) {
        auto bytes = (*reader)->read(range.offset, range.length);
        ASSERT_TRUE(bytes);
        EXPECT_EQ(payload_.substr(range.offset, range.length), *bytes);
        if (range.length == 65536) {
          // Buffer alignment may retain more than the request limit. A hit
          // must still reject an oversized request without discarding its view.
          const auto reads = (*owner)->read_stats().range_reads;
          auto oversized = (*reader)->read(0, 65537);
          ASSERT_FALSE(oversized);
          EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, oversized.error().code());
          auto cached = (*reader)->read(1, 31);
          ASSERT_TRUE(cached);
          EXPECT_EQ(payload_.substr(1, 31), *cached);
          EXPECT_EQ(reads, (*owner)->read_stats().range_reads);
        }
      }
      EXPECT_TRUE((*reader)->read(payload_.size(), 0)->empty());
      EXPECT_FALSE((*reader)->read(payload_.size(), 1));
      EXPECT_FALSE((*reader)->read(UINT64_MAX, 1));
      // One native window fits the ordinary one-page range allowance, even
      // when it contains multiple wire-format checksum chunks.
      const size_t charge = 4 * std::max<size_t>(page, 4128);
      auto bounded = FtsContainerReader::Open(path_, mmap, charge);
      ASSERT_TRUE(bounded);
      auto window = FtsSectionReader::Open(*bounded, "field.0.postings");
      ASSERT_TRUE(window);
      auto bytes = (*window)->read(boundary, 1);
      ASSERT_TRUE(bytes);
      EXPECT_EQ(payload_.substr(boundary, 1), *bytes);
    }
  }
}

TEST_F(FtsContainerTest, BypassAndConcurrentRangeBudgetAreBounded) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(0));
  for (bool mmap : {false, true}) {
    const size_t charge = 4 * std::max<size_t>(ailego::kVectorPageSize, 4128);
    auto reader = FtsContainerReader::Open(path_, mmap, charge);
    ASSERT_TRUE(reader);
    auto first = (*reader)->read("field.0.postings", 0, 100);
    ASSERT_TRUE(first);
    EXPECT_EQ((*reader)->read("field.0.postings", 2048, 10).error().code(),
              StatusCode::RESOURCE_EXHAUSTED);
    FtsReadRange moved = std::move(*first);
    EXPECT_TRUE(first->view().empty());
    EXPECT_EQ(moved.view(), std::string_view(payload_).substr(0, 100));
    moved = FtsReadRange();
    auto next = (*reader)->read("field.0.postings", 2048, 10);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->view(), std::string_view(payload_).substr(2048, 10));
  }
}

TEST_F(FtsContainerTest, PayloadChecksumIsOnlyCheckedExplicitly) {
  corrupt_section("field.0.postings");
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(reader);
    EXPECT_TRUE((*reader)->read("field.0.postings", 0, 1));
    EXPECT_EQ(0u, (*reader)->read_stats().hashed_bytes);
    EXPECT_FALSE((*reader)->verify_all().ok());
  }
}

TEST_F(FtsContainerTest, RangeReadsSkipHashingAndPreserveOwnedLifetimes) {
  for (bool mmap : {false, true}) {
    auto opened = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(opened);
    auto reader = std::move(*opened);
    const auto initial_external =
        ailego::MemoryLimitPool::get_instance().external_used();
    auto retained = reader->read("field.0.postings", 17, 20);
    ASSERT_TRUE(retained);
    const auto before = reader->read_stats();
    EXPECT_EQ(0u, before.hashed_bytes);
    {
      auto repeated = reader->read("field.0.postings", 30, 40);
      ASSERT_TRUE(repeated);
      EXPECT_EQ(std::string_view(payload_).substr(30, 40), repeated->view());
    }
    EXPECT_EQ(before.hashed_bytes, reader->read_stats().hashed_bytes);
    for (size_t offset = 1024; offset < payload_.size(); offset += 1024) {
      auto next = reader->read("field.0.postings", offset, 1);
      ASSERT_TRUE(next);
      EXPECT_EQ(payload_[offset], next->view()[0]);
    }
    EXPECT_EQ(0u, reader->read_stats().hashed_bytes);
    // Mmap ranges pin the native mapping without separate checksum copies.
    if (mmap) {
      EXPECT_EQ(initial_external,
                ailego::MemoryLimitPool::get_instance().external_used());
    }
    reader.reset();
    EXPECT_EQ(std::string_view(payload_).substr(17, 20), retained->view());
    retained = FtsReadRange{};
    EXPECT_EQ(initial_external,
              ailego::MemoryLimitPool::get_instance().external_used());
  }
}

TEST_F(FtsContainerTest, ChecksumDirectoryPayloadIsOnlyCheckedExplicitly) {
  corrupt_section("checksums");
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(reader);
    EXPECT_TRUE((*reader)->read("field.0.postings", 0, 1));
    EXPECT_EQ(0u, (*reader)->read_stats().hashed_bytes);
    EXPECT_FALSE((*reader)->verify_all().ok());
  }
}

TEST_F(FtsContainerTest, ShortReadFailsInsteadOfReturningPartialSuccess) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(0));
  auto reader = FtsContainerReader::Open(path_, false);
  ASSERT_TRUE(reader);
  std::error_code error;
  std::filesystem::resize_file(path_, 0, error);
  ASSERT_FALSE(error);
  EXPECT_EQ((*reader)->read("field.0.postings", 0, 1).error().code(),
            StatusCode::UNAVAILABLE);
}

TEST_F(FtsContainerTest, CacheIdentityIncludesFileEvenWithSameGeneration) {
  struct TemporaryFile {
    std::string path;
    ~TemporaryFile() {
      std::remove(path.c_str());
    }
  } other{path_ + ".other"};
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(other.path));
  auto writer = FtsContainerWriter::Create(dumper);
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->begin_section("field.0.postings").ok());
  ASSERT_TRUE((*writer)->append("other base").ok());
  ASSERT_TRUE((*writer)->end_section().ok());
  format::Metadata meta;
  meta.base_id[0] = 2;
  meta.base_generation = 9;
  ASSERT_TRUE((*writer)->finish(meta).ok());
  for (bool mmap : {false, true}) {
    auto first = FtsContainerReader::Open(path_, mmap);
    auto second = FtsContainerReader::Open(other.path, mmap);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    auto a = (*first)->read("field.0.postings", 0, 10);
    auto b = (*second)->read("field.0.postings", 0, 10);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_EQ(a->view(), std::string_view(payload_).substr(0, 10));
    EXPECT_EQ(b->view(), "other base");
  }
}

TEST_F(FtsContainerTest, DictionarySupportsOrdinalSeekAndBinaryTerms) {
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path_));
  auto container = FtsContainerWriter::Create(dumper);
  auto dictionary = FtsDictionaryWriter::Create();
  ASSERT_TRUE(container);
  ASSERT_TRUE(dictionary);
  const std::vector<std::string> terms{"", std::string("a\0b", 3), "alpha",
                                       "beta", std::string(1, '\xff')};
  for (size_t i = 0; i < terms.size(); ++i) {
    format::TermMetadata term;
    term.document_frequency = term.total_frequency = term.inline_tf = 1;
    term.inline_doc = static_cast<uint32_t>(i);
    ASSERT_TRUE((*dictionary)->add(terms[i], term).ok());
  }
  auto summary = (*dictionary)->finish(container->get(), 0);
  ASSERT_TRUE(summary);
  format::Metadata metadata;
  metadata.base_id[0] = 1;
  metadata.row_count = terms.size();
  metadata.fields.emplace_back();
  metadata.fields[0].id[0] = 2;
  metadata.fields[0].incarnation[0] = 3;
  metadata.fields[0].name = "text";
  metadata.fields[0].term_count = summary->term_count;
  metadata.fields[0].dictionary_fingerprint = summary->fingerprint;
  ASSERT_TRUE((*container)->finish(metadata).ok());
  for (bool mmap : {false, true}) {
    auto opened = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(opened);
    auto reader = FtsDictionaryReader::Open(*opened, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    for (size_t i = 0; i < terms.size(); ++i) {
      auto entry = (*reader)->at(i);
      ASSERT_TRUE(entry);
      EXPECT_EQ(entry->ordinal(), i);
      EXPECT_EQ(entry->metadata().inline_doc, i);
      auto bytes = entry->read_term(0, terms[i].size());
      ASSERT_TRUE(bytes);
      EXPECT_EQ(*bytes, terms[i]);
      auto found = (*reader)->seek(terms[i]);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->ordinal(), i);
    }
    EXPECT_EQ((*reader)->seek("al").error().code(), StatusCode::NOT_FOUND);
    auto next = (*reader)->seek_ge("al");
    ASSERT_TRUE(next);
    EXPECT_EQ(next->ordinal(), 2u);
    EXPECT_EQ((*reader)->at(terms.size()).error().code(),
              StatusCode::NOT_FOUND);
    EXPECT_EQ((*reader)->seek_ge(std::string(2, '\xff')).error().code(),
              StatusCode::NOT_FOUND);
  }
}

TEST_F(FtsContainerTest, DictionaryPagesAndOversizedTermsKeepGenerationAlive) {
  std::vector<std::string> terms;
  for (size_t i = 0; i < 9000; ++i) {
    terms.push_back(std::to_string(100000 + i));
  }
  terms.push_back("z" + std::string(1024 * 1024, 'a'));
  terms.push_back(terms.back() + "b");
  write_dictionary(terms);
  for (bool mmap : {false, true}) {
    auto container = FtsContainerReader::Open(path_, mmap, 256 * 1024);
    ASSERT_TRUE(container);
    auto reader = FtsDictionaryReader::Open(*container, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    for (size_t ordinal : {0u, 63u, 64u, 8191u, 8192u, 8999u, 9000u, 9001u}) {
      auto entry = (*reader)->at(ordinal);
      ASSERT_TRUE(entry);
      EXPECT_EQ(entry->term_size(), terms[ordinal].size());
      auto found = (*reader)->seek(terms[ordinal]);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->ordinal(), ordinal);
    }
    auto between = (*reader)->seek_ge("108191x");
    ASSERT_TRUE(between);
    EXPECT_EQ(between->ordinal(), 8192u);
    auto held = (*reader)->at(9000);
    ASSERT_TRUE(held);
    reader->reset();
    container->reset();
    EXPECT_EQ(held->read_term(0, 4097).error().code(),
              StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(held->read_term(UINT64_MAX, 1).error().code(),
              StatusCode::INVALID_ARGUMENT);
    for (size_t offset : {0u, 4095u, 16383u, 1048570u}) {
      auto part = held->read_term(
          offset, std::min<size_t>(4096, terms[9000].size() - offset));
      ASSERT_TRUE(part);
      EXPECT_EQ(*part, terms[9000].substr(offset, part->size()));
    }
  }
}

TEST_F(FtsContainerTest, EmptyDictionaryAndSemanticValidation) {
  write_dictionary({});
  for (bool mmap : {false, true}) {
    auto container = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(container);
    auto reader = FtsDictionaryReader::Open(*container, 0);
    ASSERT_TRUE(reader);
    EXPECT_TRUE((*reader)->verify().ok());
    EXPECT_EQ((*reader)->at(0).error().code(), StatusCode::NOT_FOUND);
    EXPECT_EQ((*reader)->seek_ge("").error().code(), StatusCode::NOT_FOUND);
  }
  write_dictionary({"alpha"}, true);
  for (bool mmap : {false, true}) {
    auto container = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(container);
    auto reader = FtsDictionaryReader::Open(*container, 0);
    ASSERT_TRUE(reader);
    EXPECT_EQ((*reader)->verify().code(), StatusCode::INTERNAL_ERROR);
  }
  write_dictionary({"alpha"}, false, UINT32_MAX - 1);
  for (bool mmap : {false, true}) {
    auto container = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(container);
    auto reader = FtsDictionaryReader::Open(*container, 0);
    ASSERT_TRUE(reader);
    EXPECT_EQ((*reader)->at(0).error().code(), StatusCode::INTERNAL_ERROR);
  }
}

TEST(FtsDictionaryWriterTest, RejectsUnsortedAndDuplicateTermsBeforeMutation) {
  auto writer = FtsDictionaryWriter::Create();
  ASSERT_TRUE(writer);
  format::TermMetadata term;
  term.document_frequency = term.total_frequency = term.inline_tf = 1;
  const std::string oversized(8192, 'b');
  ASSERT_TRUE((*writer)->add("a", term).ok());
  ASSERT_TRUE((*writer)->add(oversized, term).ok());
  EXPECT_EQ((*writer)->add(oversized, term).code(),
            StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ((*writer)->add("b", term).code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_TRUE((*writer)->add("c", term).ok());
}

TEST(FtsDictionaryWriterTest, StreamedTermsPreserveOrderingAndReadErrors) {
  format::TermMetadata term;
  term.document_frequency = term.total_frequency = term.inline_tf = 1;
  auto source = [](char value) -> FtsTermSource {
    return [value](uint64_t, size_t count) -> Result<std::string> {
      return std::string(count, value);
    };
  };
  auto writer = FtsDictionaryWriter::Create();
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->add("b", term).ok());
  ASSERT_TRUE((*writer)->add(8192, source('b'), term).ok());
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
            (*writer)->add(8192, source('b'), term).code());
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
            (*writer)->add(8191, source('b'), term).code());
  ASSERT_TRUE((*writer)->add(8193, source('b'), term).ok());
  ASSERT_TRUE((*writer)->add(1, source('c'), term).ok());
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
            (*writer)->add(8192, source('b'), term).code());
  for (bool short_read : {false, true}) {
    auto failed = FtsDictionaryWriter::Create();
    ASSERT_TRUE(failed);
    auto status = (*failed)->add(
        8192,
        [short_read](uint64_t offset, size_t count) -> Result<std::string> {
          if (offset == 0) {
            return std::string(count, 'a');
          }
          if (short_read) {
            return std::string(count - 1, 'a');
          }
          return tl::make_unexpected(
              Status{StatusCode::UNAVAILABLE, "injected term read failure"});
        },
        term);
    EXPECT_EQ(short_read ? StatusCode::INTERNAL_ERROR : StatusCode::UNAVAILABLE,
              status.code());
    EXPECT_EQ(status.code(), (*failed)->add("z", term).code());
    EXPECT_FALSE((*failed)->finish(nullptr, 0).has_value());
  }
}

TEST_F(FtsContainerTest, DictionaryRejectsMalformedDataWithValidChecksums) {
  struct Mutation {
    const char *section;
    size_t offset;
    char value;
    StatusCode code;
  };
  const Mutation mutations[] = {
      {"term_index", 16, 2, StatusCode::INTERNAL_ERROR},
      {"term_index", 44, 1, StatusCode::NOT_SUPPORTED},
      {"term_index", 48, 127, StatusCode::INTERNAL_ERROR},
      {"terms", 0, 1, StatusCode::INTERNAL_ERROR},
      {"term_meta", 36, 127, StatusCode::INTERNAL_ERROR}};
  for (const auto &mutation : mutations) {
    write_dictionary({"alpha"});
    format::Metadata metadata;
    std::vector<std::pair<std::string, std::string>> sections;
    {
      auto source = FtsContainerReader::Open(path_, false);
      ASSERT_TRUE(source);
      metadata = (*source)->metadata();
      for (const auto &section : metadata.sections) {
        auto range = (*source)->read(section.name, 0, section.length);
        ASSERT_TRUE(range);
        sections.emplace_back(section.name, range->view());
      }
    }
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path_));
    auto writer = FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(writer);
    for (auto &section : sections) {
      if (section.first == std::string("field.0.") + mutation.section) {
        ASSERT_LT(mutation.offset, section.second.size());
        section.second[mutation.offset] = mutation.value;
      }
      ASSERT_TRUE((*writer)->begin_section(section.first).ok());
      ASSERT_TRUE((*writer)->append(section.second).ok());
      ASSERT_TRUE((*writer)->end_section().ok());
    }
    metadata.sections.clear();
    ASSERT_TRUE((*writer)->finish(metadata).ok());
    for (bool mmap : {false, true}) {
      auto container = FtsContainerReader::Open(path_, mmap);
      ASSERT_TRUE(container);
      ASSERT_TRUE((*container)->verify_all().ok());
      auto reader = FtsDictionaryReader::Open(*container, 0);
      if (!reader) {
        EXPECT_EQ(reader.error().code(), mutation.code);
      } else {
        auto entry = (*reader)->at(0);
        ASSERT_FALSE(entry);
        EXPECT_EQ(entry.error().code(), mutation.code);
        EXPECT_EQ((*reader)->verify().code(), mutation.code);
      }
    }
  }
}

TEST_F(FtsContainerTest, LongPositionCursorPreservesDuplicatesAndSeeks) {
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path_));
  auto container = FtsContainerWriter::Create(dumper);
  auto writer = FtsPositionsWriter::Create();
  ASSERT_TRUE(container);
  ASSERT_TRUE(writer);
  uint32_t ordinal = 0;
  auto range = (*writer)->append_document(
      1001, [&]() -> Result<std::optional<uint32_t>> {
        if (ordinal == 1001) {
          return std::optional<uint32_t>{};
        }
        const uint32_t value = ordinal == 1000 ? UINT32_MAX : ordinal / 2;
        ++ordinal;
        return std::optional<uint32_t>{value};
      });
  ASSERT_TRUE(range);
  ASSERT_TRUE((*writer)->finish(container->get(), 0).ok());
  format::Metadata metadata;
  metadata.base_id[0] = 1;
  metadata.fields.emplace_back();
  metadata.fields[0].id[0] = 2;
  metadata.fields[0].incarnation[0] = 3;
  metadata.fields[0].name = "text";
  ASSERT_TRUE((*container)->finish(metadata).ok());
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto header = (*file)->read("field.0.positions", 0, 32);
    ASSERT_TRUE(header);
    const unsigned char expected_header[] = {1, 0, 0, 0, 233, 3, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0,   0, 1, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0,   0, 0, 0, 0, 0};
    EXPECT_EQ(header->view(),
              std::string_view(reinterpret_cast<const char *>(expected_header),
                               sizeof(expected_header)));
    auto first_record = (*file)->read("field.0.position_index", 0, 32);
    ASSERT_TRUE(first_record);
    const unsigned char expected_record[] = {
        0,  0, 0, 0, 63, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0,
        32, 0, 0, 0, 0,  0, 0, 0, 132, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(first_record->view(),
              std::string_view(reinterpret_cast<const char *>(expected_record),
                               sizeof(expected_record)));
    auto cursor = FtsPositionCursor::Open(*file, 0, *range, 1001);
    ASSERT_TRUE(cursor);
    file->reset();
    for (uint32_t i = 0; i < 1001; ++i) {
      auto value = (*cursor)->next();
      ASSERT_TRUE(value);
      ASSERT_TRUE(value->has_value());
      EXPECT_EQ(**value, i == 1000 ? UINT32_MAX : i / 2);
    }
    auto end = (*cursor)->next();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
    file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    cursor = FtsPositionCursor::Open(*file, 0, *range, 1001);
    ASSERT_TRUE(cursor);
    for (uint32_t target : {0u, 63u, 64u, 128u, 499u, 500u, UINT32_MAX}) {
      auto value = (*cursor)->seek_ge(target);
      ASSERT_TRUE(value);
      ASSERT_TRUE(value->has_value());
      EXPECT_EQ(**value, target <= 499 ? target : UINT32_MAX);
    }
  }
}

TEST_F(FtsContainerTest, FieldRoundTripKeepsStatisticsGapsAndTailPostings) {
  format::FieldMetadata field;
  field.id[0] = 2;
  field.incarnation[0] = 3;
  field.name = "text";
  field.present_count = field.scoring_document_count = 130;
  field.total_tokens = 258;
  auto writer = FtsFieldWriter::Create(260, field);
  ASSERT_TRUE(writer);
  for (uint32_t i = 0; i < 129; ++i) {
    ASSERT_TRUE((*writer)->add_document(i * 2, true, i % 3 + 1).ok());
  }
  ASSERT_TRUE((*writer)->add_document(258, true, 0).ok());
  ASSERT_TRUE((*writer)->finish_documents().ok());
  ASSERT_TRUE((*writer)->begin_term("alpha").ok());
  for (uint32_t i = 0; i < 129; ++i) {
    uint32_t position = 0;
    const uint32_t tf = i % 3 + 1;
    ASSERT_TRUE((*writer)
                    ->add_posting(i * 2, tf,
                                  [&]() -> Result<std::optional<uint32_t>> {
                                    if (position == tf) {
                                      return std::optional<uint32_t>{};
                                    }
                                    return std::optional<uint32_t>{position++};
                                  })
                    .ok());
  }
  ASSERT_TRUE((*writer)->end_term().ok());
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path_));
  auto container = FtsContainerWriter::Create(dumper);
  ASSERT_TRUE(container);
  auto completed = (*writer)->finish(container->get(), 0);
  ASSERT_TRUE(completed);
  format::Metadata metadata;
  metadata.base_id[0] = 1;
  metadata.row_count = 260;
  metadata.fields.push_back(*completed);
  ASSERT_TRUE((*container)->finish(metadata).ok());
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    auto term = (*reader)->lookup("alpha");
    ASSERT_TRUE(term);
    EXPECT_EQ(term->metadata().document_frequency, 129u);
    EXPECT_EQ(term->metadata().total_frequency, 258u);
    auto postings = (*reader)->postings(*term);
    ASSERT_TRUE(postings);
    for (uint32_t i = 0; i < 129; ++i) {
      auto next = (*postings)->next();
      ASSERT_TRUE(next);
      ASSERT_TRUE(next->has_value());
      EXPECT_EQ((**next).doc, i * 2);
      EXPECT_EQ((**next).tf, i % 3 + 1);
      auto length = (*postings)->doc_length();
      ASSERT_TRUE(length);
      EXPECT_EQ(*length, i % 3 + 1);
      auto positions = (*postings)->positions();
      ASSERT_TRUE(positions);
      for (uint32_t p = 0; p < i % 3 + 1; ++p) {
        auto position = (*positions)->next();
        ASSERT_TRUE(position);
        ASSERT_TRUE(position->has_value());
        EXPECT_EQ(**position, p);
      }
    }
    auto end = (*postings)->next();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
    postings = (*reader)->postings(*term);
    ASSERT_TRUE(postings);
    auto advanced = (*postings)->advance(129);
    ASSERT_TRUE(advanced);
    ASSERT_TRUE(advanced->has_value());
    EXPECT_EQ((**advanced).doc, 130u);
    auto missing = (*reader)->document(1);
    auto empty = (*reader)->document(258);
    ASSERT_TRUE(missing);
    ASSERT_TRUE(empty);
    EXPECT_FALSE(missing->present);
    EXPECT_EQ(missing->length, 0u);
    EXPECT_TRUE(empty->present);
    EXPECT_EQ(empty->length, 0u);
  }
}

TEST_F(FtsContainerTest, LargeDictionaryHasPagedUpperDirectory) {
  std::vector<std::string> terms;
  for (uint32_t i = 0; i < 8193; ++i) {
    terms.push_back(std::to_string(100000 + i));
  }
  write_dictionary(terms);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    uint64_t size = 0;
    for (const auto &section : (*file)->metadata().sections) {
      if (section.name == "field.0.term_index") {
        size = section.length;
      }
    }
    // 129 leaf records and two upper fences, each a 32-byte record.
    ASSERT_EQ(size, 32u + 131u * 32u);
    auto bytes = (*file)->read("field.0.term_index", 0, size);
    ASSERT_TRUE(bytes);
    EXPECT_EQ(bytes->view().substr(32 + 129 * 32, 32),
              bytes->view().substr(32, 32));
    EXPECT_EQ(bytes->view().substr(32 + 130 * 32, 32),
              bytes->view().substr(32 + 128 * 32, 32));
    auto dictionary = FtsDictionaryReader::Open(*file, 0);
    ASSERT_TRUE(dictionary);
    ASSERT_TRUE((*dictionary)->verify().ok());
    for (uint32_t ordinal : {0u, 63u, 64u, 8191u, 8192u}) {
      auto entry = (*dictionary)->at(ordinal);
      ASSERT_TRUE(entry);
      auto found = (*dictionary)->seek(terms[ordinal]);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->ordinal(), ordinal);
    }
  }
}

TEST_F(FtsContainerTest, RejectsOuterDirectoryBudgetBeforeStorageOpen) {
  core::IndexFormat::MetaFooter footer{};
  std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(file.is_open());
  file.seekg(-static_cast<std::streamoff>(sizeof(footer)), std::ios::end);
  file.read(reinterpret_cast<char *>(&footer), sizeof(footer));
  ASSERT_TRUE(file.good());
  footer.segment_count = format::kMaxSections + 4;
  core::IndexFormat::UpdateMetaFooter(&footer, 0);
  file.seekp(-static_cast<std::streamoff>(sizeof(footer)), std::ios::end);
  file.write(reinterpret_cast<const char *>(&footer), sizeof(footer));
  file.close();
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error().code(), StatusCode::RESOURCE_EXHAUSTED);
  }
}

TEST_F(FtsContainerTest, RejectsUnknownOuterVersion) {
  core::IndexFormat::MetaHeader header{};
  std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(file.is_open());
  file.read(reinterpret_cast<char *>(&header), sizeof(header));
  ASSERT_TRUE(file.good());
  ++header.version;
  core::IndexFormat::UpdateMetaHeader(&header);
  file.seekp(0);
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));
  file.close();
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error().code(), StatusCode::NOT_SUPPORTED);
  }
}

TEST_F(FtsContainerTest, RejectsChainedOuterContainer) {
  core::IndexFormat::MetaFooter footer{};
  std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(-static_cast<std::streamoff>(sizeof(footer)), std::ios::end);
  file.read(reinterpret_cast<char *>(&footer), sizeof(footer));
  ASSERT_TRUE(file.good());
  footer.next_meta_header_offset = 64;
  core::IndexFormat::UpdateMetaFooter(&footer, 0);
  file.seekp(-static_cast<std::streamoff>(sizeof(footer)), std::ios::end);
  file.write(reinterpret_cast<const char *>(&footer), sizeof(footer));
  file.close();
  for (bool mmap : {false, true}) {
    auto reader = FtsContainerReader::Open(path_, mmap);
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error().code(), StatusCode::NOT_SUPPORTED);
  }
}

TEST_F(FtsContainerTest, DictionarySearchDescendsThreeLevels) {
  std::vector<std::string> terms;
  for (uint32_t i = 0; i < 16385; ++i) {
    std::string term(2100, 'x');
    term[0] = static_cast<char>(i >> 8);
    term[1] = static_cast<char>(i);
    terms.push_back(std::move(term));
  }
  write_dictionary(terms);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap, 256 * 1024);
    ASSERT_TRUE(file);
    uint64_t size = 0;
    for (const auto &section : (*file)->metadata().sections) {
      if (section.name == "field.0.term_index") {
        size = section.length;
      }
    }
    ASSERT_EQ(size, 32u + (16385u + 129u + 2u) * 32u);
    auto dictionary = FtsDictionaryReader::Open(*file, 0);
    ASSERT_TRUE(dictionary);
    ASSERT_TRUE((*dictionary)->verify().ok());
    for (uint32_t ordinal : {0u, 127u, 128u, 16383u, 16384u}) {
      auto entry = (*dictionary)->at(ordinal);
      ASSERT_TRUE(entry);
      auto bytes = entry->read_term(0, 2100);
      ASSERT_TRUE(bytes);
      EXPECT_EQ(*bytes, terms[ordinal]);
      auto found = (*dictionary)->seek(terms[ordinal]);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->ordinal(), ordinal);
    }
    auto between = terms[127];
    between.back() = 'y';
    auto next = (*dictionary)->seek_ge(between);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->ordinal(), 128u);
  }
}

}  // namespace
}  // namespace zvec::fts
