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

// Manifest durability tests (process crash semantics):
//
// * T1: a damaged newest manifest must never make open delete segment data;
//   it must fail closed or fall back to a complete previous generation;
// * T2: a disk-full error while flush writes the manifest must be reported
//   and leave the previous state recoverable.
//
// Faults come from storage_shim.c, preloaded into storage_fault_worker only.

#include <string_view>
#include "db/index/common/manifest_codec.h"
#include "durability_harness.h"

namespace zvec::checkpoint_test {
namespace {

// ---------------------------------------------------------------------------
// T1: damaged newest manifest on reopen
// ---------------------------------------------------------------------------

constexpr int kManifestDocs = 96;

std::vector<uint64_t> ManifestIds(const fs::path &collection) {
  std::vector<uint64_t> ids;
  for (const auto &entry : fs::directory_iterator(collection)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("manifest.", 0) != 0) continue;
    const auto suffix = name.substr(9);
    if (suffix.empty() ||
        suffix.find_first_not_of("0123456789") != std::string::npos)
      continue;
    ids.push_back(std::stoull(suffix));
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

fs::path ManifestPath(const fs::path &collection, uint64_t id) {
  return collection / ("manifest." + std::to_string(id));
}

std::map<std::string, std::set<std::string>> SegmentDirectories(
    const fs::path &collection) {
  std::map<std::string, std::set<std::string>> segments;
  for (const auto &entry : fs::directory_iterator(collection)) {
    const auto name = entry.path().filename().string();
    if (!entry.is_directory() ||
        name.find_first_not_of("0123456789") != std::string::npos)
      continue;
    auto &files = segments[name];
    for (const auto &file : fs::directory_iterator(entry.path()))
      files.insert(file.path().filename().string());
  }
  return segments;
}

class ManifestDurabilityTest : public DurabilityTestBase {
 protected:
  // Three segments: two sealed through create_iterator and one flushed
  // writing segment, without deletes (the initial delete snapshot is kept).
  // `older` receives the manifest and WAL files as they were right before
  // the final flush: together with the flushed collection they form the
  // state of a crash after the new manifest was written and before the
  // previous generation was retired.
  void CreatePristine(const fs::path &path, const fs::path &older) {
    auto created =
        Collection::CreateAndOpen(path.string(), MakeSchema(false), {});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = created.value();
    constexpr int kPerSegment = kManifestDocs / 3;
    for (int begin = 0; begin < kManifestDocs; begin += kPerSegment) {
      for (int id = begin; id < begin + kPerSegment; ++id) {
        std::vector<Doc> docs{MakeDoc(id, 0)};
        ASSERT_NO_THROW(CheckWrite(collection->insert(docs), 1));
      }
      if (begin + kPerSegment < kManifestDocs) {
        auto iterator = collection->create_iterator();
        ASSERT_TRUE(iterator.has_value()) << iterator.error().message();
      }
    }
    fs::create_directories(older);
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
      const auto name = entry.path().filename().string();
      if (!entry.is_regular_file() || (name.rfind("manifest.", 0) != 0 &&
                                       entry.path().extension() != ".wal"))
        continue;
      const auto relative = fs::relative(entry.path(), path);
      fs::create_directories((older / relative).parent_path());
      fs::copy_file(entry.path(), older / relative);
    }
    auto status = collection->flush();
    ASSERT_TRUE(status.ok()) << status.message();
    status = collection->close();
    ASSERT_TRUE(status.ok()) << status.message();
  }

  WorkerResult Check(const fs::path &collection, const std::string &name) {
    return RunWorker(
        {collection.string(), "check", "plain", std::to_string(kManifestDocs)},
        {}, directory_ / (name + ".log"));
  }

  // Cuts the current decoder accepts are the dangerous ones: they silently
  // drop trailing fields such as the writing segment. Always include empty.
  static std::set<size_t> Cuts(const std::string &manifest) {
    std::set<size_t> cuts{0, 1, manifest.size() / 2, manifest.size() - 1};
    std::vector<size_t> accepted;
    for (size_t length = 1; length < manifest.size(); ++length) {
      ManifestData data;
      if (ManifestCodec::Decode(std::string_view(manifest.data(), length),
                                &data)
              .ok())
        accepted.push_back(length);
    }
    const size_t samples = std::min<size_t>(6, accepted.size());
    for (size_t i = 0; i < samples; ++i)
      cuts.insert(accepted[i * accepted.size() / samples]);
    return cuts;
  }

  void ExpectSegmentsIntact(
      const std::map<std::string, std::set<std::string>> &segments,
      const fs::path &collection, const WorkerResult &opened) {
    const auto after = SegmentDirectories(collection);
    for (const auto &[segment, files] : segments) {
      auto found = after.find(segment);
      ASSERT_NE(found, after.end())
          << "Open deleted segment directory " << segment << "\n"
          << opened.describe();
      for (const auto &file : files) {
        EXPECT_TRUE(found->second.count(file))
            << "Open deleted " << segment << "/" << file << "\n"
            << opened.describe();
      }
    }
  }
};

// Only the damaged manifest exists: open must fail closed (or recover the
// complete state) and must not remove anything.
TEST_F(ManifestDurabilityTest, DamagedOnlyManifestFailsClosedWithoutDeleting) {
  const auto pristine = directory_ / "pristine";
  ASSERT_NO_FATAL_FAILURE(CreatePristine(pristine, directory_ / "older"));
  const auto ids = ManifestIds(pristine);
  ASSERT_EQ(ids.size(), 1u);
  const auto manifest_name = ManifestPath(pristine, ids.back()).filename();
  const std::string manifest = ReadFile(pristine / manifest_name);
  const auto segments = SegmentDirectories(pristine);
  ASSERT_GE(segments.size(), 3u);
  auto baseline = Check(pristine, "pristine");
  ASSERT_TRUE(baseline.exited(0)) << baseline.describe();

  // Each cut is independent: a failing cut does not hide the others.
  auto run_cut = [&](size_t length) {
    SCOPED_TRACE("manifest truncated to " + std::to_string(length) + " of " +
                 std::to_string(manifest.size()) + " bytes");
    const auto copy = directory_ / ("cut-" + std::to_string(length));
    fs::copy(pristine, copy, fs::copy_options::recursive);
    fs::resize_file(copy / manifest_name, length);

    auto opened = Check(copy, "cut-" + std::to_string(length));
    ASSERT_NO_FATAL_FAILURE(ExpectSegmentsIntact(segments, copy, opened));
    if (opened.exited(0)) return;  // recovered the complete state
    ASSERT_TRUE(opened.exited(3))
        << "Open must fail closed or recover every document\n"
        << opened.describe();
    // Failing closed must leave the data recoverable once the manifest is
    // repaired.
    ASSERT_NO_FATAL_FAILURE(WriteFile(copy / manifest_name, manifest));
    auto repaired = Check(copy, "repaired-" + std::to_string(length));
    ASSERT_TRUE(repaired.exited(0)) << repaired.describe();
  };
  for (size_t length : Cuts(manifest)) run_cut(length);
}

// A crash after the new manifest was written and before the previous
// generation (its manifest and WAL) was retired, then the new manifest is
// damaged: open must fall back to the complete previous generation.
TEST_F(ManifestDurabilityTest, DamagedNewestManifestFallsBackToPrevious) {
  const auto pristine = directory_ / "pristine";
  const auto older = directory_ / "older";
  ASSERT_NO_FATAL_FAILURE(CreatePristine(pristine, older));
  const auto newest = ManifestIds(pristine);
  ASSERT_EQ(newest.size(), 1u);
  const auto previous = ManifestIds(older);
  ASSERT_EQ(previous.size(), 1u);
  ASSERT_LT(previous.back(), newest.back());
  const auto manifest_name = ManifestPath(pristine, newest.back()).filename();
  const std::string manifest = ReadFile(pristine / manifest_name);
  const auto segments = SegmentDirectories(pristine);

  // Positive control: without the newest manifest, the previous generation
  // (manifest and WAL) must recover every document on its own. Otherwise the
  // fallback cases below could not tell a missing fallback from a bad setup.
  {
    const auto control = directory_ / "control";
    fs::copy(pristine, control, fs::copy_options::recursive);
    fs::copy(older, control,
             fs::copy_options::recursive | fs::copy_options::skip_existing);
    fs::remove(control / manifest_name);
    auto opened = Check(control, "control");
    ASSERT_TRUE(opened.exited(0))
        << "The previous generation alone does not recover\n"
        << opened.describe();
  }

  auto run_cut = [&](size_t length) {
    SCOPED_TRACE("newest manifest truncated to " + std::to_string(length) +
                 " of " + std::to_string(manifest.size()) + " bytes");
    const auto copy = directory_ / ("fallback-" + std::to_string(length));
    fs::copy(pristine, copy, fs::copy_options::recursive);
    fs::copy(older, copy,
             fs::copy_options::recursive | fs::copy_options::skip_existing);
    fs::resize_file(copy / manifest_name, length);

    auto opened = Check(copy, "fallback-" + std::to_string(length));
    ASSERT_NO_FATAL_FAILURE(ExpectSegmentsIntact(segments, copy, opened));
    EXPECT_TRUE(opened.exited(0))
        << "Open must fall back to the previous complete manifest\n"
        << opened.describe();
  };
  for (size_t length : Cuts(manifest)) run_cut(length);
}

// ---------------------------------------------------------------------------
// T2: ENOSPC while flush writes the manifest
// ---------------------------------------------------------------------------

class ManifestEnospcTest : public DurabilityTestBase,
                           public ::testing::WithParamInterface<bool> {};

TEST_P(ManifestEnospcTest, FlushReportsErrorAndKeepsPreviousState) {
  const char *kind = GetParam() ? "fts" : "plain";
  const auto collection = directory_ / "collection";
  auto prepared = Prepare(collection, kind);
  ASSERT_TRUE(prepared.exited(0)) << prepared.describe();

  const auto arm = directory_ / "armed";
  const auto audit = directory_ / "audit.log";
  auto flushed = RunShimmed({collection.string(), "flush", kind, "64"},
                            {{"ZVEC_SHIM_ROOT", collection.string()},
                             {"ZVEC_FAULT_ARM", arm.string()},
                             {"ZVEC_FAULT_AUDIT", audit.string()},
                             {"ZVEC_FAULT_PATH", "/manifest."},
                             {"ZVEC_FAULT_ERRNO", std::to_string(ENOSPC)}},
                            "flush", [&] { WriteFile(arm, "armed"); });
  // The injected fault must actually have hit a manifest write.
  WorkerResult audit_lines;
  audit_lines.output = ReadFile(audit);
  ASSERT_TRUE(audit_lines.has_line_prefix(
      "INJECT write errno=" + std::to_string(ENOSPC) + " "))
      << "No manifest write was faulted\n"
      << flushed.describe();
  EXPECT_TRUE(flushed.exited(2) && flushed.has_line_prefix("STORAGE_ERROR: "))
      << "flush() must report the failed manifest write\n"
      << flushed.describe();
  fs::remove(arm);

  // All 64 extra inserts were acknowledged before the failed flush and the
  // process only exited: the previous state plus the WAL must be intact.
  auto verified = RunWorker({collection.string(), "verify", kind, "128"}, {},
                            directory_ / "verify.log");
  EXPECT_TRUE(verified.exited(0)) << verified.describe();
}

INSTANTIATE_TEST_SUITE_P(IndexTypes, ManifestEnospcTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &info) {
                           return info.param ? "Fts" : "Plain";
                         });

}  // namespace
}  // namespace zvec::checkpoint_test
