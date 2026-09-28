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

// Simulated power loss. Only data made durable by fsync-family calls
// survives; see PowerLossModel. storage_shim.c records the barriers inside
// storage_fault_worker.
//
// * T4: once flush() has returned, every earlier write survives, including
//   across later block rotation, segment rollover and optimize;
// * T6: after a failed WAL fsync, any later durability acknowledgement must
//   be truthful: every row it acknowledges survives.

#include <iterator>
#include <tuple>
#include "durability_harness.h"

namespace zvec::checkpoint_test {
namespace {

// PowerLossModel builds crash images from three inputs:
//   * the baseline: the prepared collection after a clean close, treated as
//     fully durable (as the block-level harness does with its baseline image);
//   * the shim's ordered record of barriers: full file contents at each
//     fsync/fdatasync/F_FULLFSYNC (which also write back pages dirtied through
//     mappings), the page-rounded file range of each msync(MS_SYNC),
//     directory entries at each directory fsync, and injected fsync failures
//     with the writes and truncates that follow them;
//   * the final namespace after the worker exits without closing.
// File data is its baseline contents (else empty) updated by the recorded
// barriers in order. An injected fsync failure models Linux dropping the
// dirty pages on a writeback error: each byte below the file size at the
// failure keeps its previously durable value (zero where there was none)
// until a later write covers it and a later barrier succeeds. Writes through
// a mapping are not tracked, so after a failure they do not heal lost bytes.
// Directory entries not covered by a directory fsync may or may not have
// persisted, so images are built with those namespace changes all applied,
// none applied, and, for every change to a manifest or WAL entry, that change
// alone applied and alone reverted. A rename (same inode removed and added)
// is one atomic change.
//
// Inodes identify objects. The shim logs an inode only while it holds a
// verified descriptor on it, and the model holds one on every baseline inode,
// so an inode number cannot be reused by a new file during the run.

struct DirEntry {
  char type;
  uint64_t inode;
  std::string name;
  bool operator<(const DirEntry &other) const {
    return std::tie(name, inode, type) <
           std::tie(other.name, other.inode, other.type);
  }
  bool operator==(const DirEntry &other) const {
    return name == other.name && inode == other.inode && type == other.type;
  }
};

using Listing = std::vector<DirEntry>;

struct NamespaceChange {
  std::string label;
  // (directory inode, entry, true = added / false = removed)
  std::vector<std::tuple<uint64_t, DirEntry, bool>> operations;
};

struct FileEvent {
  // 'F' full contents, 'R' synced range, 'E' failed fsync, and after one:
  // 'W' write, 'T' truncate.
  char kind;
  uint64_t offset = 0;  // 'R', 'W'
  uint64_t size = 0;    // 'E', 'R', 'T': file size; 'W': length
  fs::path data;        // 'F', 'R'
};

class PowerLossModel {
 public:
  PowerLossModel(fs::path collection, fs::path store)
      : collection_(std::move(collection)), store_(std::move(store)) {}

  ~PowerLossModel() {
    for (int fd : pins_) close(fd);
  }

  void capture_baseline() {
    fs::create_directories(store_ / "baseline");
    root_inode_ = inode(collection_);
    capture(collection_, &baseline_dirs_, true);
  }

  void load_after(const fs::path &shadow) {
    std::ifstream log(shadow / "log");
    std::string line;
    while (std::getline(log, line)) {
      std::istringstream fields(line);
      char kind = 0;
      uint64_t file_inode = 0;
      fields >> kind >> file_inode;
      FileEvent event{kind};
      long sequence = 0;
      if (kind == 'F') {
        fields >> sequence;
      } else if (kind == 'R') {
        fields >> event.offset >> event.size >> sequence;
      } else if (kind == 'E' || kind == 'T') {
        fields >> event.size;
      } else if (kind == 'W') {
        fields >> event.offset >> event.size;
      } else if (kind == 'D') {
        fields >> sequence;
        synced_dirs_[file_inode] =
            ReadEntries(shadow / std::to_string(sequence));
        continue;
      }
      ASSERT_TRUE(fields && kind != 0) << "Malformed shim record: " << line;
      if (kind == 'F' || kind == 'R')
        event.data = shadow / std::to_string(sequence);
      file_events_[file_inode].push_back(event);
      if (kind == 'E') ++failed_syncs_;
    }
    capture(collection_, &current_dirs_, false);
  }

  std::string summary() const {
    return std::to_string(file_events_.size()) + " synced files, " +
           std::to_string(synced_dirs_.size()) + " synced directories, " +
           std::to_string(failed_syncs_) + " failed fsyncs";
  }

  Listing synced_listing(uint64_t dir) const {
    auto synced = synced_dirs_.find(dir);
    if (synced != synced_dirs_.end()) return synced->second;
    auto baseline = baseline_dirs_.find(dir);
    if (baseline != baseline_dirs_.end()) return baseline->second;
    return {};  // new directory whose entries were never synced
  }

  Listing current_listing(uint64_t dir) const {
    auto current = current_dirs_.find(dir);
    return current != current_dirs_.end() ? current->second
                                          : synced_listing(dir);
  }

  // Namespace changes not covered by a directory fsync.
  std::vector<NamespaceChange> unsynced_changes() const {
    std::vector<std::tuple<uint64_t, DirEntry, bool>> operations;
    std::set<uint64_t> dirs;
    for (const auto &[dir, entries] : current_dirs_) dirs.insert(dir);
    for (const auto &[dir, entries] : synced_dirs_) dirs.insert(dir);
    for (const auto &[dir, entries] : baseline_dirs_) dirs.insert(dir);
    for (uint64_t dir : dirs) {
      auto before = synced_listing(dir), after = current_listing(dir);
      std::sort(before.begin(), before.end());
      std::sort(after.begin(), after.end());
      Listing added, removed;
      std::set_difference(after.begin(), after.end(), before.begin(),
                          before.end(), std::back_inserter(added));
      std::set_difference(before.begin(), before.end(), after.begin(),
                          after.end(), std::back_inserter(removed));
      for (auto &entry : added) operations.emplace_back(dir, entry, true);
      for (auto &entry : removed) operations.emplace_back(dir, entry, false);
    }
    // Group by inode so a rename is applied or reverted as a whole.
    std::map<uint64_t, NamespaceChange> grouped;
    for (auto &operation : operations) {
      auto &change = grouped[std::get<1>(operation).inode];
      change.label += (change.label.empty() ? "" : "+") +
                      std::string(std::get<2>(operation) ? "add:" : "remove:") +
                      std::get<1>(operation).name;
      change.operations.push_back(operation);
    }
    std::vector<NamespaceChange> changes;
    for (auto &[unused, change] : grouped) changes.push_back(change);
    return changes;
  }

  using ListingFn = std::function<Listing(uint64_t)>;

  void materialize(const ListingFn &listing, const fs::path &target) const {
    fs::create_directories(target);
    build(root_inode_, listing, target);
  }

 private:
  static uint64_t inode(const fs::path &path) {
    struct stat st;
    return lstat(path.c_str(), &st) == 0 ? st.st_ino : 0;
  }

  static Listing ReadEntries(const fs::path &path) {
    Listing entries;
    std::ifstream stream(path);
    std::string line;
    while (std::getline(stream, line)) {
      std::istringstream fields(line);
      DirEntry entry;
      fields >> entry.type >> entry.inode;
      fields.get();
      std::getline(fields, entry.name);
      entries.push_back(entry);
    }
    return entries;
  }

  void capture(const fs::path &directory, std::map<uint64_t, Listing> *listings,
               bool baseline) {
    if (baseline) {
      // Keep baseline inodes referenced so the worker cannot reuse them.
      const int fd = open(directory.c_str(), O_RDONLY);
      ASSERT_GE(fd, 0) << directory;
      pins_.push_back(fd);
    }
    auto &entries = (*listings)[inode(directory)];
    for (const auto &entry : fs::directory_iterator(directory)) {
      struct stat child;
      ASSERT_EQ(lstat(entry.path().c_str(), &child), 0);
      const bool is_dir = S_ISDIR(child.st_mode);
      entries.push_back({is_dir ? 'd' : 'f', uint64_t(child.st_ino),
                         entry.path().filename().string()});
      if (is_dir) {
        capture(entry.path(), listings, baseline);
      } else if (baseline) {
        const int fd = open(entry.path().c_str(), O_RDONLY);
        ASSERT_GE(fd, 0) << entry.path();
        pins_.push_back(fd);
        const auto copy = store_ / "baseline" / std::to_string(child.st_ino);
        fs::copy_file(entry.path(), copy, fs::copy_options::overwrite_existing);
        baseline_files_[child.st_ino] = copy;
      }
    }
  }

  // Replays the barriers of one inode over its baseline contents. After an
  // injected fsync failure, the bytes below the file size at the failure
  // were dirty or already durable; the failure drops the dirty ones, so each
  // of those bytes keeps its previously durable value (zero if none) until a
  // later write covers it and a later barrier succeeds.
  std::string durable_contents(uint64_t file_inode) const {
    std::string contents;
    auto baseline = baseline_files_.find(file_inode);
    if (baseline != baseline_files_.end())
      contents = ReadFile(baseline->second);
    auto events = file_events_.find(file_inode);
    if (events == file_events_.end()) return contents;
    std::string frozen;           // value kept by each lost byte
    std::vector<char> lost;       // byte dropped by a failed fsync
    std::vector<char> rewritten;  // byte written since the last barrier
    auto mark = [](std::vector<char> &bits, uint64_t from, uint64_t to) {
      if (bits.size() < to) bits.resize(to, 0);
      std::fill(bits.begin() + from, bits.begin() + to, 1);
    };
    // Applies synced bytes [offset, offset + data.size()) to `contents`.
    auto apply = [&](uint64_t offset, const std::string &data) {
      if (contents.size() < offset + data.size())
        contents.resize(offset + data.size(), '\0');
      for (uint64_t i = 0; i < data.size(); ++i) {
        const uint64_t at = offset + i;
        const bool is_lost = at < lost.size() && lost[at];
        const bool is_rewritten = at < rewritten.size() && rewritten[at];
        if (is_lost && !is_rewritten) {
          contents[at] = frozen[at];
          continue;
        }
        contents[at] = data[i];
        if (is_lost) lost[at] = 0;
        if (is_rewritten) rewritten[at] = 0;
      }
    };
    for (const auto &event : events->second) {
      switch (event.kind) {
        case 'E': {
          if (frozen.size() < event.size) frozen.resize(event.size, '\0');
          for (uint64_t i = 0; i < event.size; ++i) {
            if (i < lost.size() && lost[i]) continue;
            frozen[i] = i < contents.size() ? contents[i] : '\0';
          }
          mark(lost, 0, event.size);
          rewritten.assign(rewritten.size(), 0);  // those pages were dropped
          break;
        }
        case 'W':
          mark(rewritten, event.offset, event.offset + event.size);
          break;
        case 'T':
          if (lost.size() > event.size) lost.resize(event.size);
          if (rewritten.size() > event.size) rewritten.resize(event.size);
          break;
        case 'F': {
          const std::string synced = ReadFile(event.data);
          contents.resize(synced.size());
          apply(0, synced);
          rewritten.assign(rewritten.size(), 0);
          break;
        }
        case 'R': {
          // A range sync persists the file size too (grow or shrink), then
          // the synced bytes within it.
          std::string range = ReadFile(event.data);
          if (event.offset >= event.size) {
            range.clear();
          } else if (event.offset + range.size() > event.size) {
            range.resize(event.size - event.offset);
          }
          contents.resize(event.size, '\0');
          if (!range.empty()) apply(event.offset, range);
          break;
        }
      }
    }
    return contents;
  }

  void build(uint64_t dir, const ListingFn &listing,
             const fs::path &target) const {
    for (const auto &entry : listing(dir)) {
      const auto destination = target / entry.name;
      if (entry.type == 'd') {
        fs::create_directory(destination);
        build(entry.inode, listing, destination);
      } else {
        // Created and never synced: empty.
        WriteFile(destination, durable_contents(entry.inode));
      }
    }
  }

  fs::path collection_;
  fs::path store_;
  uint64_t root_inode_ = 0;
  std::vector<int> pins_;
  size_t failed_syncs_ = 0;
  std::map<uint64_t, fs::path> baseline_files_;
  std::map<uint64_t, Listing> baseline_dirs_;
  std::map<uint64_t, Listing> current_dirs_;
  std::map<uint64_t, std::vector<FileEvent>> file_events_;
  std::map<uint64_t, Listing> synced_dirs_;
};

Listing Apply(Listing entries, const NamespaceChange &change, uint64_t dir,
              bool forward) {
  for (const auto &[change_dir, entry, added] : change.operations) {
    if (change_dir != dir) continue;
    if (added == forward) {
      entries.push_back(entry);
    } else {
      entries.erase(std::remove(entries.begin(), entries.end(), entry),
                    entries.end());
    }
  }
  return entries;
}

// Model checks with hand-written shim records; no worker involved.
class PowerLossModelTest : public DurabilityTestBase {
 protected:
  // One baseline file `f` of `size` bytes of 'b'; returns its inode.
  uint64_t MakeCollection(size_t size) {
    collection_ = directory_ / "collection";
    fs::create_directories(collection_);
    WriteFile(collection_ / "f", std::string(size, 'b'));
    struct stat st;
    EXPECT_EQ(stat((collection_ / "f").c_str(), &st), 0);
    return st.st_ino;
  }

  // Replays `records` (with `payloads[i]` stored as shadow file i + 1) and
  // returns the rebuilt contents of `f`.
  std::string Rebuild(const std::vector<std::string> &records,
                      const std::vector<std::string> &payloads) {
    PowerLossModel model(collection_, directory_ / "model");
    model.capture_baseline();
    const auto shadow = directory_ / "shadow";
    fs::create_directories(shadow);
    std::string log;
    for (const auto &record : records) log += record + "\n";
    WriteFile(shadow / "log", log);
    for (size_t i = 0; i < payloads.size(); ++i)
      WriteFile(shadow / std::to_string(i + 1), payloads[i]);
    model.load_after(shadow);
    const auto image = directory_ / ("image-" + std::to_string(images_++));
    model.materialize([&](uint64_t dir) { return model.current_listing(dir); },
                      image);
    return ReadFile(image / "f");
  }

  fs::path collection_;
  int images_ = 0;
};

TEST_F(PowerLossModelTest, RangeSyncPastTruncatedEndKeepsRecordedSize) {
  const auto inode = std::to_string(MakeCollection(12288));
  // The synced page is [8192, 12288), but the file was truncated to 4096
  // before msync: nothing in the range exists, and the size shrinks.
  for (const char *offset : {"4096", "8192"}) {
    SCOPED_TRACE(offset);
    const auto contents =
        Rebuild({"R " + inode + " " + offset + " 4096 1"}, {""});
    EXPECT_EQ(contents, std::string(4096, 'b'));
  }
}

TEST_F(PowerLossModelTest, RangeSyncSetsSizeThenBytes) {
  const auto inode = std::to_string(MakeCollection(4096));
  // Growing sparse file: the size is persisted and the synced page applied.
  const auto contents =
      Rebuild({"R " + inode + " 0 8192 1"}, {std::string(4096, 'n')});
  EXPECT_EQ(contents, std::string(4096, 'n') + std::string(4096, '\0'));
  // Range clipped to a smaller recorded size.
  const auto clipped =
      Rebuild({"R " + inode + " 0 100 1"}, {std::string(4096, 'n')});
  EXPECT_EQ(clipped, std::string(100, 'n'));
}

TEST_F(PowerLossModelTest, FailedFsyncDropsBytesUntilRewrittenAndSynced) {
  const auto inode = std::to_string(MakeCollection(4));
  // Durable "bbbb"; the app appended "xxxx", its fsync failed, then it
  // rewrote bytes 6-7 and synced "bbbbxxyy". Bytes 4-5 stay lost (zero).
  const auto contents =
      Rebuild({"E " + inode + " 8", "W " + inode + " 6 2", "F " + inode + " 1"},
              {"bbbbxxyy"});
  EXPECT_EQ(contents, std::string("bbbb\0\0yy", 8));
}

class PowerLossTestBase : public DurabilityTestBase {
 protected:
  // Builds every crash image and requires rows [64, minimum) in each.
  void VerifyImages(const PowerLossModel &model, const char *kind, int minimum,
                    const std::string &context) {
    std::vector<std::pair<std::string, PowerLossModel::ListingFn>> images{
        {"namespace-applied",
         [&](uint64_t dir) { return model.current_listing(dir); }},
        {"namespace-not-applied",
         [&](uint64_t dir) { return model.synced_listing(dir); }}};
    const auto changes = model.unsynced_changes();
    for (const auto &change : changes) {
      if (change.label.find("manifest.") == std::string::npos &&
          change.label.find(".wal") == std::string::npos)
        continue;
      images.emplace_back("only-" + change.label, [&, change](uint64_t dir) {
        return Apply(model.synced_listing(dir), change, dir, true);
      });
      images.emplace_back("all-but-" + change.label, [&, change](uint64_t dir) {
        return Apply(model.current_listing(dir), change, dir, false);
      });
    }
    int index = 0;
    for (const auto &[name, listing] : images) {
      SCOPED_TRACE(name);
      const auto image = directory_ / ("image-" + std::to_string(index++));
      ASSERT_NO_FATAL_FAILURE(model.materialize(listing, image / "collection"));
      auto verified = RunWorker({(image / "collection").string(), "verify",
                                 kind, std::to_string(minimum)},
                                {}, image / "verify.log");
      EXPECT_TRUE(verified.exited(0))
          << context << ": acknowledged data lost after simulated power loss, "
          << "image " << name << " (" << model.summary() << ", "
          << changes.size() << " unsynced namespace changes)\n"
          << verified.describe();
    }
  }
};

struct PowerLossCase {
  std::string operation;
  bool fts;
  // Rows [64, minimum) were acknowledged by a flush() that returned OK.
  int minimum;
};

void PrintTo(const PowerLossCase &value, std::ostream *out) {
  *out << value.operation << (value.fts ? "/fts" : "/plain")
       << " minimum=" << value.minimum;
}

class SimulatedPowerLossTest
    : public PowerLossTestBase,
      public ::testing::WithParamInterface<PowerLossCase> {};

TEST_P(SimulatedPowerLossTest, FlushedWritesSurvive) {
  const auto &param = GetParam();
  const char *kind = param.fts ? "fts" : "plain";
  const auto collection = directory_ / "collection";
  auto prepared = Prepare(collection, kind);
  ASSERT_TRUE(prepared.exited(0)) << prepared.describe();

  PowerLossModel model(collection, directory_ / "model");
  ASSERT_NO_FATAL_FAILURE(model.capture_baseline());
  const auto shadow = directory_ / "shadow";
  fs::create_directories(shadow);
  auto operation =
      RunShimmed({collection.string(), param.operation, kind, "64"},
                 {{"ZVEC_SHIM_ROOT", collection.string()},
                  {"ZVEC_SYNC_SHADOW", shadow.string()}},
                 "operation");
  ASSERT_TRUE(operation.exited(0) && operation.has_line("DURABLE"))
      << operation.describe();
  ASSERT_NO_FATAL_FAILURE(model.load_after(shadow));
  VerifyImages(model, kind, param.minimum, "flush() returned OK");
}

INSTANTIATE_TEST_SUITE_P(
    AfterFlush, SimulatedPowerLossTest,
    ::testing::Values(
        // T4a: the operation ends with flush(); all 128 rows are required.
        PowerLossCase{"insert", false, 128}, PowerLossCase{"flush", false, 128},
        PowerLossCase{"flush", true, 128}, PowerLossCase{"rotate", false, 128},
        PowerLossCase{"optimize", false, 128},
        PowerLossCase{"optimize", true, 128},
        PowerLossCase{"create_index", false, 128},
        PowerLossCase{"drop_index", false, 128},
        // T4b: rows 64-95 are flushed, then later writes rotate blocks, roll
        // the segment over, or optimize; the flushed rows must survive.
        PowerLossCase{"flushed_rotate", false, 96},
        PowerLossCase{"flushed_seal", false, 96},
        PowerLossCase{"flushed_optimize", false, 96},
        PowerLossCase{"flushed_optimize", true, 96}),
    [](const ::testing::TestParamInfo<PowerLossCase> &info) {
      std::string name =
          info.param.operation + (info.param.fts ? "Fts" : "Plain");
      name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
      return name;
    });

// ---------------------------------------------------------------------------
// T6: fsync failure
// ---------------------------------------------------------------------------

// One WAL fsync fails during flush(). The worker then keeps writing and
// flushes again. The collection may refuse (fail-stop) or recover, but every
// durability acknowledgement it does give must hold: if the later flush()
// returns OK, all 128 rows must survive a power loss, including the rows whose
// WAL pages the failed fsync dropped.
TEST_F(PowerLossTestBase, AcknowledgementAfterFailedWalFsyncIsTruthful) {
  const auto collection = directory_ / "collection";
  auto prepared = Prepare(collection, "plain");
  ASSERT_TRUE(prepared.exited(0)) << prepared.describe();

  PowerLossModel model(collection, directory_ / "model");
  ASSERT_NO_FATAL_FAILURE(model.capture_baseline());
  const auto shadow = directory_ / "shadow";
  fs::create_directories(shadow);
  const auto arm = directory_ / "armed";
  const auto audit = directory_ / "audit.log";
  auto result = RunShimmed({collection.string(), "fsync_fail", "plain", "64"},
                           {{"ZVEC_SHIM_ROOT", collection.string()},
                            {"ZVEC_SYNC_SHADOW", shadow.string()},
                            {"ZVEC_FAULT_ARM", arm.string()},
                            {"ZVEC_FAULT_AUDIT", audit.string()},
                            {"ZVEC_FAULT_OPERATION", "sync"},
                            {"ZVEC_FAULT_PATH", ".wal"},
                            {"ZVEC_FAULT_ONCE", "1"}},
                           "fsync_fail", [&] { WriteFile(arm, "armed"); });
  WorkerResult audit_lines;
  audit_lines.output = ReadFile(audit);
  ASSERT_TRUE(audit_lines.has_line_prefix("INJECT sync errno="))
      << "No WAL fsync was faulted\n"
      << result.describe();
  ASSERT_TRUE(result.exited(0) && (result.has_line("FIRST_FLUSH_OK") ||
                                   result.has_line("FIRST_FLUSH_FAILED")))
      << result.describe();
  ASSERT_NO_FATAL_FAILURE(model.load_after(shadow));

  // The strongest acknowledgement given decides what must survive.
  int minimum = 64;
  std::string context = "no durability acknowledgement";
  if (result.has_line("DURABLE")) {
    minimum = 128;
    context = "flush() after the failed fsync returned OK";
  } else if (result.has_line("FIRST_FLUSH_OK")) {
    minimum = 96;
    context = "flush() with the failed fsync returned OK";
  }
  VerifyImages(model, "plain", minimum, context + "\n" + result.output);
}

}  // namespace
}  // namespace zvec::checkpoint_test
