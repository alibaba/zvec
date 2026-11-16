// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
namespace {
constexpr size_t kUnit = 1UL << 20;
constexpr size_t kLimit = 160 * kUnit;

// Object residency and retained heap pages are the oracle. No implementation
// baseline, epoch counter or overlap estimate participates in its arithmetic.
struct LedgerModel {
  struct Object {
    size_t capacity{0}, bytes{0}, resident{0};
    bool pending{false}, heap{false};
    ProcessRssBudget::MaterializationStamp creation{};
    ProcessRssBudget::HeapProjection heap_projection{};
  };
  static LedgerModel *active;
  ProcessRssBudget budget;
  std::array<Object, 2> objects{};
  size_t permanent{0}, retained_heap{0}, terminal_owner{0}, host{64 * kUnit};
  int mode, phase;
  bool enabled{false}, rss_valid{true}, native_valid{true};
  std::map<std::string, size_t> &failures;
  std::vector<std::string> &examples;
  const std::vector<int> &schedule;

  LedgerModel(int observer, int lifecycle, bool heap, bool lifetime,
              std::map<std::string, size_t> &errors,
              std::vector<std::string> &details, const std::vector<int> &events)
      : permanent(lifetime ? 8 * kUnit : 0),
        mode(observer),
        phase(lifecycle),
        failures(errors),
        examples(details),
        schedule(events) {
    active = this;
    objects[0].capacity = 16 * kUnit;
    objects[0].heap = heap;
    objects[1].capacity = 24 * kUnit;
    if (permanent) budget.account(permanent);
    if (phase == 0 || phase == 3) {
      objects[0].bytes = objects[0].resident = objects[0].capacity;
      budget.account(objects[0].bytes);
    }
    if (phase != 1) configure();
    if (phase == 2) {
      budget.configure(0, nullptr);
      enabled = false;
    }
  }
  size_t backing() const {
    return terminal_owner + objects[0].bytes + objects[1].bytes;
  }
  size_t pending() const {
    size_t bytes = 0;
    for (const auto &object : objects)
      if (object.pending) bytes += object.capacity;
    return bytes;
  }
  size_t rss() const {
    return host + permanent + retained_heap + terminal_owner +
           objects[0].resident + objects[1].resident;
  }
  size_t owned() const {
    // Permanent image/stack allowances are outside the mapping provider.
    size_t bytes = 0;
    for (const auto &object : objects)
      if (!object.heap) bytes += object.resident;
    return bytes;
  }
  static size_t ReadRss() {
    return active->rss_valid ? active->rss() : 0;
  }
  static ProcessRssBudget::ResidentSample ReadOwned() {
    return {ReadRss(), active->owned()};
  }
  static bool ReadNative(ProcessRssBudget::NativeSample *sample) {
    if (!active->native_valid) return false;
    const size_t footprint = active->rss() + 16 * kUnit;
    *sample = {footprint, kLimit - std::min(kLimit, footprint)};
    return true;
  }
  void check(bool valid, const char *property) {
    if (valid) return;
    ++failures[property];
    if (examples.size() < 8) {
      std::ostringstream message;
      message << property << " observer=" << mode << " phase=" << phase
              << " heap=" << objects[0].heap << " permanent=" << permanent
              << " rss=" << rss() << " backing=" << backing()
              << " available=" << budget.available() << " events=";
      for (int event : schedule) message << event << ',';
      examples.push_back(message.str());
    }
  }
  void configure() {
    const auto exclusion = budget.permanent_backing() ? 0 : permanent;
    check(budget.configure(kLimit, ReadRss, mode == 2 ? ReadNative : nullptr,
                           exclusion, mode == 0 ? ReadOwned : nullptr),
          "configuration");
    enabled = true;
  }
  void proof() {
    if (mode == 2) return;
    size_t credit = owned();
    if (mode == 1) {
      const auto token = budget.begin_resident_sample();
      const auto sample = ReadOwned();
      // A cancelled OS reservation invalidates that much stale proof, even
      // though it did not become an object or change physical residency.
      check(budget.account_pending(4 * kUnit), "pending accounting");
      budget.finish_reservation(4 * kUnit, false);
      credit -= std::min(credit, 4 * kUnit + pending());
      check(budget.publish_resident_sample(token, sample), "proof publication");
    }
    const size_t upper =
        rss() + backing() - std::min(backing(), credit) + pending();
    const size_t guaranteed_room = kLimit - std::min(kLimit, upper);
    check(budget.available() >= guaranteed_room, "proof recovery");
  }
  void validate() {
    if (!enabled) return;
    // With a stable host, every live object can eventually fault all its
    // charged bytes. Freed heap pages remain resident independently.
    const size_t future =
        host + permanent + retained_heap + backing() + pending() +
        (mode == 2 ? 16 * kUnit + ProcessRssBudget::headroom(kLimit) : 0);
    const size_t impossible = kLimit - std::min(kLimit, future) + 1;
    for (bool sampled : {false, true}) {
      if (sampled) {
        budget.refresh(true);
        if (mode == 0) proof();
      }
      const bool admitted = budget.try_reserve(impossible);
      check(!admitted, "unsafe admission");
      if (admitted) budget.finish_reservation(impossible, false);
    }
  }
  void step(int event) {
    if (event == 10) {
      if (phase == 3) {
        const auto stale = budget.begin_resident_sample();
        // Also cover current-epoch unmapping during a disabled interval.
        check(budget.account_pending(4 * kUnit), "pending accounting");
        const auto temporary = budget.finish_reservation(4 * kUnit, true);
        budget.configure(0, nullptr);
        budget.release_unmapped(4 * kUnit, temporary);
        configure();
        check(!budget.publish_resident_sample(stale, ReadOwned()),
              "stale configuration");
      } else if (!enabled) {
        configure();
      }
      // Failed observations and host pressure must not make a proof a backdoor
      // around admission. Successful recovery uses the same live object state.
      const auto pre_pressure_token = budget.begin_resident_sample();
      const auto pre_pressure_sample = ReadOwned();
      rss_valid = false;
      for (int i = 0; i < 3; ++i) budget.refresh(true);
      check(budget.available() == 0, "failed reader admission");
      budget.set_external_pressure(ProcessRssBudget::PressureLevel::kCritical);
      check(!budget.try_reserve(1), "critical admission");
      budget.set_external_pressure(ProcessRssBudget::PressureLevel::kNormal);
      if (mode == 2) native_valid = false;
      rss_valid = true;
      if (mode != 2)
        check(!budget.publish_resident_sample(pre_pressure_token,
                                              pre_pressure_sample),
              "stale pressure proof");
      if (mode == 2) {
        budget.refresh(true);
        check(budget.available() == 0, "native recovery barrier");
      }
      native_valid = true;
      budget.refresh(true);
      proof();
    } else {
      auto &object = objects[event / 5];
      const int action = event % 5;
      if (action == 0) {
        check(budget.account_pending(object.capacity), "pending accounting");
        object.pending = true;
      } else if (action == 1) {
        object.creation = budget.finish_reservation(object.capacity, true);
        object.heap_projection = ProcessRssBudget::HeapProjection::Full(
            object.capacity, object.creation);
        object.bytes = object.capacity;
        object.pending = false;
      } else if (action == 2) {
        object.resident = object.bytes;
      } else {
        const size_t bytes = action == 3 ? object.capacity / 2 : object.bytes;
        if (object.heap) {
          budget.release_heap(bytes,
                              object.heap_projection.take(object.bytes, bytes));
          retained_heap += std::min(bytes, object.resident);
        } else {
          budget.release_unmapped(bytes, object.creation);
        }
        object.resident -= std::min(bytes, object.resident);
        object.bytes -= bytes;
      }
    }
    validate();
  }
  void finish() {
    budget.refresh(true);
    const size_t used =
        rss() +
        (mode == 2 ? 16 * kUnit + ProcessRssBudget::headroom(kLimit) : 0);
    check(budget.available() == kLimit - std::min(kLimit, used),
          "terminal recovery");
    budget.configure(0, nullptr);
    terminal_owner = 8 * kUnit;
    budget.account(terminal_owner);
    configure();
    const size_t startup =
        rss() + terminal_owner +
        (mode == 2 ? 16 * kUnit + ProcessRssBudget::headroom(kLimit) : 0);
    check(budget.available() == kLimit - std::min(kLimit, startup),
          "unproved startup backing");
    budget.release_heap(terminal_owner, {});
    terminal_owner = 0;
  }
};
LedgerModel *LedgerModel::active = nullptr;

TEST(ProcessRssLedgerModelTest, BoundedLifecycleSchedules) {
  std::map<std::string, size_t> failures;
  std::vector<std::string> examples;
  size_t traces = 0;
  for (int phase = 0; phase < 4; ++phase) {
    const int first = phase == 0 || phase == 3 ? 3 : 0;
    std::vector<std::vector<int>> schedules;
    std::vector<int> events;
    const auto enumerate = [&](auto &&self, int a, int b,
                               bool observed) -> void {
      if (a == 5 && b == 5 && observed) {
        schedules.push_back(events);
        return;
      }
      if (a < 5) {
        events.push_back(a);
        self(self, a + 1, b, observed);
        events.pop_back();
      }
      if (b < 5) {
        events.push_back(5 + b);
        self(self, a, b + 1, observed);
        events.pop_back();
      }
      if (!observed) {
        events.push_back(10);
        self(self, a, b, true);
        events.pop_back();
      }
    };
    enumerate(enumerate, first, 0, false);
    for (int mode = 0; mode < 3; ++mode)
      for (bool heap : {false, true})
        for (bool permanent : {false, true})
          for (const auto &schedule : schedules) {
            LedgerModel model(mode, phase, heap, permanent, failures, examples,
                              schedule);
            for (int event : schedule) model.step(event);
            model.finish();
            ++traces;
          }
  }
  std::ostringstream summary;
  summary << "traces=" << traces;
  for (const auto &[property, count] : failures)
    summary << '\n' << property << '=' << count;
  for (const auto &example : examples) summary << '\n' << example;
  RecordProperty("model_traces", std::to_string(traces));
  EXPECT_TRUE(failures.empty()) << summary.str();
}

TEST(ProcessRssLedgerModelTest, MaterializationKeepsOtherOwnersAllowance) {
  for (bool native : {false, true}) {
    std::map<std::string, size_t> failures;
    std::vector<std::string> examples;
    const std::vector<int> events;
    LedgerModel model(native ? 2 : 1, 0, true, false, failures, examples,
                      events);
    // Replace the schedule oracle with a fixed sample for the focused interval:
    // a resident adopted owner and an unrelated, still untouched owner.
    ASSERT_TRUE(model.budget.try_reserve(24 * kUnit));
    const auto lazy = model.budget.finish_reservation(24 * kUnit, true);
    model.objects[1].bytes = 24 * kUnit;
    model.budget.refresh(true);
    const size_t before = model.budget.available();
    ASSERT_TRUE(model.budget.try_reserve_materialization(8 * kUnit));
    EXPECT_EQ(before - 8 * kUnit, model.budget.available());
    const auto copy = model.budget.finish_materialization(8 * kUnit, true);
    EXPECT_EQ(before - 8 * kUnit, model.budget.available());
    model.budget.release_materialization(8 * kUnit, copy);
    EXPECT_EQ(before, model.budget.available());
    // Heap retirement keeps possibly retained physical pages charged.
    ASSERT_TRUE(model.budget.try_reserve(8 * kUnit));
    const auto heap = model.budget.finish_reservation(8 * kUnit, true);
    EXPECT_EQ(before - 8 * kUnit, model.budget.available());
    model.budget.release_heap(
        8 * kUnit, ProcessRssBudget::HeapProjection::Full(8 * kUnit, heap));
    EXPECT_EQ(before - 8 * kUnit, model.budget.available());
    const size_t remaining = model.budget.available();
    model.rss_valid = model.native_valid = false;
    EXPECT_FALSE(model.budget.try_acquire_with_pending_copy(remaining, kUnit));
    EXPECT_EQ(remaining, model.budget.available());
    model.rss_valid = model.native_valid = true;
    model.budget.release_unmapped(24 * kUnit, lazy);
    model.budget.release_heap(16 * kUnit, {});
  }
}
TEST(ProcessRssLedgerModelTest,
     ProofReanchorsSurviveHostExcursionsAndRetainedHeap) {
  for (int observer : {0, 1, 2}) {
    std::map<std::string, size_t> failures;
    std::vector<std::string> examples;
    const std::vector<int> events;
    LedgerModel model(observer, 1, true, true, failures, examples, events);
    for (int event : {0, 1, 2, 5, 6, 10}) model.step(event);
    model.host += 24 * kUnit;
    model.budget.refresh(true);
    model.validate();
    model.proof();
    model.host -= 24 * kUnit;
    model.validate();
    // Retained retired heap pages and the newly witnessed mapping are disjoint.
    for (int event : {3, 4, 7}) model.step(event);
    model.proof();
    model.retained_heap = 0;
    model.validate();
    model.proof();
    for (int event : {8, 9}) model.step(event);
    model.finish();
    {
      LedgerModel mapping(observer, 0, false, false, failures, examples,
                          events);
      for (int event : {5, 6}) mapping.step(event);
      ProcessRssBudget::MappingProjection extension;
      ASSERT_TRUE(
          mapping.budget.try_extend_mapping(8 * kUnit, 16 * kUnit, extension));
      mapping.budget.release_projected_mapping(16 * kUnit, extension);
      mapping.objects[0].bytes = mapping.objects[0].resident = 0;
      mapping.validate();
      for (int event : {7, 8, 9}) mapping.step(event);
      mapping.finish();
    }
    std::ostringstream details;
    for (const auto &[property, count] : failures)
      details << property << '=' << count << '\n';
    for (const auto &example : examples) details << example << '\n';
    EXPECT_TRUE(failures.empty()) << details.str();
  }
}

TEST(ProcessRssLedgerModelTest, NewObservationsExpireOldResidencyCredit) {
  for (int observer : {0, 1, 2}) {
    {
      std::map<std::string, size_t> failures;
      std::vector<std::string> examples;
      const std::vector<int> events;
      LedgerModel cold(observer, 1, false, false, failures, examples, events);
      cold.objects[0].bytes = cold.objects[0].capacity;
      cold.budget.account(cold.objects[0].bytes);
      cold.configure();
      cold.validate();
      for (int event : {3, 4}) cold.step(event);
      cold.finish();
      EXPECT_TRUE(failures.empty());
    }
    for (bool balanced_host_growth : {false, true}) {
      SCOPED_TRACE(observer);
      SCOPED_TRACE(balanced_host_growth);
      std::map<std::string, size_t> failures;
      std::vector<std::string> examples;
      const std::vector<int> events;
      LedgerModel model(observer, 0, false, false, failures, examples, events);
      const auto stale = model.budget.begin_resident_sample();
      const auto earlier = LedgerModel::ReadOwned();
      // The object stays live. Its pages disappear, optionally replaced by
      // host pages so even unchanged total RSS cannot renew an old proof.
      model.objects[0].resident = 0;
      if (balanced_host_growth) model.host += 16 * kUnit;
      model.budget.refresh(true);
      if (observer == 1)
        EXPECT_FALSE(model.budget.publish_resident_sample(stale, earlier));
      const size_t reserve =
          observer == 2 ? 16 * kUnit + ProcessRssBudget::headroom(kLimit) : 0;
      const size_t upper = model.rss() + model.backing() + reserve;
      const size_t room = kLimit - std::min(kLimit, upper);
      EXPECT_EQ(room, model.budget.available());
      // The state-derived admission boundary checks both directions.
      ASSERT_TRUE(model.budget.try_reserve(room));
      EXPECT_FALSE(model.budget.try_reserve(1));
      model.budget.finish_reservation(room, false);
      model.validate();
      model.proof();
      model.objects[0].resident = model.objects[0].bytes;
      model.budget.refresh(true);
      model.proof();
      for (int event : {3, 4}) model.step(event);
      model.finish();
      EXPECT_TRUE(failures.empty());
    }
  }
}
}  // namespace
}  // namespace zvec::ailego
