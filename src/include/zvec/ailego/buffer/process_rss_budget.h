// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace zvec::ailego {

// A second admission signal alongside allocation accounting. RSS includes
// caller-owned memory and allocator retention; it is never added to used().
// Sampling and uninstrumented allocations make this a soft process target.
// Each observation charges process usage + live backing - that observation's
// owned residency, plus pending work and independent copy/refault exposure.
// A new process-only observation cannot renew an older ownership witness.
class ProcessRssBudget {
  friend struct ProcessRssBudgetTestPeer;

 public:
  using Reader = size_t (*)();
  struct ResidentSample {
    size_t rss{0};
    // A lower bound in mappings still charged to this ledger. Exclude
    // permanent native image/stack allowances, reserved virtual pages and
    // pending OS allocations. The provider owns this classification.
    size_t owned_resident{0};
  };
  using ResidentReader = ResidentSample (*)();
  struct ResidentSampleToken {
    uint64_t configuration{0};
    uint64_t observation{0};
    size_t retired_bytes{0};
    uint64_t scan{0};
  };
  struct NativeSample {
    size_t footprint{0};
    // Zero is a valid exhausted limit, not a failed sample.
    size_t available{0};
  };
  using NativeReader = bool (*)(NativeSample *);
  // Allocation age is relative to each reader's successful observations.
  // RSS and native footprint can refresh separately.
  struct MaterializationStamp {
    uint64_t rss_epoch{0};
    uint64_t native_epoch{0};
  };
  // Heap owners can contain allocations from several observations. Only
  // bytes created after a reader's last sample are certainly absent from it.
  struct HeapProjection {
    struct Sample {
      uint64_t epoch{0};
      size_t bytes{0};

      void append(Sample rhs) {
        if (rhs.epoch > epoch) {
          *this = rhs;
        } else if (rhs.epoch == epoch) {
          bytes += rhs.bytes;
        }
      }
      Sample take(size_t total, size_t released) {
        // Without individual allocation identities, retire potentially
        // observed bytes first. Never assign more new bytes than are certain.
        const size_t observed = total - bytes;
        const size_t transferred =
            released > observed ? released - observed : 0;
        bytes -= transferred;
        return {epoch, transferred};
      }
      size_t for_epoch(uint64_t current) const {
        return epoch && epoch == current ? bytes : 0;
      }
    };

    static HeapProjection Full(size_t bytes, MaterializationStamp stamp) {
      return {{stamp.rss_epoch, stamp.rss_epoch ? bytes : 0},
              {stamp.native_epoch, stamp.native_epoch ? bytes : 0}};
    }
    void append(HeapProjection rhs) {
      rss.append(rhs.rss);
      native.append(rhs.native);
    }
    HeapProjection take(size_t total, size_t released) {
      return {rss.take(total, released), native.take(total, released)};
    }

    Sample rss{};
    Sample native{};
  };
  struct MappingProjection {
    struct Sample {
      uint64_t epoch{0};
      size_t bytes{0};
      // Existing backing may have been resident in the last sample even
      // when another owner has since reserved headroom to reuse it.
      size_t unsampled{0};

      Sample for_epoch(uint64_t current) const {
        return epoch && epoch == current ? *this : Sample{};
      }
    };

    static MappingProjection Full(size_t bytes, MaterializationStamp stamp) {
      return ForReuse(bytes, stamp, stamp);
    }

    static MappingProjection ForReuse(size_t bytes, MaterializationStamp stamp,
                                      MaterializationStamp creation) {
      return {{stamp.rss_epoch, bytes,
               stamp.rss_epoch == creation.rss_epoch ? bytes : 0},
              {stamp.native_epoch, bytes,
               stamp.native_epoch == creation.native_epoch ? bytes : 0}};
    }

    Sample rss{};
    Sample native{};
  };

  // One stamp per projected page, compressed into the slab's existing
  // header page. All access uses the RSS accounting mutex, including reuse
  // across different free-list shards.
  template <size_t PageCount>
  class ReusablePages {
    friend class ProcessRssBudget;
    struct SamplePages {
      void record(size_t slot, uint64_t sample_epoch, bool new_backing) {
        if (epoch != sample_epoch) {
          projected.reset();
          unsampled.reset();
          epoch = sample_epoch;
        }
        projected[slot] = sample_epoch != 0;
        if (new_backing) unsampled[slot] = sample_epoch != 0;
      }

      uint64_t epoch{0};
      std::bitset<PageCount> projected;
      std::bitset<PageCount> unsampled;
    };

    MaterializationStamp stamp(size_t slot) const {
      return {rss_.projected[slot] ? rss_.epoch : 0,
              native_.projected[slot] ? native_.epoch : 0};
    }

    void record(size_t slot, MaterializationStamp stamp, bool new_backing) {
      rss_.record(slot, stamp.rss_epoch, new_backing);
      native_.record(slot, stamp.native_epoch, new_backing);
    }

    MappingProjection take(size_t slot, size_t bytes) {
      const MaterializationStamp projected = stamp(slot);
      const MaterializationStamp creation{
          rss_.unsampled[slot] ? rss_.epoch : 0,
          native_.unsampled[slot] ? native_.epoch : 0};
      rss_.projected[slot] = false;
      native_.projected[slot] = false;
      rss_.unsampled[slot] = false;
      native_.unsampled[slot] = false;
      return MappingProjection::ForReuse(bytes, projected, creation);
    }

    SamplePages rss_;
    SamplePages native_;
  };
  enum class PressureLevel { kNormal, kBackground, kCritical };

  void set_external_pressure(PressureLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level != external_pressure_) {
      // An unlocked ownership scan must not cross a host policy change,
      // including NORMAL -> CRITICAL -> NORMAL. Keep allocation epochs
      // intact: a policy notification is not an RSS observation.
      invalidate_resident_samples();
    }
    if (limit() && level < external_pressure_) {
      // The host can relax its policy independently of the current RSS or
      // application limit. Keep admission closed until the next fresh sample,
      // without performing a system read in the lifecycle notification.
      recovery_sample_pending_ = true;
      // Make the first recovery observation immediately eligible. If it
      // fails, later admissions still use the ordinary sampling interval.
      sampled_at_ = {};
    }
    external_pressure_ = level;
    update_available();
  }

  static size_t headroom(size_t limit) {
    if (!limit) return 0;
    return std::min(std::max(size_t{1}, limit / 8),
                    std::clamp(limit / 20, size_t{8} << 20, size_t{64} << 20));
  }

  bool configure(size_t limit, Reader reader,
                 NativeReader native_reader = nullptr,
                 size_t permanent_backing = 0,
                 ResidentReader resident_reader = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (permanent_backing &&
        (!limit || this->limit() ||
         permanent_backing_.load(std::memory_order_relaxed) != 0 ||
         permanent_backing > backing_))
      return false;
    const auto resident =
        limit && resident_reader
            ? resident_reader()
            : ResidentSample{limit && reader ? reader() : 0, 0};
    const size_t rss = resident.rss;
    if (limit && !rss) return false;
    NativeSample native;
    if (limit && native_reader && !native_reader(&native)) return false;
    // Permanent image/stack allowances are outside the ownership provider's
    // mapping directory. Keep their measured RSS, but remove their lifetime
    // quota here. A provider must not report them as owned_resident.
    backing_ -= permanent_backing;
    if (permanent_backing) {
      permanent_backing_.store(permanent_backing, std::memory_order_relaxed);
      estimate_.exclude_permanent(permanent_backing, backing_);
      native_estimate_.exclude_permanent(permanent_backing, backing_);
    }
    // A configuration change takes a new observation. Startup accounting,
    // disabled intervals and reader replacement never establish residency.
    if (limit && native_reader) {
      publish_native(native);
    }
    native_reader_ = native_reader;
    reader_ = reader;
    resident_reader_ = resident_reader;
    invalidate_resident_samples();
    resident_retired_bytes_ = 0;
    recovery_sample_pending_ = false;
    rss_sample_failures_ = native_sample_failures_ = 0;
    limit_.store(limit, std::memory_order_relaxed);
    pressure_.store(false, std::memory_order_relaxed);
    if (limit) {
      publish(rss, resident.owned_resident);
    } else {
      // No observation occurs while disabled. Keep epochs and projections so
      // releases can still retire their own debits. A drained ledger can be
      // reused by a new native pool without inheriting old overlap history.
      if (!backing_ && !pending_allocations_ && !pending_materializations_) {
        estimate_ = {};
        native_estimate_ = {};
      }
      update_available();
    }
    return true;
  }

  // Keep new backing charged for its entire lifetime, including before it is
  // faulted in. Sampling must not repeatedly admit the same unused headroom.
  // Existing backing reused by the pool is already included in this ledger.
  bool try_acquire(size_t bytes) {
    return try_acquire(bytes, 0);
  }

  // A growing allowance can materialize more than the new lifetime charge
  // when it also consumes some previously reserved capacity.
  bool try_acquire(size_t bytes, size_t materialization) {
    return try_charge(bytes, backing_, materialization);
  }

  // Admit a planning allowance and its in-flight copy as one operation.
  // Without an owner identity, the copy cannot spend another allowance's
  // untouched backing. Refusal must leave both charges unchanged.
  bool try_acquire_with_pending_copy(size_t bytes, size_t materialization) {
    return try_charge(bytes, backing_, materialization, true);
  }

  // Reserve for an OS allocation that has yet to return. Samples cannot
  // assign these bytes to resident backing before the allocation succeeds.
  bool try_reserve(size_t bytes) {
    return try_charge(bytes, pending_allocations_);
  }

  MaterializationStamp finish_reservation(size_t bytes, bool allocated) {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_reservation_locked(bytes, allocated);
  }

  template <size_t PageCount>
  void commit_reusable_page(size_t bytes, ReusablePages<PageCount> &pages,
                            size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    pages.record(slot, finish_reservation_locked(bytes, true),
                 /*new_backing=*/true);
  }

  // Some consumers (RocksDB's block reader) cannot safely reject allocation
  // here. Their cache owns admission; still include temporary and live backing
  // in the process estimate so other consumers stop growing under pressure.
  void account(size_t bytes) {
    const bool accounted = account(bytes, backing_);
    assert(accounted);
    (void)accounted;
  }

  bool account_pending(size_t bytes) {
    return account(bytes, pending_allocations_);
  }

  void release(size_t bytes) {
    release_unmapped(bytes, {});
  }

  void release_heap(size_t bytes, HeapProjection projection) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t rss_unsampled = projection.rss.for_epoch(rss_epoch_);
    const size_t native_unsampled = projection.native.for_epoch(native_epoch_);
    assert(rss_unsampled <= bytes && native_unsampled <= bytes);
    assert(bytes <= backing_);
    backing_ -= bytes;
    retire_new_backing(bytes, projection.rss.epoch, rss_unsampled,
                       rss_new_backing_);
    retire_new_backing(bytes, projection.native.epoch, native_unsampled,
                       native_new_backing_);
    remember_resident_retirement(bytes);
    // Unlike unmapping, free() may leave every page resident in the allocator.
    // Keep its materialization debit until a successful sample observes it.
    // A new allocation absent from that sample cannot retire the sample's
    // overlap with a different, still-live owner.
    estimate_.release(bytes - rss_unsampled, backing_);
    if (bytes != rss_unsampled) ++resident_generation_;
    native_estimate_.release(bytes - native_unsampled, backing_);
    update_available();
  }

  // A stamp is valid only after successful unmapping. A partial release
  // keeps the same stamp for the remaining bytes. Measured RSS is unchanged.
  void release_unmapped(size_t bytes, MaterializationStamp stamp) {
    release_projected_mapping(bytes, MappingProjection::Full(bytes, stamp));
  }

  void release_projected_mapping(size_t bytes, MappingProjection projection) {
    std::lock_guard<std::mutex> lock(mutex_);
    release_projected_mapping_locked(bytes, projection);
  }

  template <size_t PageCount>
  void release_reusable_page(size_t bytes, ReusablePages<PageCount> &pages,
                             size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    release_projected_mapping_locked(bytes, pages.take(slot, bytes));
  }

  void refresh(bool force = false) {
    if (!limit()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(force);
  }

  // Native ownership scans may run without the budget mutex. Their provider
  // must snapshot only owned, precharged mapping ranges. Retirements (including
  // cancelled pending allocations) invalidate the corresponding credit even
  // when an address is unmapped and reused during the scan.
  ResidentSampleToken begin_resident_sample() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!limit() || !reader_ || native_reader_ ||
        resident_retired_bytes_ == std::numeric_limits<size_t>::max())
      return {};
    // An abandoned scan cannot defer observations indefinitely. Refresh
    // before starting a replacement window, including after reader recovery.
    const auto now = std::chrono::steady_clock::now();
    if (active_resident_scan_ &&
        now - resident_scan_started_ >= kResidentScanWindow)
      refresh_locked(true);
    if (rss_sample_failures_) return {};
    if (!active_resident_scan_) {
      active_resident_scan_ = ++resident_scan_sequence_;
      if (!active_resident_scan_)
        active_resident_scan_ = ++resident_scan_sequence_;
      resident_scan_started_ = std::chrono::steady_clock::now();
    }
    return {resident_configuration_, rss_epoch_, resident_retired_bytes_,
            active_resident_scan_};
  }

  bool publish_resident_sample(ResidentSampleToken token,
                               ResidentSample sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!token.configuration ||
        token.configuration != resident_configuration_ || !limit() ||
        token.observation != rss_epoch_ || !token.scan ||
        token.scan != active_resident_scan_ || !reader_ || native_reader_ ||
        resident_retired_bytes_ == std::numeric_limits<size_t>::max())
      return false;
    if (!sample.rss) {
      record_sample_result(false, rss_sample_failures_);
      sampled_at_ = std::chrono::steady_clock::now();
      update_available();
      return false;
    }
    if (std::chrono::steady_clock::now() - resident_scan_started_ >=
        kResidentScanWindow) {
      refresh_locked(true);
      return false;
    }
    const size_t rss = reader_();
    record_sample_result(rss != 0, rss_sample_failures_);
    if (!rss) {
      sampled_at_ = std::chrono::steady_clock::now();
      update_available();
      return false;
    }
    size_t credit = sample.owned_resident;
    const size_t retired = resident_retired_bytes_ - token.retired_bytes;
    credit -= std::min(credit, retired);
    // A sampled page may belong to an allocation whose OS call has not
    // committed its backing debit yet. It cannot cover another live owner.
    credit -= std::min(credit, pending_allocations_);
    if (rss < sample.rss) credit -= std::min(credit, sample.rss - rss);
    recovery_sample_pending_ = false;
    publish(rss, credit);
    return true;
  }

  // Ordinary-heap allowances retain accounting, not reusable physical buffers.
  // Reusing their high water still calls malloc/new and must obey a new host
  // pressure notification or an exhausted observed process/application limit.
  // Do not charge the reserved backing twice or block mandatory work merely
  // because optional caches are under BACKGROUND pressure.
  bool materialization_allowed(size_t bytes = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes) {
      refresh_for_materialization_locked(bytes);
    } else {
      refresh_locked(false);
    }
    return materialization_allowed_locked(bytes);
  }

  // Reuses of already-charged backing share the measured headroom until
  // the corresponding reader supplies its next successful observation.
  bool try_materialize(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_for_materialization_locked(bytes);
    if (!materialization_allowed_locked(bytes)) return false;
    if (limit() && bytes) {
      remember_materialization(bytes);
      update_available();
    }
    return true;
  }

  // A live reusable mapping's full future fault cost is already charged as
  // backing. Unlike a separate heap copy, reuse creates no extra storage.
  bool try_materialize_reusable(size_t bytes, MaterializationStamp &stamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    return try_materialize_reusable_locked(bytes, stamp);
  }

  template <size_t PageCount>
  bool try_materialize_reusable_page(size_t bytes,
                                     ReusablePages<PageCount> &pages,
                                     size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!limit()) return !admission_paused();
    auto stamp = pages.stamp(slot);
    return try_materialize_reusable_locked(bytes, stamp);
  }

  // Registered mappings retain their complete lifetime fault allowance.
  // Extending within one mapping creates no independent storage, and must
  // not change its allocation age or refund another owner's projection.
  bool try_extend_mapping(size_t /*bytes*/, size_t /*mapping_bytes*/,
                          MappingProjection & /*projection*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(false);
    return materialization_allowed_locked(0, 0, /*allow_at_limit=*/true);
  }

  // Reserve measured headroom until mapping or prefaulting completes.
  // An intervening sample must not clear a debit for work still in flight.
  bool try_reserve_materialization(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_for_materialization_locked(bytes);
    if (!materialization_allowed_locked(bytes) ||
        bytes > std::numeric_limits<size_t>::max() - pending_materializations_)
      return false;
    pending_materializations_ += bytes;
    update_available();
    return true;
  }

  MaterializationStamp finish_materialization(size_t bytes, bool mapped) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(bytes <= pending_materializations_);
    pending_materializations_ -= bytes;
    if (mapped) remember_materialization(bytes);
    update_available();
    if (!mapped || !limit()) return {};
    return {rss_epoch_, native_reader_ ? native_epoch_ : 0};
  }

  // Unmapping prepaid storage leaves the owner's lifetime charge in place.
  void release_materialization(size_t bytes, MaterializationStamp stamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    refund_materialization(bytes, stamp);
    update_available();
  }

  size_t limit() const {
    return limit_.load(std::memory_order_relaxed);
  }
  size_t permanent_backing() const {
    return permanent_backing_.load(std::memory_order_relaxed);
  }
  size_t available() const {
    return available_.load(std::memory_order_relaxed);
  }
  size_t rss() const {
    return rss_.load(std::memory_order_relaxed);
  }
  bool pressure() const {
    return pressure_.load(std::memory_order_relaxed);
  }

  bool needs_resident_confirmation() const {
    return resident_confirmation_.load(std::memory_order_relaxed);
  }
  bool has_resident_uncertainty() const {
    return resident_uncertainty_.load(std::memory_order_relaxed);
  }
  uint64_t resident_generation() const {
    return resident_generation_.load(std::memory_order_relaxed);
  }

 private:
  MaterializationStamp finish_reservation_locked(size_t bytes, bool allocated) {
    assert(bytes <= pending_allocations_);
    pending_allocations_ -= bytes;
    if (allocated) {
      backing_ += bytes;
      // Keep the in-flight debit until a successful sample can observe it.
      remember_materialization(bytes);
      if (limit()) {
        rss_new_backing_ = add_saturated(rss_new_backing_, bytes);
        if (native_reader_)
          native_new_backing_ = add_saturated(native_new_backing_, bytes);
      }
      ++resident_generation_;
    } else {
      remember_resident_retirement(bytes);
    }
    update_available();
    if (!allocated || !limit()) return {};
    return {rss_epoch_, native_reader_ ? native_epoch_ : 0};
  }

  void release_projected_mapping_locked(size_t bytes,
                                        const MappingProjection &projection) {
    const auto rss = projection.rss.for_epoch(rss_epoch_);
    const auto native = projection.native.for_epoch(native_epoch_);
    assert(rss.unsampled <= rss.bytes && rss.bytes <= bytes);
    assert(native.unsampled <= native.bytes && native.bytes <= bytes);
    refund_projection(rss.bytes, rss_materialized_);
    refund_projection(native.bytes, native_materialized_);
    assert(bytes <= backing_);
    backing_ -= bytes;
    retire_new_backing(bytes, projection.rss.epoch, rss.unsampled,
                       rss_new_backing_);
    retire_new_backing(bytes, projection.native.epoch, native.unsampled,
                       native_new_backing_);
    remember_resident_retirement(bytes);
    // Only backing that could have contributed to a successful RSS sample
    // can explain a later RSS decrease.
    estimate_.release(bytes - rss.unsampled, backing_);
    if (bytes != rss.unsampled) ++resident_generation_;
    native_estimate_.release(bytes - native.unsampled, backing_);
    update_available();
  }

  bool try_materialize_reusable_locked(size_t /*bytes*/,
                                       MaterializationStamp & /*stamp*/) {
    refresh_locked(false);
    // Do not advance the birth stamp: only a real OS allocation is absent
    // from an older observation. The complete mapping stays in backing_,
    // including its cold pages, so an in-place refault needs no copy debit.
    return materialization_allowed_locked(0, 0, /*allow_at_limit=*/true);
  }

  static void refund_projection(size_t bytes, size_t &projection) {
    // Keep saturated projections conservative until the next sample.
    if (projection == std::numeric_limits<size_t>::max()) return;
    assert(bytes <= projection);
    projection -= bytes;
  }

  static void retire_new_backing(size_t bytes, uint64_t epoch, size_t unsampled,
                                 size_t &overlap) {
    // Legacy owners with no stamp cannot establish which cohort retired.
    // Removing possible overlap is conservative until the next observation.
    overlap -= std::min(overlap, epoch ? unsampled : bytes);
  }

  void remember_resident_retirement(size_t bytes) {
    resident_retired_bytes_ = add_saturated(resident_retired_bytes_, bytes);
  }

  void refund_materialization(size_t bytes, MaterializationStamp stamp) {
    if (stamp.rss_epoch && stamp.rss_epoch == rss_epoch_)
      refund_projection(bytes, rss_materialized_);
    if (stamp.native_epoch && stamp.native_epoch == native_epoch_)
      refund_projection(bytes, native_materialized_);
  }

  void remember_materialization(size_t bytes) {
    if (!limit()) return;
    rss_materialized_ = add_saturated(rss_materialized_, bytes);
    if (native_reader_)
      native_materialized_ = add_saturated(native_materialized_, bytes);
  }

  bool admission_paused() const {
    return external_pressure_ == PressureLevel::kCritical ||
           recovery_sample_pending_ ||
           rss_sample_failures_ == kMaxSampleFailures ||
           native_sample_failures_ == kMaxSampleFailures;
  }

  bool materialization_allowed_locked(size_t bytes) const {
    return materialization_allowed_locked(bytes, bytes,
                                          /*allow_at_limit=*/false);
  }

  bool materialization_allowed_locked(size_t rss_bytes, size_t native_bytes,
                                      bool allow_at_limit) const {
    if (admission_paused()) {
      return false;
    }
    const size_t cap = limit();
    if (!cap) return true;
    const size_t usage =
        projected_usage(estimate_, rss_materialized_, rss_new_backing_);
    if (usage > cap) return false;
    const size_t room = cap - usage;
    // A mapping already covered by this sample may reuse that debit even
    // when no room remains. A new projection still needs its full headroom.
    if ((!allow_at_limit && room == 0) || rss_bytes > room) return false;
    if (native_reader_) {
      const size_t native_cap = std::min(cap, native_limit_);
      const size_t footprint = projected_usage(
          native_estimate_, native_materialized_, native_new_backing_);
      const size_t reserve = headroom(native_cap);
      if (footprint > native_cap) return false;
      const size_t native_room = native_cap - footprint;
      if (native_room < reserve ||
          (!allow_at_limit && native_room == reserve) ||
          native_bytes > native_room - reserve)
        return false;
    }
    return true;
  }

  bool account(size_t bytes, size_t &counter) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes >
        std::numeric_limits<size_t>::max() - backing_ - pending_allocations_)
      return false;
    counter += bytes;
    update_available();
    return true;
  }

  bool try_charge(size_t bytes, size_t &counter, size_t materialization = 0,
                  bool pending_copy = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(false);
    const size_t required = add_saturated(bytes, materialization);
    if (required > available_.load(std::memory_order_relaxed) ||
        (materialization && !materialization_allowed_locked(materialization)))
      refresh_after_denial_locked();
    if (materialization && !materialization_allowed_locked(materialization))
      return false;
    const size_t ledger_available =
        std::numeric_limits<size_t>::max() - backing_ - pending_allocations_;
    if (required > available_.load(std::memory_order_relaxed) ||
        bytes > ledger_available ||
        (pending_copy && materialization > std::numeric_limits<size_t>::max() -
                                               pending_materializations_)) {
      if (limit()) pressure_.store(true, std::memory_order_relaxed);
      return false;
    }
    counter += bytes;
    if (&counter == &backing_) {
      ++resident_generation_;
    }
    if (pending_copy) {
      pending_materializations_ += materialization;
    } else {
      remember_materialization(materialization);
    }
    update_available();
    return true;
  }

  void refresh_for_materialization_locked(size_t bytes) {
    refresh_locked(false);
    if (!materialization_allowed_locked(bytes)) refresh_after_denial_locked();
  }

  void refresh_after_denial_locked() {
    // A caller can free a rejected input or completed result and retry within
    // the normal sampling interval. Before denying that retry, observe its
    // current RSS instead of retaining a transient peak or copy projection.
    // Pending allocations/copies remain charged across this observation.
    // Failed readers keep their sampling backoff, and host critical pressure
    // cannot be relaxed by an RSS observation.
    if (active_resident_scan_ &&
        std::chrono::steady_clock::now() - resident_scan_started_ <
            kResidentScanWindow)
      return;
    if (!rss_sample_failures_ && !native_sample_failures_ &&
        external_pressure_ != PressureLevel::kCritical &&
        !(has_resident_uncertainty() &&
          std::chrono::steady_clock::now() - sampled_at_ <
              std::chrono::milliseconds(20)))
      refresh_locked(true);
  }

  void refresh_locked(bool force) {
    if (!limit()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - sampled_at_ < std::chrono::milliseconds(20)) {
      return;
    }
    // A normal raw read would erase the scan's proof before it can publish.
    // Let the single maintenance scan finish in a bounded window. Forced
    // reads, pressure/configuration changes and failures still expire it;
    // native application-footprint reads never use this RSS-only window.
    if (!force && active_resident_scan_ &&
        now - resident_scan_started_ < kResidentScanWindow)
      return;
    const auto resident = resident_reader_
                              ? resident_reader_()
                              : ResidentSample{reader_ ? reader_() : 0, 0};
    const size_t rss = resident.rss;
    // Keep the last observation across a transient failure. If either
    // configured reader keeps failing, pause growth until it recovers.
    record_sample_result(rss != 0, rss_sample_failures_);
    bool sampled_native = false;
    if (native_reader_) {
      NativeSample native;
      sampled_native = native_reader_(&native);
      if (sampled_native) publish_native(native);
      record_sample_result(sampled_native, native_sample_failures_);
    }
    if (rss && (!native_reader_ || sampled_native)) {
      recovery_sample_pending_ = false;
    }
    if (rss) {
      ++resident_generation_;
      publish(rss, resident.owned_resident);
    } else {
      // Throttle unsuccessful reads too, so an OS sampling error does not
      // put a system call on every admission attempt.
      sampled_at_ = now;
      update_available();
    }
  }

  static constexpr uint8_t kMaxSampleFailures = 3;
  static constexpr auto kResidentScanWindow = std::chrono::milliseconds(100);
  void invalidate_resident_samples() {
    active_resident_scan_ = 0;
    ++resident_configuration_;
    ++resident_generation_;
  }
  void record_sample_result(bool success, uint8_t &failures) {
    if (!success && !failures) invalidate_resident_samples();
    failures = success ? 0 : next_sample_failure(failures);
  }
  static uint8_t next_sample_failure(uint8_t failures) {
    return failures < kMaxSampleFailures ? failures + 1 : failures;
  }

  void publish_native(const NativeSample &native) {
    native_estimate_.observe(native.footprint, backing_);
    native_materialized_ = 0;
    native_new_backing_ = 0;
    ++native_epoch_;
    native_limit_ = add_saturated(native.footprint, native.available);
  }

  void publish(size_t rss, size_t owned_resident = 0) {
    // The provider already excludes permanent native owners. Deducting their
    // unrelated quota again would erase valid registered-mapping residency.
    active_resident_scan_ = 0;
    if (limit()) estimate_.observe(rss, backing_, owned_resident);
    rss_materialized_ = 0;
    rss_new_backing_ = 0;
    ++rss_epoch_;
    rss_.store(rss, std::memory_order_relaxed);
    update_available();
    sampled_at_ = std::chrono::steady_clock::now();
  }

  static size_t add_saturated(size_t left, size_t right) {
    return left + std::min(right, std::numeric_limits<size_t>::max() - left);
  }

  // One observation, one residency lower bound. Keeping a second historical
  // RSS/backing anchor made its overlap outlive the pages that justified it.
  struct Estimate {
    void exclude_permanent(size_t bytes, size_t backing) {
      credit_ -= std::min(credit_, bytes);
      credit_ = std::min(credit_, backing);
    }
    void release(size_t bytes, size_t backing) {
      // Without individual identities, retire potentially credited bytes
      // first. An unsampled OS birth is excluded by its release projection.
      credit_ -= std::min(credit_, bytes);
      credit_ = std::min(credit_, backing);
    }
    size_t usage(size_t backing) const {
      return add_saturated(sample_, backing - std::min(backing, credit_));
    }
    void observe(size_t sample, size_t backing, size_t owned_resident = 0) {
      // Even unchanged process usage can hide live page loss compensated by
      // host growth. Only this observation's witness can offset live backing.
      // A missing/partial proof is conservative; it never renews old credit.
      sample_ = sample;
      credit_ = std::min({owned_resident, backing, sample});
    }

    size_t sample_{0};
    size_t credit_{0};
  };

  size_t projected_usage(const Estimate &estimate, size_t materialized,
                         size_t new_backing) const {
    // Only current-epoch OS births prove the lifetime and materialization
    // debits refer to the same bytes. Reuse/copy work belongs to independent
    // exposure and cannot hide behind a different owner's lazy allowance.
    const size_t extra =
        materialized == std::numeric_limits<size_t>::max()
            ? materialized
            : materialized - std::min(materialized, new_backing);
    return add_saturated(
        add_saturated(estimate.usage(backing_), extra),
        add_saturated(pending_materializations_, pending_allocations_));
  }

  void update_available() {
    const size_t cap = limit();
    const size_t measured = add_saturated(
        add_saturated(rss(), rss_materialized_), pending_materializations_);
    const size_t usage =
        projected_usage(estimate_, rss_materialized_, rss_new_backing_);
    size_t available = cap ? (usage < cap ? cap - usage : 0)
                           : std::numeric_limits<size_t>::max();
    if (cap && native_reader_) {
      const size_t native_cap = std::min(cap, native_limit_);
      const size_t native_usage = projected_usage(
          native_estimate_, native_materialized_, native_new_backing_);
      const size_t remaining =
          native_usage < native_cap ? native_cap - native_usage : 0;
      // Keep a reserve below the OS application limit even for mandatory work.
      const size_t reserve = headroom(native_cap);
      available =
          std::min(available, remaining > reserve ? remaining - reserve : 0);
    }
    if (admission_paused()) available = 0;
    const size_t margin = headroom(cap);
    const bool pressure = external_pressure_ != PressureLevel::kNormal ||
                          (cap && (pressure_.load(std::memory_order_relaxed)
                                       ? available < 2 * margin
                                       : available <= margin));
    available_.store(available, std::memory_order_relaxed);
    pressure_.store(pressure, std::memory_order_relaxed);
    // A fresh proof can recover an initial or retired owner's overlap. Ask
    // before an uncertain lifetime projection drives cache reclamation.
    const size_t confirmed_usage =
        add_saturated(measured, pending_allocations_);
    const bool uncertain = cap && resident_reader_ && !native_reader_ &&
                           !admission_paused() && confirmed_usage < usage &&
                           confirmed_usage < cap;
    resident_uncertainty_.store(uncertain, std::memory_order_relaxed);
    resident_confirmation_.store(
        pressure && external_pressure_ == PressureLevel::kNormal && uncertain,
        std::memory_order_relaxed);
  }

  std::mutex mutex_;
  Reader reader_{nullptr};
  ResidentReader resident_reader_{nullptr};
  size_t backing_{0};
  // This exclusion lasts for the owners' entire process lifetime, including
  // while the RSS gate is disabled. It must never be deducted a second time.
  std::atomic<size_t> permanent_backing_{0};
  size_t pending_allocations_{0};
  size_t pending_materializations_{0};
  Estimate estimate_;
  Estimate native_estimate_;
  NativeReader native_reader_{nullptr};
  size_t native_limit_{0};
  PressureLevel external_pressure_{PressureLevel::kNormal};
  bool recovery_sample_pending_{false};
  uint8_t rss_sample_failures_{0};
  uint8_t native_sample_failures_{0};
  std::chrono::steady_clock::time_point sampled_at_{};
  std::atomic<size_t> limit_{0};
  std::atomic<size_t> available_{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> rss_{0};
  std::atomic<bool> pressure_{false};
  size_t rss_materialized_{0};
  size_t native_materialized_{0};
  size_t rss_new_backing_{0};
  size_t native_new_backing_{0};
  uint64_t rss_epoch_{0};
  uint64_t native_epoch_{0};
  std::atomic<bool> resident_confirmation_{false};
  std::atomic<bool> resident_uncertainty_{false};
  std::atomic<uint64_t> resident_generation_{0};
  uint64_t resident_configuration_{0};
  uint64_t resident_scan_sequence_{0};
  uint64_t active_resident_scan_{0};
  std::chrono::steady_clock::time_point resident_scan_started_{};
  size_t resident_retired_bytes_{0};
};
}  // namespace zvec::ailego
