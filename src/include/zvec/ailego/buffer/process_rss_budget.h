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
class ProcessRssBudget {
 public:
  using Reader = size_t (*)();
  struct ResidentSample {
    size_t rss{0};
    // A lower bound observed in live, owned mappings in this RSS snapshot.
    // Reserved virtual pages and pending OS allocations must be excluded.
    size_t owned_resident{0};
  };
  using ResidentReader = ResidentSample (*)();
  struct ResidentSampleToken {
    uint64_t configuration{0};
    size_t retired_bytes{0};
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
    // Process-lifetime startup reservations stay charged to pool capacity.
    // Their residency is observed in the total process samples. Exclude them
    // from the reclaimable RSS ledger so later temporary frees cannot consume
    // their overlap or make an unchanged sample count them twice.
    backing_ -= permanent_backing;
    if (permanent_backing) {
      permanent_backing_.store(permanent_backing, std::memory_order_relaxed);
      estimate_.exclude_permanent(permanent_backing, backing_);
      native_estimate_.exclude_permanent(permanent_backing, backing_);
    }
    // Suspending or replacing observers must preserve the allowance for live,
    // possibly untouched backing. Only a fresh ledger adopts a new baseline.
    if (limit && !estimate_.initialized_) estimate_.reset(rss, backing_);
    if (limit && native_reader) {
      if (!native_estimate_.initialized_)
        native_estimate_.reset(native.footprint, backing_);
      publish_native(native);
    }
    native_reader_ = native_reader;
    reader_ = reader;
    resident_reader_ = resident_reader;
    ++resident_configuration_;
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
  // They cover overlapping bytes, and refusal must leave both unchanged.
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
    remember_resident_retirement(bytes);
    // Unlike unmapping, free() may leave every page resident in the allocator.
    // Keep its materialization debit until a successful sample observes it.
    // A new allocation absent from that sample cannot retire the sample's
    // overlap with a different, still-live owner.
    estimate_.release(bytes - rss_unsampled, backing_);
    if (bytes != rss_unsampled) ++resident_release_generation_;
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
    return {resident_configuration_, resident_retired_bytes_};
  }

  bool publish_resident_sample(ResidentSampleToken token,
                               ResidentSample sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!token.configuration ||
        token.configuration != resident_configuration_ || !limit() ||
        !reader_ || native_reader_ || !sample.rss ||
        resident_retired_bytes_ == std::numeric_limits<size_t>::max())
      return false;
    const size_t rss = reader_();
    rss_sample_failures_ = rss ? 0 : next_sample_failure(rss_sample_failures_);
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

  // A reusable mapping can fault in at most one mapping's worth of pages
  // per sample. Each reader can refresh independently.
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
    const auto previous = pages.stamp(slot);
    auto stamp = previous;
    if (!try_materialize_reusable_locked(bytes, stamp)) return false;
    if (stamp.rss_epoch != previous.rss_epoch ||
        stamp.native_epoch != previous.native_epoch) {
      pages.record(slot, stamp, /*new_backing=*/false);
    }
    return true;
  }

  // A partially used mapping can advance into fresh pages. Within each
  // reader's current sample, its projection cannot exceed the whole mapping.
  bool try_extend_mapping(size_t bytes, size_t mapping_bytes,
                          MappingProjection &projection) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(false);
    const auto rss = projection.rss.for_epoch(rss_epoch_);
    const auto native = projection.native.for_epoch(native_epoch_);
    assert(rss.bytes <= mapping_bytes && native.bytes <= mapping_bytes);
    const size_t rss_growth = std::min(bytes, mapping_bytes - rss.bytes);
    const size_t native_growth = std::min(bytes, mapping_bytes - native.bytes);
    if (!materialization_allowed_locked(rss_growth, native_growth,
                                        /*allow_at_limit=*/true))
      return false;
    if (limit()) {
      rss_materialized_ = add_saturated(rss_materialized_, rss_growth);
      if (native_reader_) {
        native_materialized_ =
            add_saturated(native_materialized_, native_growth);
      }
      projection.rss = {rss_epoch_, rss.bytes + rss_growth,
                        rss.unsampled + rss_growth};
      projection.native = {native_reader_ ? native_epoch_ : 0,
                           native.bytes + native_growth,
                           native.unsampled + native_growth};
      update_available();
    }
    return true;
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
  uint64_t resident_release_generation() const {
    return resident_release_generation_.load(std::memory_order_relaxed);
  }

 private:
  MaterializationStamp finish_reservation_locked(size_t bytes, bool allocated) {
    assert(bytes <= pending_allocations_);
    pending_allocations_ -= bytes;
    if (allocated) {
      backing_ += bytes;
      // Keep the in-flight debit until a successful sample can observe it.
      remember_materialization(bytes);
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
    remember_resident_retirement(bytes);
    // Only backing that could have contributed to a successful RSS sample
    // can explain a later RSS decrease.
    estimate_.release(bytes - rss.unsampled, backing_);
    if (bytes != rss.unsampled) ++resident_release_generation_;
    native_estimate_.release(bytes - native.unsampled, backing_);
    update_available();
  }

  bool try_materialize_reusable_locked(size_t bytes,
                                       MaterializationStamp &stamp) {
    refresh_locked(false);
    const bool charge_rss = stamp.rss_epoch != rss_epoch_ || !stamp.rss_epoch;
    const bool charge_native =
        native_reader_ &&
        (stamp.native_epoch != native_epoch_ || !stamp.native_epoch);
    if (!materialization_allowed_locked(charge_rss ? bytes : 0,
                                        charge_native ? bytes : 0,
                                        /*allow_at_limit=*/true))
      return false;
    if (limit() && bytes && (charge_rss || charge_native)) {
      if (charge_rss)
        rss_materialized_ = add_saturated(rss_materialized_, bytes);
      if (charge_native)
        native_materialized_ = add_saturated(native_materialized_, bytes);
      stamp = {rss_epoch_, native_reader_ ? native_epoch_ : 0};
      update_available();
    }
    return true;
  }

  static void refund_projection(size_t bytes, size_t &projection) {
    // Keep saturated projections conservative until the next sample.
    if (projection == std::numeric_limits<size_t>::max()) return;
    assert(bytes <= projection);
    projection -= bytes;
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
    // The existing allowance may overlap the RSS estimate even though this
    // operation still needs new pages. Check measured room independently of
    // the lifetime charge, including other backing allocations in flight.
    const size_t pending =
        add_saturated(pending_allocations_, pending_materializations_);
    const size_t needed = add_saturated(rss_bytes, pending);
    const size_t measured = add_saturated(rss(), rss_materialized_);
    if (measured > cap) return false;
    const size_t room = cap - measured;
    // A mapping already covered by this sample may reuse that debit even
    // when no room remains. A new projection still needs its full headroom.
    if ((!allow_at_limit && room == 0) || needed > room) return false;
    if (native_reader_) {
      const size_t native_cap = std::min(cap, native_limit_);
      const size_t footprint =
          add_saturated(native_estimate_.sample_, native_materialized_);
      const size_t reserve = headroom(native_cap);
      if (footprint > native_cap) return false;
      const size_t native_room = native_cap - footprint;
      if (native_room < reserve ||
          (!allow_at_limit && native_room == reserve) ||
          add_saturated(native_bytes, pending) > native_room - reserve)
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
    if (bytes > available_.load(std::memory_order_relaxed) ||
        (materialization && !materialization_allowed_locked(materialization)))
      refresh_after_denial_locked();
    if (materialization && !materialization_allowed_locked(materialization))
      return false;
    const size_t ledger_available =
        std::numeric_limits<size_t>::max() - backing_ - pending_allocations_;
    if (bytes > available_.load(std::memory_order_relaxed) ||
        bytes > ledger_available ||
        (pending_copy && materialization > std::numeric_limits<size_t>::max() -
                                               pending_materializations_)) {
      if (limit()) pressure_.store(true, std::memory_order_relaxed);
      return false;
    }
    counter += bytes;
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
    if (!rss_sample_failures_ && !native_sample_failures_ &&
        external_pressure_ != PressureLevel::kCritical)
      refresh_locked(true);
  }

  void refresh_locked(bool force) {
    if (!limit()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - sampled_at_ < std::chrono::milliseconds(20)) {
      return;
    }
    const auto resident = resident_reader_
                              ? resident_reader_()
                              : ResidentSample{reader_ ? reader_() : 0, 0};
    const size_t rss = resident.rss;
    // Keep the last observation across a transient failure. If either
    // configured reader keeps failing, pause growth until it recovers.
    rss_sample_failures_ = rss ? 0 : next_sample_failure(rss_sample_failures_);
    bool sampled_native = false;
    if (native_reader_) {
      NativeSample native;
      sampled_native = native_reader_(&native);
      if (sampled_native) publish_native(native);
      native_sample_failures_ =
          sampled_native ? 0 : next_sample_failure(native_sample_failures_);
    }
    if (rss && (!native_reader_ || sampled_native)) {
      recovery_sample_pending_ = false;
    }
    if (rss) {
      publish(rss, resident.owned_resident);
    } else {
      // Throttle unsuccessful reads too, so an OS sampling error does not
      // put a system call on every admission attempt.
      sampled_at_ = now;
      update_available();
    }
  }

  static constexpr uint8_t kMaxSampleFailures = 3;
  static uint8_t next_sample_failure(uint8_t failures) {
    return failures < kMaxSampleFailures ? failures + 1 : failures;
  }

  void publish_native(const NativeSample &native) {
    native_estimate_.observe(native.footprint, backing_);
    native_materialized_ = 0;
    ++native_epoch_;
    native_limit_ = add_saturated(native.footprint, native.available);
  }

  void publish(size_t rss, size_t owned_resident = 0) {
    // Ownership samplers also see lifetime mappings excluded from backing_.
    // Without per-owner identities, deduct their full permanent allowance:
    // those resident pages cannot witness overlap for temporary backing.
    owned_resident -= std::min(
        owned_resident, permanent_backing_.load(std::memory_order_relaxed));
    if (limit()) estimate_.observe(rss, backing_, owned_resident);
    rss_materialized_ = 0;
    ++rss_epoch_;
    rss_.store(rss, std::memory_order_relaxed);
    update_available();
    sampled_at_ = std::chrono::steady_clock::now();
  }

  static size_t add_saturated(size_t left, size_t right) {
    return left + std::min(right, std::numeric_limits<size_t>::max() - left);
  }

  struct Estimate {
    void reset(size_t sample, size_t backing) {
      baseline_rss_ = recovery_floor_ = sample_ = sample;
      baseline_backing_ = sampled_backing_ = backing;
      resident_upper_ = std::min(sample, backing);
      pending_releases_ = 0;
      initialized_ = true;
    }
    void exclude_permanent(size_t bytes, size_t backing) {
      // Excluded lifetime owners stay in measured RSS but no longer belong to
      // this ledger's overlap. They cannot retire unrelated live allowances.
      baseline_backing_ -= std::min(baseline_backing_, bytes);
      sampled_backing_ -= std::min(sampled_backing_, bytes);
      sampled_backing_ = std::min(sampled_backing_, backing);
      resident_upper_ -= std::min(bytes, resident_upper_ - pending_releases_);
    }
    void release(size_t bytes, size_t backing) {
      const size_t possibly_resident =
          std::min(bytes, resident_upper_ - pending_releases_);
      pending_releases_ += possibly_resident;
      // The old sample may still include this owner's pages. Those bytes
      // cannot also cover another owner that has not yet become resident.
      sampled_backing_ -= std::min(sampled_backing_, possibly_resident);
      sampled_backing_ = std::min(backing, sampled_backing_);
    }
    size_t usage(size_t backing) const {
      return std::max(projected(baseline_rss_, baseline_backing_, backing),
                      projected(sample_, sampled_backing_, backing));
    }
    void observe(size_t rss, size_t backing, size_t owned_resident = 0) {
      if (backing == 0) {
        // With no live reservations, any remaining allocator retention belongs
        // to this new RSS baseline. Later admissions start from zero backing.
        baseline_rss_ = rss;
        baseline_backing_ = 0;
        recovery_floor_ = rss;
        resident_upper_ = pending_releases_ = 0;
        sampled_backing_ = backing;
        sample_ = rss;
        return;
      }
      const size_t previous = sample_;
      if (rss < previous) {
        const size_t owned_drop = std::min(previous - rss, pending_releases_);
        pending_releases_ -= owned_drop;
        resident_upper_ -= owned_drop;

        // Move the RSS/backing anchor together when a release reaches RSS.
        // This preserves the charge for another, still untouched owner even
        // when it replaced resident backing that existed before configure().
        const size_t retired =
            std::min({owned_drop, baseline_rss_, baseline_backing_});
        baseline_rss_ -= retired;
        baseline_backing_ -= retired;

        // Ignore the return from an RSS excursion above the previous floor:
        // that growth was already covered by the sampled-RSS gate. Only a new
        // low, after accounting for owned releases, lowers the host baseline.
        const size_t floor_after_release =
            std::min(recovery_floor_, previous - owned_drop);
        if (rss < floor_after_release) {
          baseline_rss_ -= std::min(baseline_rss_, floor_after_release - rss);
        }
        recovery_floor_ = std::min(floor_after_release, rss);
      }

      // A live allocation's residency is not individually observable here.
      // Keep an upper bound rather than assuming all allocated bytes were
      // touched. RSS growth can increase it; allocation churn alone cannot.
      const size_t growth = rss > previous ? rss - previous : 0;
      resident_upper_ =
          std::min({rss, add_saturated(backing, pending_releases_),
                    add_saturated(resident_upper_, growth)});
      assert(pending_releases_ <= resident_upper_);
      // Ordinary observations cannot prove untouched backing became resident,
      // even before any owner retires. Only observed growth can cover more
      // backing; a later release must not transfer an old owner's overlap to
      // an untouched replacement.
      sampled_backing_ =
          std::min(backing, add_saturated(sampled_backing_, growth));
      // A fresh ownership/residency snapshot can prove overlap even when
      // evictions and refaults leave total RSS unchanged. It cannot cover
      // untouched owners, pending allocations, or more than this sample.
      const size_t witnessed = std::min({owned_resident, backing, rss});
      // Live witnessed pages and retired pages cannot occupy the same RSS.
      // Keep their bounds disjoint so a later witness retirement still
      // removes its own credit, rather than covering an untouched owner.
      pending_releases_ = std::min(pending_releases_, rss - witnessed);
      resident_upper_ =
          std::max(resident_upper_, witnessed + pending_releases_);
      sampled_backing_ = std::max(sampled_backing_, witnessed);
      sample_ = rss;
    }

    static size_t projected(size_t rss, size_t baseline, size_t backing) {
      const size_t growth = backing > baseline ? backing - baseline : 0;
      return rss + std::min(growth, std::numeric_limits<size_t>::max() - rss);
    }

    size_t baseline_rss_{0};
    size_t baseline_backing_{0};
    size_t sampled_backing_{0};
    size_t recovery_floor_{0};
    size_t resident_upper_{0};
    size_t pending_releases_{0};
    size_t sample_{0};
    bool initialized_{false};
  };

  void update_available() {
    const size_t cap = limit();
    // Take the larger estimate, never add managed bytes to measured RSS.
    // The lifetime estimate preserves lazy allocations across samples; the
    // latest sample catches host growth and includes subsequent admissions.
    const size_t measured = add_saturated(
        add_saturated(rss(), rss_materialized_), pending_materializations_);
    const size_t usage = add_saturated(
        std::max(estimate_.usage(backing_), measured), pending_allocations_);
    size_t available = cap ? (usage < cap ? cap - usage : 0)
                           : std::numeric_limits<size_t>::max();
    if (cap && native_reader_) {
      const size_t native_cap = std::min(cap, native_limit_);
      const size_t footprint = add_saturated(
          add_saturated(native_estimate_.sample_, native_materialized_),
          pending_materializations_);
      const size_t native_usage =
          add_saturated(std::max(native_estimate_.usage(backing_), footprint),
                        pending_allocations_);
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
    // A retired mapping may still be included in RSS while its replacement
    // is already resident. Ask the ownership sampler to confirm that overlap
    // before this conservative projection drives cache reclamation. Preserve
    // the lifetime reservation and every pending allocation/materialization.
    const size_t confirmed_usage = add_saturated(
        std::max(Estimate::projected(estimate_.baseline_rss_,
                                     estimate_.baseline_backing_, backing_),
                 measured),
        pending_allocations_);
    resident_confirmation_.store(
        pressure && cap && resident_reader_ && !native_reader_ &&
            external_pressure_ == PressureLevel::kNormal &&
            !admission_paused() && estimate_.pending_releases_ &&
            confirmed_usage < usage && confirmed_usage < cap &&
            cap - confirmed_usage >= 2 * margin,
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
  uint64_t rss_epoch_{0};
  uint64_t native_epoch_{0};
  std::atomic<bool> resident_confirmation_{false};
  std::atomic<uint64_t> resident_release_generation_{0};
  uint64_t resident_configuration_{0};
  size_t resident_retired_bytes_{0};
};
}  // namespace zvec::ailego
