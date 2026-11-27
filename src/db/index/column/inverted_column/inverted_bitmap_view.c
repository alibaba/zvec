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

#include "inverted_bitmap_view.h"
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(_MSC_VER)
#include <immintrin.h>
#define ZVEC_BITMAP_X86_TARGETS 1
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Read-only container ABI of the pinned CRoaring 2.0.4 C implementation.
// Keep this adapter in C: these are the same tagged structure types used by
// roaring.c, without depending on its unaligned portable frozen reader.
// 以下四个结构体保留 CRoaring 标签名，确保跨编译单元的 C 类型兼容性。
// NOLINTNEXTLINE(readability-identifier-naming)
struct array_container_s {
  int32_t cardinality;
  int32_t capacity;
  uint16_t *array;
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct bitset_container_s {
  int32_t cardinality;
  uint64_t *words;
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct rle16_s {
  uint16_t value;
  uint16_t length;
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct run_container_s {
  int32_t n_runs;
  int32_t capacity;
  struct rle16_s *runs;
};

union ViewContainer {
  struct array_container_s array;
  struct bitset_container_s bitset;
  struct run_container_s run;
};

enum { VIEW_BITSET = 1, VIEW_ARRAY = 2, VIEW_RUN = 3 };

// Container helper exported by the same pinned CRoaring C implementation.
int bitset_container_compute_cardinality(const struct bitset_container_s *);
void bitset_container_free(struct bitset_container_s *);
struct array_container_s *array_container_create_given_capacity(int32_t);
void array_container_free(struct array_container_s *);
size_t bitset_extract_setbits_uint16(const uint64_t *, size_t, uint16_t *,
                                     uint16_t);

static uint16_t read_u16(const char *data) {
  uint16_t value;
  memcpy(&value, data, sizeof(value));
  return value;
}

static uint32_t read_u32(const char *data) {
  uint32_t value;
  memcpy(&value, data, sizeof(value));
  return value;
}

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

static int reserve_buffer(void **buffer, size_t *capacity, size_t size) {
  if (size <= *capacity) return 1;
  void *replacement = roaring_realloc(*buffer, size);
  if (!replacement) return 0;
  *buffer = replacement;
  *capacity = size;
  return 1;
}

const roaring_bitmap_t *zvec_inverted_bitmap_view_load(
    zvec_inverted_bitmap_view *view, const char *data, size_t size) {
  if (!data || size < 4) return NULL;
  const uint32_t cookie = read_u32(data);
  const int has_runs = (cookie & 0xffff) == 12347;
  uint32_t count;
  size_t keys_offset;
  if (has_runs) {
    count = (cookie >> 16) + 1;
    keys_offset = 4 + (count + 7) / 8;
  } else {
    if (cookie != 12346 || size < 8) return NULL;
    count = read_u32(data + 4);
    if (count > 65536) return NULL;
    keys_offset = 8;
  }
  const size_t position_begin =
      keys_offset + (size_t)count * (has_runs && count < 4 ? 4 : 8);
  if (position_begin > size) return NULL;

  // Pointer array and descriptors are independently naturally aligned, even
  // for odd container counts. No per-container allocations are needed.
  const size_t descriptor_offset = (size_t)count * sizeof(void *);
  const size_t keys_storage_offset =
      descriptor_offset + (size_t)count * sizeof(union ViewContainer);
  const size_t metadata_bytes = keys_storage_offset + (size_t)count * 3;
  if (!reserve_buffer(&view->metadata, &view->metadata_capacity,
                      metadata_bytes)) {
    return NULL;
  }
  roaring_array_t *array = &view->bitmap.high_low_container;
  array->size = 0;
  array->allocation_size = (int32_t)count;
  array->flags = ROARING_FLAG_FROZEN;
  // The empty portable bitmap needs no metadata storage.
  if (count == 0) {
    array->containers = NULL;
    array->keys = NULL;
    array->typecodes = NULL;
    return &view->bitmap;
  }
  array->containers = (void **)view->metadata;
  union ViewContainer *containers =
      (union ViewContainer *)((char *)view->metadata + descriptor_offset);
  array->keys = (uint16_t *)((char *)view->metadata + keys_storage_offset);
  array->typecodes = (uint8_t *)(array->keys + count);

  // A single pass validates sequential container bounds and builds aligned
  // descriptors. The caller only sees the view after every bound has passed.
  // Reserve payload storage lazily on the first unaligned container. Bounding
  // it by the source payload plus padding avoids reallocating earlier copies.
  const size_t padding = (size_t)count * (sizeof(uint64_t) - 1);
  const size_t payload_bytes = size - position_begin;
  const int bulk_copy =
      (uintptr_t)(data + position_begin) % sizeof(uint16_t) != 0;
  if (payload_bytes > (SIZE_MAX - padding - 7) / 2) return NULL;
  const size_t payload_capacity =
      (bulk_copy ? payload_bytes * 2 + 7 : payload_bytes) + padding;
  size_t position = position_begin;
  size_t copy_bytes = 0;
  if (bulk_copy && payload_bytes) {
    if (!reserve_buffer(&view->payload, &view->payload_capacity,
                        payload_capacity)) {
      return NULL;
    }
    // All portable payloads occupy an even number of bytes. One copy aligns
    // every array/run payload, replacing a memcpy for each sparse container.
    // Internally unaligned bitsets can still use the reserved tail below.
    memcpy(view->payload, data + position_begin, payload_bytes);
    copy_bytes = align_up(payload_bytes, sizeof(uint64_t));
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t cardinality = read_u16(data + keys_offset + i * 4 + 2) + 1;
    const int is_run =
        has_runs && ((unsigned char)data[4 + i / 8] & (1u << (i % 8)));
    uint16_t runs = 0;
    size_t bytes;
    size_t alignment;
    if (is_run) {
      if (size - position < 2) return NULL;
      runs = read_u16(data + position);
      position += 2;
      bytes = (size_t)runs * sizeof(struct rle16_s);
      alignment = sizeof(uint16_t);
    } else {
      bytes = cardinality > 4096 ? 8192 : cardinality * sizeof(uint16_t);
      alignment = cardinality > 4096 ? sizeof(uint64_t) : sizeof(uint16_t);
    }
    if (bytes > size - position) return NULL;
    const char *payload =
        bulk_copy && payload_bytes
            ? (const char *)view->payload + position - position_begin
            : data + position;
    if (bytes && (uintptr_t)payload % alignment != 0) {
      if (!reserve_buffer(&view->payload, &view->payload_capacity,
                          payload_capacity)) {
        return NULL;
      }
      copy_bytes = align_up(copy_bytes, alignment);
      char *copy = (char *)view->payload + copy_bytes;
      memcpy(copy, payload, bytes);
      payload = copy;
      copy_bytes += bytes;
    }
    array->keys[i] = read_u16(data + keys_offset + i * 4);
    if (is_run) {
      containers[i].run.n_runs = runs;
      containers[i].run.capacity = runs;
      containers[i].run.runs = (struct rle16_s *)payload;
      array->containers[i] = &containers[i].run;
      array->typecodes[i] = VIEW_RUN;
    } else if (cardinality > 4096) {
      containers[i].bitset.cardinality = (int32_t)cardinality;
      containers[i].bitset.words = (uint64_t *)payload;
      array->containers[i] = &containers[i].bitset;
      array->typecodes[i] = VIEW_BITSET;
    } else {
      containers[i].array.cardinality = (int32_t)cardinality;
      containers[i].array.capacity = (int32_t)cardinality;
      containers[i].array.array = (uint16_t *)payload;
      array->containers[i] = &containers[i].array;
      array->typecodes[i] = VIEW_ARRAY;
    }
    position += bytes;
  }
  array->size = (int32_t)count;
  return &view->bitmap;
}

void zvec_inverted_bitmap_view_clear(zvec_inverted_bitmap_view *view) {
  roaring_free(view->metadata);
  roaring_free(view->payload);
  memset(view, 0, sizeof(*view));
}

#if defined(ZVEC_BITMAP_X86_TARGETS)
__attribute__((target("avx512f,avx512vpopcntdq"))) static int32_t
merge_bitset_avx512(uint64_t *words, const char *payload, int subtract,
                    int lazy) {
  if (lazy) {
    for (size_t j = 0; j < 1024; j += 32) {
      const __m512i a = _mm512_loadu_si512(payload + j * sizeof(uint64_t));
      const __m512i b =
          _mm512_loadu_si512(payload + (j + 8) * sizeof(uint64_t));
      const __m512i c =
          _mm512_loadu_si512(payload + (j + 16) * sizeof(uint64_t));
      const __m512i d =
          _mm512_loadu_si512(payload + (j + 24) * sizeof(uint64_t));
      _mm512_storeu_si512(words + j,
                          _mm512_or_si512(a, _mm512_loadu_si512(words + j)));
      _mm512_storeu_si512(
          words + j + 8, _mm512_or_si512(b, _mm512_loadu_si512(words + j + 8)));
      _mm512_storeu_si512(
          words + j + 16,
          _mm512_or_si512(c, _mm512_loadu_si512(words + j + 16)));
      _mm512_storeu_si512(
          words + j + 24,
          _mm512_or_si512(d, _mm512_loadu_si512(words + j + 24)));
    }
    return -1;
  }
  __m512i count = _mm512_setzero_si512();
  for (size_t j = 0; j < 1024; j += 8) {
    const __m512i source = _mm512_loadu_si512(payload + j * sizeof(uint64_t));
    const __m512i target = _mm512_loadu_si512(words + j);
    const __m512i result = subtract ? _mm512_andnot_si512(source, target)
                                    : _mm512_or_si512(source, target);
    _mm512_storeu_si512(words + j, result);
    count = _mm512_add_epi64(count, _mm512_popcnt_epi64(result));
  }
  return (int32_t)_mm512_reduce_add_epi64(count);
}

__attribute__((target("avx2"))) static void merge_bitset_avx2(
    uint64_t *words, const char *payload, int subtract) {
  for (size_t j = 0; j < 1024; j += 4) {
    const __m256i source =
        _mm256_loadu_si256((const __m256i_u *)(payload + j * sizeof(uint64_t)));
    const __m256i target = _mm256_loadu_si256((const __m256i_u *)(words + j));
    const __m256i result = subtract ? _mm256_andnot_si256(source, target)
                                    : _mm256_or_si256(source, target);
    _mm256_storeu_si256((__m256i_u *)(words + j), result);
  }
}

// Adapted from CRoaring 2.0.4's Apache-licensed bitset_clear_list loop,
// Copyright 2016-2022 The CRoaring authors. Byte pointers avoid creating an
// unaligned uint16_t pointer; all loads are bounded by the preflight below.
static int32_t clear_array_x86(uint64_t *words, const char *list,
                               uint32_t length, int32_t cardinality) {
  uint64_t offset, load, pos;
  const uint64_t shift = 6;
  uint64_t card = (uint64_t)cardinality;
  const char *end = list + (size_t)length * 2;
  __asm__ volatile(
      "1:\n"
      "movzwq (%[list]), %[pos]\n"
      "shrx %[shift], %[pos], %[offset]\n"
      "mov (%[words],%[offset],8), %[load]\n"
      "btr %[pos], %[load]\n"
      "mov %[load], (%[words],%[offset],8)\n"
      "sbb $0, %[card]\n"
      "add $2, %[list]\n"
      "cmp %[list], %[end]\n"
      "jnz 1b"
      : [card] "+&r"(card), [list] "+&r"(list), [load] "=&r"(load),
        [pos] "=&r"(pos), [offset] "=&r"(offset)
      : [end] "r"(end), [words] "r"(words), [shift] "r"(shift)
      : "cc", "memory");
  return (int32_t)card;
}
#endif

// Keep the instruction dispatch inside the existing dense loop when this
// helper is also used by deferred subtraction.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
static inline int32_t merge_bitset_bytes(struct bitset_container_s *target,
                                        const char *payload, int subtract,
                                        int lazy) {
#if defined(ZVEC_BITMAP_X86_TARGETS)
  if (__builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512vpopcntdq")) {
    return merge_bitset_avx512(target->words, payload, subtract, lazy);
  }
  if (__builtin_cpu_supports("avx2")) {
    merge_bitset_avx2(target->words, payload, subtract);
    return lazy ? -1 : bitset_container_compute_cardinality(target);
  }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
  uint16x8_t count = vdupq_n_u16(0);
  for (size_t j = 0; j < 8192; j += 16) {
    const uint8x16_t source = vld1q_u8((const uint8_t *)payload + j);
    const uint8x16_t before = vld1q_u8((const uint8_t *)target->words + j);
    const uint8x16_t after =
        subtract ? vbicq_u8(before, source) : vorrq_u8(before, source);
    vst1q_u8((uint8_t *)target->words + j, after);
    if (!lazy) count = vaddq_u16(count, vpaddlq_u8(vcntq_u8(after)));
  }
  return lazy ? -1 : (int32_t)vaddlvq_u16(count);
#else
  if (subtract) {
    for (size_t j = 0; j < 1024; ++j) {
      uint64_t word;
      memcpy(&word, payload + j * sizeof(word), sizeof(word));
      target->words[j] &= ~word;
    }
  } else {
    for (size_t j = 0; j < 1024; ++j) {
      uint64_t word;
      memcpy(&word, payload + j * sizeof(word), sizeof(word));
      target->words[j] |= word;
    }
  }
  return lazy ? -1 : bitset_container_compute_cardinality(target);
#endif
}

int zvec_inverted_bitmap_has_only_bitsets(const roaring_bitmap_t *bitmap) {
  const roaring_array_t *array = &bitmap->high_low_container;
  for (int32_t i = 0; i < array->size; ++i) {
    if (array->typecodes[i] != VIEW_BITSET) return 0;
  }
  return array->size != 0;
}

int zvec_inverted_bitmap_try_merge_dense(roaring_bitmap_t *bitmap,
                                         const char *data, size_t size,
                                         int subtract, int lazy) {
  if (!bitmap || !data || size < 8 || read_u32(data) != 12346) return 0;
  roaring_array_t *array = &bitmap->high_low_container;
  const uint32_t count = read_u32(data + 4);
  if (!count || count > 65536 || count != (uint32_t)array->size ||
      (array->flags & (ROARING_FLAG_COW | ROARING_FLAG_FROZEN))) {
    return 0;
  }
  size_t position = 8 + (size_t)count * 8;
  if (position > size) return 0;
  // Validate the entire source and destination before making any writes.
  // ANDNOT stays on this path only when every result remains a bitset; the
  // general reader handles shrinking, empty containers and shared storage.
  for (uint32_t i = 0; i < count; ++i) {
    if (array->typecodes[i] != VIEW_BITSET ||
        array->keys[i] != read_u16(data + 8 + i * 4)) {
      return 0;
    }
    const uint32_t cardinality = read_u16(data + 8 + i * 4 + 2) + 1u;
    const size_t bytes = cardinality > 4096 ? 8192 : cardinality * 2;
    if (bytes > size - position) return 0;
    const struct bitset_container_s *target = array->containers[i];
    if ((subtract && target->cardinality <= (int32_t)cardinality + 4096) ||
        (!lazy && target->cardinality < 0)) {
      return 0;
    }
    position += bytes;
  }

  position = 8 + (size_t)count * 8;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t cardinality = read_u16(data + 8 + i * 4 + 2) + 1u;
    struct bitset_container_s *target = array->containers[i];
    uint64_t *words = target->words;
    const char *payload = data + position;
    if (!subtract && target->cardinality == 65536) {
      position += cardinality > 4096 ? 8192 : cardinality * 2;
      continue;
    }
    if (cardinality > 4096) {
      target->cardinality =
          merge_bitset_bytes(target, payload, subtract, lazy && !subtract);
      position += 8192;
    } else {
      int32_t change = 0;
      if (subtract) {
#if defined(ZVEC_BITMAP_X86_TARGETS)
        if (__builtin_cpu_supports("bmi2")) {
          target->cardinality =
              clear_array_x86(words, payload, cardinality, target->cardinality);
        } else
#endif
        {
          for (uint32_t j = 0; j < cardinality; ++j) {
            const uint16_t value = read_u16(payload + j * 2);
            const uint64_t mask = UINT64_C(1) << (value % 64);
            const uint64_t before = words[value / 64];
            change += (before & mask) != 0;
            words[value / 64] = before & ~mask;
          }
          target->cardinality -= change;
        }
      } else if (lazy) {
        for (uint32_t j = 0; j < cardinality; ++j) {
          const uint16_t value = read_u16(payload + j * 2);
          words[value / 64] |= UINT64_C(1) << (value % 64);
        }
        target->cardinality = -1;
      } else {
        for (uint32_t j = 0; j < cardinality; ++j) {
          const uint16_t value = read_u16(payload + j * 2);
          const uint64_t mask = UINT64_C(1) << (value % 64);
          const uint64_t before = words[value / 64];
          change += (before & mask) == 0;
          words[value / 64] = before | mask;
        }
        target->cardinality += change;
      }
      position += cardinality * 2;
    }
  }
  return 1;
}

enum { BITSET_WORDS = 1024, MAX_BATCH_ARRAYS = 256 };

int zvec_inverted_bitmap_defer_subtract(zvec_inverted_bitmap_subtract *scratch,
                                        roaring_bitmap_t *bitmap,
                                        const char *data, size_t size) {
  if (!data || size < 8 || read_u32(data) != 12346) return 0;
  roaring_array_t *array = &bitmap->high_low_container;
  const uint32_t count = read_u32(data + 4);
  if (!count || count > 65536 || count != (uint32_t)array->size ||
      (array->flags & (ROARING_FLAG_COW | ROARING_FLAG_FROZEN))) {
    return 0;
  }

  const size_t payload_begin = 8 + (size_t)count * 8;
  if (payload_begin > size) return 0;
  size_t position = payload_begin;
  size_t array_count = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (array->keys[i] != read_u16(data + 8 + i * 4)) return 0;
    const uint8_t type = array->typecodes[i];
    if (type == VIEW_ARRAY) {
      ++array_count;
    } else {
      if (type != VIEW_BITSET) return 0;
      const struct bitset_container_s *target = array->containers[i];
      if (target->cardinality < 0) return 0;
    }
    const uint32_t cardinality = read_u16(data + 10 + i * 4) + 1u;
    const size_t bytes = cardinality > 4096 ? 8192 : cardinality * 2;
    if (bytes > size - position) return 0;
    position += bytes;
  }
  // Keep the existing dense path for all-bitset destinations and cap the
  // temporary storage for sparse containers at 2 MiB per query.
  if (!array_count || array_count == count || array_count > MAX_BATCH_ARRAYS) {
    return 0;
  }

  if (!scratch->array_words) {
    const size_t bytes = array_count * BITSET_WORDS * sizeof(uint64_t);
    scratch->array_words = roaring_aligned_malloc(64, bytes);
    if (!scratch->array_words) return 0;
    memset(scratch->array_words, 0, bytes);
    uint64_t *words = scratch->array_words;
    for (uint32_t i = 0; i < count; ++i) {
      if (array->typecodes[i] != VIEW_ARRAY) continue;
      const struct array_container_s *source = array->containers[i];
      for (int32_t j = 0; j < source->cardinality; ++j) {
        const uint16_t value = source->array[j];
        words[value / 64] |= UINT64_C(1) << (value % 64);
      }
      words += BITSET_WORDS;
    }
  }

  // Keep the original sparse arrays until the end. Their scratch bitsets
  // avoid scanning the larger destination array for every small posting.
  uint64_t *array_words = scratch->array_words;
  position = payload_begin;
  for (uint32_t i = 0; i < count; ++i) {
    struct bitset_container_s array_mask;
    struct bitset_container_s *target = array->containers[i];
    if (array->typecodes[i] == VIEW_ARRAY) {
      const struct array_container_s *source = array->containers[i];
      array_mask.cardinality = source->cardinality;
      array_mask.words = array_words;
      target = &array_mask;
      array_words += BITSET_WORDS;
    }
    const uint32_t cardinality = read_u16(data + 10 + i * 4) + 1u;
    const char *payload = data + position;
    if (cardinality > 4096) {
      target->cardinality = merge_bitset_bytes(target, payload, 1, 0);
      position += 8192;
      continue;
    }
    position += cardinality * 2;
#if defined(ZVEC_BITMAP_X86_TARGETS)
    if (__builtin_cpu_supports("bmi2")) {
      target->cardinality = clear_array_x86(
          target->words, payload, cardinality, target->cardinality);
      continue;
    }
#endif
    int32_t removed = 0;
    for (uint32_t j = 0; j < cardinality; ++j) {
      const uint16_t value = read_u16(payload + j * 2);
      const uint64_t mask = UINT64_C(1) << (value % 64);
      removed += (target->words[value / 64] & mask) != 0;
      target->words[value / 64] &= ~mask;
    }
    target->cardinality -= removed;
  }
  return 1;
}

int zvec_inverted_bitmap_finish_subtract(zvec_inverted_bitmap_subtract *scratch,
                                         roaring_bitmap_t *bitmap) {
  if (!scratch->array_words) return 1;
  roaring_array_t *array = &bitmap->high_low_container;
  const uint64_t *words = scratch->array_words;
  for (int32_t i = 0; i < array->size; ++i) {
    if (array->typecodes[i] == VIEW_ARRAY) {
      struct array_container_s *target = array->containers[i];
      int32_t out = 0;
      for (int32_t j = 0; j < target->cardinality; ++j) {
        const uint16_t value = target->array[j];
        target->array[out] = value;
        out += (words[value / 64] >> (value % 64)) & 1;
      }
      target->cardinality = out;
      words += BITSET_WORDS;
    } else {
      struct bitset_container_s *bits = array->containers[i];
      const int32_t cardinality = bits->cardinality;
      if (cardinality == 0 || cardinality > 4096) continue;
      struct array_container_s *target =
          array_container_create_given_capacity(cardinality);
      if (!target) return 0;
      target->cardinality = (int32_t)bitset_extract_setbits_uint16(
          bits->words, BITSET_WORDS, target->array, 0);
      bitset_container_free(bits);
      array->containers[i] = target;
      array->typecodes[i] = VIEW_ARRAY;
    }
  }

  // Remove empty containers after all allocations succeed, so even a partly
  // normalized bitmap can be freed safely on allocation failure.
  int32_t out = 0;
  for (int32_t i = 0; i < array->size; ++i) {
    if (array->typecodes[i] == VIEW_ARRAY) {
      struct array_container_s *target = array->containers[i];
      if (!target->cardinality) {
        array_container_free(target);
        continue;
      }
    } else {
      struct bitset_container_s *target = array->containers[i];
      if (!target->cardinality) {
        bitset_container_free(target);
        continue;
      }
    }
    array->containers[out] = array->containers[i];
    array->typecodes[out] = array->typecodes[i];
    array->keys[out] = array->keys[i];
    ++out;
  }
  array->size = out;
  roaring_aligned_free(scratch->array_words);
  scratch->array_words = NULL;
  return 1;
}
