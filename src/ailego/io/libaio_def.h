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

// Private replacement for <libaio.h>.
//
// This header declares *only* the types, constants, and inline helpers that
// zvec needs from libaio.  By doing so the project is completely decoupled
// from the system libaio-dev header: there is no `#include <libaio.h>` anywhere
// in the source tree, which means libaio-dev does not need to be installed at
// build time and the code is portable to cross-compilation environments that
// lack the header.
//
// The struct layouts (struct iocb, struct io_event, ...) are part of the Linux
// kernel ABI.  They are copied verbatim from the upstream <libaio.h>, including
// the PADDED macros that handle architecture-specific padding.  The inline
// helper io_prep_pread() is likewise copied — it only manipulates struct fields
// and does not call into the library.

#pragma once

#include <time.h>   // struct timespec (used by io_getevents signature)
#include <cstring>  // memset() — used by io_prep_pread() inline helper

#if defined(__linux) || defined(__linux__)

struct sockaddr;
struct iovec;

// ---------------------------------------------------------------------------
// Type and struct definitions copied from <libaio.h>
// ---------------------------------------------------------------------------

typedef struct io_context *io_context_t;

typedef enum io_iocb_cmd {
  IO_CMD_PREAD = 0,
  IO_CMD_PWRITE = 1,
  IO_CMD_FSYNC = 2,
  IO_CMD_FDSYNC = 3,
  IO_CMD_POLL = 5,
  IO_CMD_NOOP = 6,
  IO_CMD_PREADV = 7,
  IO_CMD_PWRITEV = 8,
} io_iocb_cmd_t;

// PADDED macros — copied verbatim from <libaio.h> to guarantee ABI-compatible
// struct layout on every supported architecture.

/* little endian, 32 bits */
#if defined(__i386__) || (defined(__x86_64__) && defined(__ILP32__)) ||     \
    (defined(__arm__) && !defined(__ARMEB__)) ||                            \
    (defined(__sh__) && defined(__LITTLE_ENDIAN__)) || defined(__bfin__) || \
    (defined(__MIPSEL__) && !defined(__mips64)) || defined(__cris__) ||     \
    defined(__loongarch32) || (defined(__riscv) && __riscv_xlen == 32) ||   \
    (defined(__GNUC__) && defined(__BYTE_ORDER__) &&                        \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && __SIZEOF_LONG__ == 4)
#define AIO_PADDED(x, y) \
  x;                     \
  unsigned y
#define AIO_PADDEDptr(x, y) \
  x;                        \
  unsigned y
#define AIO_PADDEDul(x, y) \
  unsigned long x;         \
  unsigned y

/* little endian, 64 bits */
#elif defined(__ia64__) || defined(__x86_64__) || defined(__alpha__) ||   \
    (defined(__mips64) && defined(__MIPSEL__)) ||                         \
    (defined(__aarch64__) && defined(__AARCH64EL__)) ||                   \
    defined(__loongarch64) || (defined(__riscv) && __riscv_xlen == 64) || \
    (defined(__GNUC__) && defined(__BYTE_ORDER__) &&                      \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && __SIZEOF_LONG__ == 8)
#define AIO_PADDED(x, y) x, y
#define AIO_PADDEDptr(x, y) x
#define AIO_PADDEDul(x, y) unsigned long x

/* big endian, 64 bits */
#elif defined(__powerpc64__) || defined(__s390x__) ||   \
    (defined(__hppa__) && defined(__arch64__)) ||       \
    (defined(__sparc__) && defined(__arch64__)) ||      \
    (defined(__mips64) && defined(__MIPSEB__)) ||       \
    (defined(__aarch64__) && defined(__AARCH64EB__)) || \
    (defined(__GNUC__) && defined(__BYTE_ORDER__) &&    \
     __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ && __SIZEOF_LONG__ == 8)
#define AIO_PADDED(x, y) \
  unsigned y;            \
  x
#define AIO_PADDEDptr(x, y) x
#define AIO_PADDEDul(x, y) unsigned long x

/* big endian, 32 bits */
#elif defined(__PPC__) || defined(__s390__) ||                            \
    (defined(__arm__) && defined(__ARMEB__)) ||                           \
    (defined(__sh__) && defined(__BIG_ENDIAN__)) || defined(__sparc__) || \
    defined(__MIPSEB__) || defined(__m68k__) || defined(__hppa__) ||      \
    defined(__frv__) || defined(__avr32__) ||                             \
    (defined(__GNUC__) && defined(__BYTE_ORDER__) &&                      \
     __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ && __SIZEOF_LONG__ == 4)
#define AIO_PADDED(x, y) \
  unsigned y;            \
  x
#define AIO_PADDEDptr(x, y) \
  unsigned y;               \
  x
#define AIO_PADDEDul(x, y) \
  unsigned y;              \
  unsigned long x

#else
#error endianness?
#endif

struct io_iocb_poll {
  AIO_PADDED(int events, __pad1);
};

struct io_iocb_sockaddr {
  AIO_PADDEDptr(struct sockaddr *addr, __pad1);
  AIO_PADDEDul(len, __pad2);
};

struct io_iocb_common {
  AIO_PADDEDptr(void *buf, __pad1);
  AIO_PADDEDul(nbytes, __pad2);
  long long offset;
  long long __pad3;
  unsigned flags;
  unsigned resfd;
};

struct io_iocb_vector {
  AIO_PADDEDptr(const struct iovec *vec, __pad1);
  AIO_PADDEDul(nr, __pad2);
  long long offset;
};

struct iocb {
  AIO_PADDEDptr(void *data, __pad1);
  AIO_PADDED(unsigned key, aio_rw_flags);
  short aio_lio_opcode;
  short aio_reqprio;
  int aio_fildes;
  union {
    struct io_iocb_common c;
    struct io_iocb_vector v;
    struct io_iocb_poll poll;
    struct io_iocb_sockaddr saddr;
  } u;
};

struct io_event {
  AIO_PADDEDptr(void *data, __pad1);
  AIO_PADDEDptr(struct iocb *obj, __pad2);
  AIO_PADDEDul(res, __pad3);
  AIO_PADDEDul(res2, __pad4);
};

#undef AIO_PADDED
#undef AIO_PADDEDptr
#undef AIO_PADDEDul

// Inline helper — copied from <libaio.h>.  Only manipulates struct fields.
static inline void io_prep_pread(struct iocb *iocb, int fd, void *buf,
                                 size_t count, long long offset) {
  memset(iocb, 0, sizeof(*iocb));
  iocb->aio_fildes = fd;
  iocb->aio_lio_opcode = IO_CMD_PREAD;
  iocb->aio_reqprio = 0;
  iocb->u.c.buf = buf;
  iocb->u.c.nbytes = count;
  iocb->u.c.offset = offset;
}

// ---------------------------------------------------------------------------
// End: type and struct definitions from <libaio.h>
// ---------------------------------------------------------------------------

#endif  // __linux__
