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

// Private header defining the io_uring kernel ABI structures and constants.
//
// This header is the io_uring analogue of libaio_def.h: it declares *only*
// the types, constants, and inline helpers that zvec needs from the io_uring
// kernel interface.  By defining these structures ourselves we avoid any
// build-time dependency on <linux/io_uring.h> or liburing-dev, mirroring the
// project's zero-dependency philosophy established by the libaio dlopen
// approach.
//
// The struct layouts (io_uring_sqe, io_uring_cqe, io_uring_params,
// io_sqring_offsets, io_cqring_offsets) are part of the Linux kernel ABI
// and are copied verbatim from <linux/io_uring.h>.

#pragma once

#include <cstdint>

#if defined(__linux) || defined(__linux__)

// ---------------------------------------------------------------------------
// Syscall numbers
// ---------------------------------------------------------------------------
// io_uring was introduced in Linux 5.1 (2019).  The three syscalls share the
// same numbers across all supported architectures.  We prefer the values
// from <sys/syscall.h> when available and fall back to hardcoded numbers.
#include <sys/syscall.h>

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif

#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif

#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif

// ---------------------------------------------------------------------------
// Constants (from <linux/io_uring.h>)
// ---------------------------------------------------------------------------

// mmap offsets for the three shared regions.
#define IORING_OFF_SQ_RING 0ULL
#define IORING_OFF_CQ_RING 0x8000000ULL
#define IORING_OFF_SQES 0x10000000ULL

// io_uring_enter flags.
#define IORING_ENTER_GETEVENTS (1U << 0)

// io_uring_setup flags (none used by default).
// IORING_SETUP_IOPOLL      (1U << 0)
// IORING_SETUP_SQPOLL      (1U << 1)
// IORING_SETUP_SQ_AFF      (1U << 2)

// SQE opcode values.
#define IORING_OP_NOP 0
#define IORING_OP_READV 1
#define IORING_OP_WRITEV 2
#define IORING_OP_FSYNC 3
#define IORING_OP_READ_FIXED 4
#define IORING_OP_WRITE_FIXED 5
#define IORING_OP_POLL_ADD 6
#define IORING_OP_POLL_REMOVE 7
#define IORING_OP_SYNC_FILE_RANGE 8
#define IORING_OP_SENDMSG 9
#define IORING_OP_RECVMSG 10
#define IORING_OP_TIMEOUT 11
#define IORING_OP_TIMEOUT_REMOVE 12
#define IORING_OP_ACCEPT 13
#define IORING_OP_ASYNC_CANCEL 14
#define IORING_OP_LINK_TIMEOUT 15
#define IORING_OP_CONNECT 16
#define IORING_OP_FALLOCATE 17
#define IORING_OP_OPENAT 18
#define IORING_OP_CLOSE 19
#define IORING_OP_FILES_UPDATE 20
#define IORING_OP_STATX 21
#define IORING_OP_READ 22
#define IORING_OP_WRITE 23

// ---------------------------------------------------------------------------
// Struct definitions (copied verbatim from <linux/io_uring.h>)
// ---------------------------------------------------------------------------

// Submission queue entry — 64 bytes.
struct io_uring_sqe {
  uint8_t opcode;   // type of operation for this sqe
  uint8_t flags;    // IOSQE_ flags
  uint16_t ioprio;  // ioprio for the request
  int32_t fd;       // file descriptor to do IO on
  union {
    uint64_t off;  // offset into file
    uint64_t addr2;
  };
  union {
    uint64_t addr;  // buffer or iovecs
    uint64_t splice_off_in;
  };
  uint32_t len;  // buffer size or number of iovecs
  union {
    uint32_t rw_flags;  // read/write flags (union of all flag types)
  };
  uint64_t user_data;  // data to be passed back at completion time
  union {
    struct {
      uint16_t buf_index;  // index into fixed buffers, if used
      uint16_t personality;
    };
    uint64_t __pad2[3];
  };
};

// Completion queue entry — 16 bytes.
struct io_uring_cqe {
  uint64_t user_data;  // sqe->user_data
  int32_t res;         // result code for this event
  uint32_t flags;
};

// SQ ring offsets — returned by io_uring_setup in io_uring_params.
struct io_sqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t flags;
  uint32_t dropped;
  uint32_t array;
  uint32_t resv1;
  uint64_t resv2;
};

// CQ ring offsets — returned by io_uring_setup in io_uring_params.
struct io_cqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t overflow;
  uint32_t cqes;
  uint32_t flags;
  uint32_t resv1;
  uint64_t resv2;
};

// Parameters passed to io_uring_setup().
struct io_uring_params {
  uint32_t sq_entries;
  uint32_t cq_entries;
  uint32_t flags;
  uint32_t sq_thread_cpu;
  uint32_t sq_thread_idle;
  uint32_t features;
  uint32_t wq_fd;
  uint32_t resv[3];
  struct io_sqring_offsets sq_off;
  struct io_cqring_offsets cq_off;
};

// ---------------------------------------------------------------------------
// Inline helper — prepare an SQE for a read operation.
// ---------------------------------------------------------------------------
static inline void io_uring_prep_read(struct io_uring_sqe *sqe, int fd,
                                      void *buf, uint32_t nbytes,
                                      uint64_t offset) {
  sqe->opcode = IORING_OP_READ;
  sqe->flags = 0;
  sqe->ioprio = 0;
  sqe->fd = fd;
  sqe->off = offset;
  sqe->addr = reinterpret_cast<uint64_t>(buf);
  sqe->len = nbytes;
  sqe->rw_flags = 0;
  sqe->user_data = 0;
  sqe->buf_index = 0;
  sqe->personality = 0;
}

// ---------------------------------------------------------------------------
// End: struct and constant definitions from <linux/io_uring.h>
// ---------------------------------------------------------------------------

#endif  // __linux__
