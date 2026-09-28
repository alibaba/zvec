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

/* Test-process-only syscall shim for crash recovery tests; never linked into
 * zvec. Loaded with DYLD_INSERT_LIBRARIES (macOS) or LD_PRELOAD (Linux).
 *
 * Two independent features, both limited to paths below ZVEC_SHIM_ROOT:
 *
 * 1. Fault injection. When ZVEC_FAULT_ARM names an existing file, calls of
 *    kind ZVEC_FAULT_OPERATION ("write", the default, or "sync") on paths
 *    containing ZVEC_FAULT_PATH fail with ZVEC_FAULT_ERRNO (default EIO);
 *    with ZVEC_FAULT_ONCE only the first matching call fails. Every injection
 *    is appended to ZVEC_FAULT_AUDIT.
 *
 * 2. Sync recording for simulated power loss. With ZVEC_SYNC_SHADOW set,
 *    each successful barrier appends one line to <shadow>/log, in order:
 *      F <inode> <seq>          fsync/fdatasync/F_FULLFSYNC/F_BARRIERFSYNC
 *                               of a file: <shadow>/<seq> holds its contents
 *      R <inode> <offset> <size> <seq>
 *                               msync(MS_SYNC): <shadow>/<seq> holds the
 *                               page-rounded file range of the mapping and
 *                               <size> is the file size, which a range sync
 *                               also persists
 *      D <inode> <seq>          fsync of a directory: <shadow>/<seq> holds
 *                               lines "<f|d> <inode> <name>"
 *    and for fsync failures injected by feature 1:
 *      E <inode> <size>         the failed fsync; the file size at that time
 *      W <inode> <offset> <len> a later successful write to that inode
 *      T <inode> <size>         a later successful truncate of that inode
 *    Nothing else (close, rename, msync(MS_ASYNC), sync_file_range) counts as
 *    durable. Contents are copied right after the barrier returns, so a write
 *    from another thread in that window is attributed to the barrier.
 *
 *    Identity: an inode number is logged only while the shim holds an open
 *    descriptor on that inode (verified with fstat), kept until exit, so the
 *    number cannot be reused by another file during the run. Contents are
 *    read through that descriptor, never through a path. When no verified
 *    descriptor can be obtained, nothing is recorded, i.e. the data is
 *    treated as not persisted.
 *
 * Load check: when ZVEC_SHIM_HANDSHAKE is set, the constructor creates that
 * file, so a caller can prove the shim was loaded (macOS silently drops
 * DYLD_INSERT_LIBRARIES for protected binaries).
 */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
extern int fdatasync(int);
#endif

static const char *root;
static size_t root_len;
static const char *arm;
static const char *audit;
static const char *fault_path;
static int fault_errno = EIO;
static int fault_sync;
static int fault_once;
static long fault_count;
static const char *shadow;
static long sequence;
/* Guards the record log and the tables; never taken recursively. */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- raw calls that are never routed back into this shim ---------------- */

#if defined(__APPLE__)
/* Calls made from the interposing image itself are not interposed. */
#define RAW_WRITE(fd, buf, n) write(fd, buf, n)
#define RAW_PREAD(fd, buf, n, off) pread(fd, buf, n, off)
#define RAW_OPEN(path, flags, mode) open(path, flags, mode)
#define RAW_CLOSE(fd) close(fd)
#define RAW_FCNTL(fd, cmd) fcntl(fd, cmd)
#else
#define RAW_WRITE(fd, buf, n) syscall(SYS_write, fd, buf, n)
#define RAW_PREAD(fd, buf, n, off) syscall(SYS_pread64, fd, buf, n, off)
#define RAW_OPEN(path, flags, mode) \
  syscall(SYS_openat, AT_FDCWD, path, flags, mode)
#define RAW_CLOSE(fd) syscall(SYS_close, fd)
#define RAW_FCNTL(fd, cmd) syscall(SYS_fcntl, fd, cmd, 0)
#endif

static void die(const char *message) {
  static const char prefix[] = "storage_shim: fatal: ";
  (void)!RAW_WRITE(STDERR_FILENO, prefix, sizeof(prefix) - 1);
  (void)!RAW_WRITE(STDERR_FILENO, message, strlen(message));
  (void)!RAW_WRITE(STDERR_FILENO, "\n", 1);
  _exit(125);
}

#if defined(__linux__)
/* Real functions, resolved once before first use. */
static ssize_t (*real_write)(int, const void *, size_t);
static ssize_t (*real_pwrite)(int, const void *, size_t, off_t);
static ssize_t (*real_pwrite64)(int, const void *, size_t, off64_t);
static ssize_t (*real_writev)(int, const struct iovec *, int);
static int (*real_fsync)(int);
static int (*real_fdatasync)(int);
static int (*real_msync)(void *, size_t, int);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static void *(*real_mmap64)(void *, size_t, int, int, int, off64_t);
static void *(*real_mremap)(void *, size_t, size_t, int, ...);
static int (*real_munmap)(void *, size_t);
static int (*real_ftruncate)(int, off_t);
static int (*real_ftruncate64)(int, off64_t);
static pthread_once_t resolved = PTHREAD_ONCE_INIT;

static void *resolve_symbol(const char *name) {
  void *symbol = dlsym(RTLD_NEXT, name);
  if (!symbol) die(name);
  return symbol;
}
static void resolve_all(void) {
  real_write = resolve_symbol("write");
  real_pwrite = resolve_symbol("pwrite");
  real_pwrite64 = resolve_symbol("pwrite64");
  real_writev = resolve_symbol("writev");
  real_fsync = resolve_symbol("fsync");
  real_fdatasync = resolve_symbol("fdatasync");
  real_msync = resolve_symbol("msync");
  real_mmap = resolve_symbol("mmap");
  real_mmap64 = resolve_symbol("mmap64");
  real_mremap = resolve_symbol("mremap");
  real_munmap = resolve_symbol("munmap");
  real_ftruncate = resolve_symbol("ftruncate");
  real_ftruncate64 = resolve_symbol("ftruncate64");
}
#define RESOLVE() pthread_once(&resolved, resolve_all)
#endif

__attribute__((constructor)) static void configure(void) {
#if defined(__linux__)
  RESOLVE();
#endif
  root = getenv("ZVEC_SHIM_ROOT");
  root_len = root ? strlen(root) : 0;
  arm = getenv("ZVEC_FAULT_ARM");
  audit = getenv("ZVEC_FAULT_AUDIT");
  fault_path = getenv("ZVEC_FAULT_PATH");
  const char *value = getenv("ZVEC_FAULT_ERRNO");
  if (value) fault_errno = (int)strtol(value, NULL, 10);
  shadow = getenv("ZVEC_SYNC_SHADOW");
  value = getenv("ZVEC_FAULT_OPERATION");
  fault_sync = value && strcmp(value, "sync") == 0;
  fault_once = getenv("ZVEC_FAULT_ONCE") != NULL;
  if (shadow) {
    /* Pinned descriptors must not exhaust the process's own limit. */
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0) {
      rlim_t wanted = limit.rlim_max;
#if defined(__APPLE__) && defined(OPEN_MAX)
      if (wanted > OPEN_MAX) wanted = OPEN_MAX;
#endif
      if (wanted > limit.rlim_cur) {
        limit.rlim_cur = wanted;
        setrlimit(RLIMIT_NOFILE, &limit);
      }
    }
  }
  const char *handshake = getenv("ZVEC_SHIM_HANDSHAKE");
  if (handshake) {
    char line[64];
    int n = snprintf(line, sizeof(line), "loaded %d\n", (int)getpid());
    int fd = (int)RAW_OPEN(handshake, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || RAW_WRITE(fd, line, (size_t)n) != n) die("handshake");
    RAW_CLOSE(fd);
  }
}

static int under_root(const char *path) {
  return root && path && strncmp(path, root, root_len) == 0 &&
         (path[root_len] == '/' || path[root_len] == '\0');
}

static int fd_path(int fd, char *path) {
#if defined(__APPLE__)
  if (fcntl(fd, F_GETPATH, path) != 0) return -1;
  return 0;
#else
  char link[64];
  snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
  ssize_t size = readlink(link, path, PATH_MAX - 1);
  if (size < 0) return -1;
  path[size] = '\0';
  return 0;
#endif
}

static int fd_under_root(int fd) {
  char path[PATH_MAX];
  return root && fd_path(fd, path) == 0 && under_root(path);
}

static void append_line(const char *file, const char *line, size_t length) {
  int fd = (int)RAW_OPEN(file, O_WRONLY | O_APPEND | O_CREAT, 0600);
  /* A lost record would make the simulation unsound: fail loudly. */
  if (fd < 0 || RAW_WRITE(fd, line, length) != (ssize_t)length)
    die("cannot append record");
  RAW_CLOSE(fd);
}

static void append_log(const char *format, ...) {
  char line[256];
  va_list ap;
  va_start(ap, format);
  int n = vsnprintf(line, sizeof(line), format, ap);
  va_end(ap);
  if (n <= 0 || (size_t)n >= sizeof(line)) die("record too long");
  char log[PATH_MAX];
  snprintf(log, sizeof(log), "%s/log", shadow);
  append_line(log, line, (size_t)n);
}

/* ---- pinned inodes (caller holds lock) ----------------------------------- */

/* `fd` is an open, readable descriptor on `inode`, kept until exit. */
struct pinned_inode {
  unsigned long long inode;
  int fd;
};
#define MAX_PINNED 65536
static struct pinned_inode pinned[MAX_PINNED];
static int pinned_count;

static int find_pinned(unsigned long long inode) {
  for (int i = 0; i < pinned_count; ++i)
    if (pinned[i].inode == inode) return pinned[i].fd;
  return -1;
}

/* Adopts `fd` as the pin for its own inode (closing it when one exists) and
 * returns the pinned descriptor and its inode, or -1. */
static int adopt(int fd, unsigned long long *inode) {
  struct stat st;
  if (fd < 0) return -1;
  if (fstat(fd, &st) != 0) {
    RAW_CLOSE(fd);
    return -1;
  }
  *inode = (unsigned long long)st.st_ino;
  int existing = find_pinned(*inode);
  if (existing >= 0) {
    RAW_CLOSE(fd);
    return existing;
  }
  if (pinned_count == MAX_PINNED) die("too many pinned inodes");
  pinned[pinned_count].inode = *inode;
  pinned[pinned_count].fd = fd;
  ++pinned_count;
  return fd;
}

/* A buffered read-only descriptor, opened independently of `fd` (so none of
 * the application's flags such as O_DIRECT apply) and verified to refer to
 * the same inode. Returns -1 when none can be opened. */
static int pin_fd(int fd, unsigned long long *inode) {
  struct stat st;
  if (fstat(fd, &st) != 0) return -1;
  *inode = (unsigned long long)st.st_ino;
  int existing = find_pinned(*inode);
  if (existing >= 0) return existing;
  char path[PATH_MAX];
  if (fd_path(fd, path) != 0) return -1;
  unsigned long long opened = 0;
  int result = adopt((int)RAW_OPEN(path, O_RDONLY | O_CLOEXEC, 0), &opened);
  return result >= 0 && opened == *inode ? result : -1;
}

/* ---- fault injection ----------------------------------------------------- */

/* Inodes with an injected fsync failure; later writes to them are logged. */
#define MAX_FAILED 64
static unsigned long long failed[MAX_FAILED];
static int failed_count;

static int failed_inode(int fd, unsigned long long *inode) {
  if (!shadow || __atomic_load_n(&failed_count, __ATOMIC_ACQUIRE) == 0)
    return 0;
  struct stat st;
  if (fstat(fd, &st) != 0) return 0;
  *inode = (unsigned long long)st.st_ino;
  pthread_mutex_lock(&lock);
  int found = 0;
  for (int i = 0; i < failed_count && !found; ++i) found = failed[i] == *inode;
  pthread_mutex_unlock(&lock);
  return found;
}

static int inject_kind(int fd, int sync) {
  if (sync != fault_sync) return 0;
  if (!arm || !fault_path || access(arm, F_OK) != 0) return 0;
  char path[PATH_MAX];
  if (fd_path(fd, path) != 0 || !under_root(path) ||
      !strstr(path + root_len, fault_path))
    return 0;
  if (__atomic_add_fetch(&fault_count, 1, __ATOMIC_SEQ_CST) > 1 && fault_once)
    return 0;
  if (audit) {
    char line[PATH_MAX + 64];
    int n = snprintf(line, sizeof(line), "INJECT %s errno=%d %s\n",
                     sync ? "sync" : "write", fault_errno, path);
    if (n > 0 && (size_t)n < sizeof(line)) append_line(audit, line, (size_t)n);
  }
  if (sync && shadow) {
    pthread_mutex_lock(&lock);
    unsigned long long inode = 0;
    struct stat st;
    /* The failure must be modelled: without a pin, stop the run. */
    if (pin_fd(fd, &inode) < 0 || fstat(fd, &st) != 0)
      die("cannot pin inode with a failed fsync");
    if (failed_count == MAX_FAILED) die("too many failed fsyncs");
    failed[failed_count] = inode;
    __atomic_store_n(&failed_count, failed_count + 1, __ATOMIC_RELEASE);
    append_log("E %llu %lld\n", inode, (long long)st.st_size);
    pthread_mutex_unlock(&lock);
  }
  errno = fault_errno;
  return 1;
}

static int inject(int fd) {
  return inject_kind(fd, 0);
}
static int inject_sync(int fd) {
  return inject_kind(fd, 1);
}

/* Offset a write() on `fd` will use, or -1. */
static off_t write_offset(int fd) {
  int flags = (int)RAW_FCNTL(fd, F_GETFL);
  if (flags != -1 && (flags & O_APPEND)) {
    struct stat st;
    return fstat(fd, &st) == 0 ? st.st_size : -1;
  }
  return lseek(fd, 0, SEEK_CUR);
}

static void log_write(unsigned long long inode, off_t offset, ssize_t length) {
  if (length <= 0) return;
  pthread_mutex_lock(&lock);
  if (offset < 0) die("cannot determine write offset");
  append_log("W %llu %lld %lld\n", inode, (long long)offset, (long long)length);
  pthread_mutex_unlock(&lock);
}

/* ---- sync recording ------------------------------------------------------ */

static void copy_range(int from, off_t offset, size_t length, const char *to) {
  int out = (int)RAW_OPEN(to, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (out < 0) die("cannot create shadow copy");
  char buffer[1 << 16];
  while (length > 0) {
    size_t want = length < sizeof(buffer) ? length : sizeof(buffer);
    ssize_t n = RAW_PREAD(from, buffer, want, offset);
    if (n < 0) die("cannot read synced data");
    if (n == 0) break;
    if (RAW_WRITE(out, buffer, (size_t)n) != n) die("cannot write shadow");
    offset += n;
    length -= (size_t)n;
  }
  RAW_CLOSE(out);
}

static void list_directory(int directory, const char *to) {
  int handle = openat(directory, ".", O_RDONLY | O_DIRECTORY);
  DIR *dir = handle >= 0 ? fdopendir(handle) : NULL;
  int out = (int)RAW_OPEN(to, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (!dir || out < 0) die("cannot list synced directory");
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
    /* Entries without a verified descriptor are left out (not persisted). */
    unsigned long long inode = 0;
    int child = adopt(
        openat(dirfd(dir), entry->d_name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC),
        &inode);
    if (child < 0) continue;
    struct stat st;
    if (fstat(child, &st) != 0) continue;
    char line[PATH_MAX + 64];
    int n = snprintf(line, sizeof(line), "%c %llu %s\n",
                     S_ISDIR(st.st_mode) ? 'd' : 'f', inode, entry->d_name);
    if (n <= 0 || (size_t)n >= sizeof(line) ||
        RAW_WRITE(out, line, (size_t)n) != n)
      die("cannot write directory listing");
  }
  closedir(dir);
  RAW_CLOSE(out);
}

static void record_fd(int fd) {
  if (!shadow || !fd_under_root(fd)) return;
  pthread_mutex_lock(&lock);
  unsigned long long inode = 0;
  int reader = pin_fd(fd, &inode);
  struct stat st;
  if (reader >= 0 && fstat(reader, &st) == 0) {
    long seq = ++sequence;
    char copy[PATH_MAX];
    snprintf(copy, sizeof(copy), "%s/%ld", shadow, seq);
    if (S_ISDIR(st.st_mode)) {
      list_directory(reader, copy);
    } else {
      copy_range(reader, 0, (size_t)-1, copy);
    }
    append_log("%c %llu %ld\n", S_ISDIR(st.st_mode) ? 'D' : 'F', inode, seq);
  }
  pthread_mutex_unlock(&lock);
}

/* Shared file mappings, in whole pages: [begin, begin + length) maps the file
 * range starting at `offset` of the pinned `inode`. */
struct mapping {
  uintptr_t begin;
  uintptr_t length;
  off_t offset;
  unsigned long long inode;
};
#define MAX_MAPPINGS 4096
static struct mapping mappings[MAX_MAPPINGS];

static uintptr_t page_size(void) {
  return (uintptr_t)sysconf(_SC_PAGESIZE);
}
static uintptr_t page_down(uintptr_t value) {
  return value & ~(page_size() - 1);
}
static uintptr_t page_up(uintptr_t value) {
  return page_down(value + page_size() - 1);
}

static struct mapping *free_slot(void) {
  for (int i = 0; i < MAX_MAPPINGS; ++i)
    if (mappings[i].length == 0) return &mappings[i];
  die("mapping table full");
  return NULL;
}

/* Removes [begin, end) from every mapping, splitting partial overlaps.
 * Caller holds lock. */
static void forget_range(uintptr_t begin, uintptr_t end) {
  for (int i = 0; i < MAX_MAPPINGS; ++i) {
    struct mapping *m = &mappings[i];
    if (m->length == 0) continue;
    const uintptr_t m_end = m->begin + m->length;
    if (m_end <= begin || end <= m->begin) continue;
    const struct mapping original = *m;
    m->length = 0;
    if (original.begin < begin) {
      struct mapping *left = free_slot();
      *left = original;
      left->length = begin - original.begin;
    }
    if (end < m_end) {
      struct mapping *right = free_slot();
      *right = original;
      right->begin = end;
      right->length = m_end - end;
      right->offset = original.offset + (off_t)(end - original.begin);
    }
  }
}

static void remember_mapping(void *address, size_t length, int flags, int fd,
                             off_t offset) {
  if (!shadow || address == MAP_FAILED || length == 0) return;
  const uintptr_t begin = (uintptr_t)address;
  const uintptr_t end = page_up(begin + length);
  pthread_mutex_lock(&lock);
  /* A new mapping (MAP_FIXED included) replaces whatever was there. */
  forget_range(begin, end);
  unsigned long long inode = 0;
  if (fd >= 0 && (flags & MAP_SHARED) && fd_under_root(fd) &&
      pin_fd(fd, &inode) >= 0) {
    struct mapping *m = free_slot();
    m->begin = begin;
    m->length = end - begin;
    m->offset = offset;
    m->inode = inode;
  }
  pthread_mutex_unlock(&lock);
}

static void forget_mapping(void *address, size_t length) {
  if (!shadow) return;
  const uintptr_t begin = page_down((uintptr_t)address);
  pthread_mutex_lock(&lock);
  forget_range(begin, page_up((uintptr_t)address + length));
  pthread_mutex_unlock(&lock);
}

/* msync(MS_SYNC) writes back whole pages: record every page-rounded file
 * range it covered, read through the pinned descriptor. */
static void record_msync(void *address, size_t length) {
  if (!shadow) return;
  const uintptr_t begin = page_down((uintptr_t)address);
  const uintptr_t end = page_up((uintptr_t)address + length);
  pthread_mutex_lock(&lock);
  for (int i = 0; i < MAX_MAPPINGS; ++i) {
    const struct mapping *m = &mappings[i];
    if (m->length == 0) continue;
    const uintptr_t from = m->begin > begin ? m->begin : begin;
    const uintptr_t to =
        m->begin + m->length < end ? m->begin + m->length : end;
    if (from >= to) continue;
    const int reader = find_pinned(m->inode);
    if (reader < 0) die("mapping lost its pin");
    const off_t offset = m->offset + (off_t)(from - m->begin);
    long seq = ++sequence;
    char copy[PATH_MAX];
    snprintf(copy, sizeof(copy), "%s/%ld", shadow, seq);
    struct stat st;
    if (fstat(reader, &st) != 0) die("cannot stat synced mapping");
    copy_range(reader, offset, (size_t)(to - from), copy);
    append_log("R %llu %lld %lld %ld\n", m->inode, (long long)offset,
               (long long)st.st_size, seq);
  }
  pthread_mutex_unlock(&lock);
}

#if defined(__linux__)
/* mremap keeps the file offset of the mapping that starts at `old`. */
static void move_mapping(void *old, size_t old_length, void *moved,
                         size_t new_length) {
  if (!shadow || moved == MAP_FAILED) return;
  const uintptr_t old_begin = (uintptr_t)old;
  const uintptr_t new_begin = (uintptr_t)moved;
  pthread_mutex_lock(&lock);
  struct mapping found = {0};
  for (int i = 0; i < MAX_MAPPINGS; ++i) {
    if (mappings[i].length != 0 && mappings[i].begin == old_begin) {
      found = mappings[i];
      break;
    }
  }
  forget_range(old_begin, page_up(old_begin + old_length));
  forget_range(new_begin, page_up(new_begin + new_length));
  if (found.length != 0) {
    struct mapping *m = free_slot();
    *m = found;
    m->begin = new_begin;
    m->length = page_up(new_begin + new_length) - new_begin;
  }
  pthread_mutex_unlock(&lock);
}
#endif

/* ---- interposed functions ------------------------------------------------ */

/* Shared bodies: `CALL` performs the real operation. */
#define WRITE_BODY(OFFSET, CALL)                                 \
  if (inject(fd)) return -1;                                     \
  unsigned long long failed_ino = 0;                             \
  const int track = failed_inode(fd, &failed_ino);               \
  const off_t start = track ? (OFFSET) : 0;                      \
  ssize_t result = CALL;                                         \
  if (track && result > 0) log_write(failed_ino, start, result); \
  return result

#define SYNC_BODY(CALL)           \
  if (inject_sync(fd)) return -1; \
  int result = CALL;              \
  if (result == 0) record_fd(fd); \
  return result

#define TRUNCATE_BODY(CALL)                                     \
  unsigned long long failed_ino = 0;                            \
  const int track = failed_inode(fd, &failed_ino);              \
  int result = CALL;                                            \
  if (track && result == 0) {                                   \
    pthread_mutex_lock(&lock);                                  \
    append_log("T %llu %lld\n", failed_ino, (long long)length); \
    pthread_mutex_unlock(&lock);                                \
  }                                                             \
  return result

#if defined(__APPLE__)

extern ssize_t write_nocancel(int, const void *,
                              size_t) __asm("_write$NOCANCEL");
extern ssize_t pwrite_nocancel(int, const void *, size_t,
                               off_t) __asm("_pwrite$NOCANCEL");
extern ssize_t writev_nocancel(int, const struct iovec *,
                               int) __asm("_writev$NOCANCEL");
extern int fsync_nocancel(int) __asm("_fsync$NOCANCEL");
extern int fcntl_nocancel(int, int, ...) __asm("_fcntl$NOCANCEL");
extern int msync_nocancel(void *, size_t, int) __asm("_msync$NOCANCEL");

#define SHIM(name) shim_##name

static ssize_t SHIM(write)(int fd, const void *buf, size_t n) {
  WRITE_BODY(write_offset(fd), write(fd, buf, n));
}
static ssize_t SHIM(write_nocancel)(int fd, const void *buf, size_t n) {
  WRITE_BODY(write_offset(fd), write_nocancel(fd, buf, n));
}
static ssize_t SHIM(pwrite)(int fd, const void *buf, size_t n, off_t off) {
  WRITE_BODY(off, pwrite(fd, buf, n, off));
}
static ssize_t SHIM(pwrite_nocancel)(int fd, const void *buf, size_t n,
                                     off_t off) {
  WRITE_BODY(off, pwrite_nocancel(fd, buf, n, off));
}
static ssize_t SHIM(writev)(int fd, const struct iovec *v, int c) {
  WRITE_BODY(write_offset(fd), writev(fd, v, c));
}
static ssize_t SHIM(writev_nocancel)(int fd, const struct iovec *v, int c) {
  WRITE_BODY(write_offset(fd), writev_nocancel(fd, v, c));
}
static int SHIM(ftruncate)(int fd, off_t length) {
  TRUNCATE_BODY(ftruncate(fd, length));
}
static int SHIM(fsync)(int fd) {
  SYNC_BODY(fsync(fd));
}
static int SHIM(fsync_nocancel)(int fd) {
  SYNC_BODY(fsync_nocancel(fd));
}
static int SHIM(fdatasync)(int fd) {
  SYNC_BODY(fdatasync(fd));
}

/* fcntl's third argument depends on the command. Only the barriers are
 * handled here; every other command is forwarded with its exact argument
 * type, and an unknown command stops the test rather than guessing. */
enum fcntl_argument { kUnknown, kNoArgument, kIntArgument, kPointerArgument };

static enum fcntl_argument fcntl_argument_kind(int cmd) {
  switch (cmd) {
    case F_GETFD:
    case F_GETFL:
    case F_GETOWN:
    case F_FULLFSYNC:
    case F_BARRIERFSYNC:
    case F_FLUSH_DATA:
    case F_CHKCLEAN:
    case F_GETNOSIGPIPE:
    case F_GETPROTECTIONCLASS:
    case F_GETPROTECTIONLEVEL:
#ifdef F_GETLEASE
    case F_GETLEASE:
#endif
      return kNoArgument;
    case F_DUPFD:
    case F_DUPFD_CLOEXEC:
#ifdef F_DUPFD_CLOFORK
    case F_DUPFD_CLOFORK:
#endif
    case F_SETFD:
    case F_SETFL:
    case F_SETOWN:
    case F_RDAHEAD:
    case F_NOCACHE:
    case F_GLOBAL_NOCACHE:
    case F_NODIRECT:
    case F_SETNOSIGPIPE:
    case F_SETPROTECTIONCLASS:
    case F_SINGLE_WRITER:
#ifdef F_SETLEASE
    case F_SETLEASE:
#endif
      return kIntArgument;
    case F_GETLK:
    case F_SETLK:
    case F_SETLKW:
    case F_SETLKWTIMEOUT:
#ifdef F_OFD_SETLK
    case F_OFD_SETLK:
    case F_OFD_SETLKW:
    case F_OFD_GETLK:
    case F_OFD_SETLKWTIMEOUT:
#endif
    case F_PREALLOCATE:
    case F_PUNCHHOLE:
    case F_SETSIZE:
    case F_RDADVISE:
    case F_LOG2PHYS:
    case F_LOG2PHYS_EXT:
    case F_GETPATH:
#ifdef F_GETPATH_NOFIRMLINK
    case F_GETPATH_NOFIRMLINK:
#endif
    case F_GETLKPID:
      return kPointerArgument;
    default:
      return kUnknown;
  }
}

#define FCNTL_BODY(real)                                                   \
  const int barrier = cmd == F_FULLFSYNC || cmd == F_BARRIERFSYNC;         \
  if (barrier && inject_sync(fd)) return -1;                               \
  int result = -1;                                                         \
  va_list ap;                                                              \
  va_start(ap, cmd);                                                       \
  switch (fcntl_argument_kind(cmd)) {                                      \
    case kNoArgument:                                                      \
      result = real(fd, cmd);                                              \
      break;                                                               \
    case kIntArgument:                                                     \
      result = real(fd, cmd, va_arg(ap, int));                             \
      break;                                                               \
    case kPointerArgument:                                                 \
      result = real(fd, cmd, va_arg(ap, void *));                          \
      break;                                                               \
    case kUnknown: {                                                       \
      char message[64];                                                    \
      snprintf(message, sizeof(message), "unknown fcntl command %d", cmd); \
      die(message);                                                        \
    }                                                                      \
  }                                                                        \
  va_end(ap);                                                              \
  if (barrier && result != -1) record_fd(fd);                              \
  return result

static int SHIM(fcntl)(int fd, int cmd, ...) {
  FCNTL_BODY(fcntl);
}
static int SHIM(fcntl_nocancel)(int fd, int cmd, ...) {
  FCNTL_BODY(fcntl_nocancel);
}
static int SHIM(msync)(void *address, size_t length, int flags) {
  int result = msync(address, length, flags);
  if (result == 0 && (flags & MS_SYNC)) record_msync(address, length);
  return result;
}
static int SHIM(msync_nocancel)(void *address, size_t length, int flags) {
  int result = msync_nocancel(address, length, flags);
  if (result == 0 && (flags & MS_SYNC)) record_msync(address, length);
  return result;
}
static void *SHIM(mmap)(void *address, size_t length, int prot, int flags,
                        int fd, off_t offset) {
  void *result = mmap(address, length, prot, flags, fd, offset);
  remember_mapping(result, length, flags, fd, offset);
  return result;
}
static int SHIM(munmap)(void *address, size_t length) {
  int result = munmap(address, length);
  if (result == 0) forget_mapping(address, length);
  return result;
}

#define INTERPOSE(replacement, original) \
  {(const void *)(replacement), (const void *)(original)}
__attribute__((used)) static const struct {
  const void *replacement;
  const void *original;
} interposers[] __attribute__((section("__DATA,__interpose"))) = {
    INTERPOSE(SHIM(write), write),
    INTERPOSE(SHIM(write_nocancel), write_nocancel),
    INTERPOSE(SHIM(pwrite), pwrite),
    INTERPOSE(SHIM(pwrite_nocancel), pwrite_nocancel),
    INTERPOSE(SHIM(writev), writev),
    INTERPOSE(SHIM(writev_nocancel), writev_nocancel),
    INTERPOSE(SHIM(ftruncate), ftruncate),
    INTERPOSE(SHIM(fsync), fsync),
    INTERPOSE(SHIM(fsync_nocancel), fsync_nocancel),
    INTERPOSE(SHIM(fdatasync), fdatasync),
    INTERPOSE(SHIM(fcntl), fcntl),
    INTERPOSE(SHIM(fcntl_nocancel), fcntl_nocancel),
    INTERPOSE(SHIM(msync), msync),
    INTERPOSE(SHIM(msync_nocancel), msync_nocancel),
    INTERPOSE(SHIM(mmap), mmap),
    INTERPOSE(SHIM(munmap), munmap),
};

#else /* Linux: symbol preemption through LD_PRELOAD */

ssize_t write(int fd, const void *buf, size_t n) {
  RESOLVE();
  WRITE_BODY(write_offset(fd), real_write(fd, buf, n));
}
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
  RESOLVE();
  WRITE_BODY(off, real_pwrite(fd, buf, n, off));
}
ssize_t pwrite64(int fd, const void *buf, size_t n, off64_t off) {
  RESOLVE();
  WRITE_BODY((off_t)off, real_pwrite64(fd, buf, n, off));
}
ssize_t writev(int fd, const struct iovec *v, int c) {
  RESOLVE();
  WRITE_BODY(write_offset(fd), real_writev(fd, v, c));
}
int ftruncate(int fd, off_t length) {
  RESOLVE();
  TRUNCATE_BODY(real_ftruncate(fd, length));
}
int ftruncate64(int fd, off64_t length) {
  RESOLVE();
  TRUNCATE_BODY(real_ftruncate64(fd, length));
}
int fsync(int fd) {
  RESOLVE();
  SYNC_BODY(real_fsync(fd));
}
int fdatasync(int fd) {
  RESOLVE();
  SYNC_BODY(real_fdatasync(fd));
}
int msync(void *address, size_t length, int flags) {
  RESOLVE();
  int result = real_msync(address, length, flags);
  if (result == 0 && (flags & MS_SYNC)) record_msync(address, length);
  return result;
}
void *mmap(void *address, size_t length, int prot, int flags, int fd,
           off_t offset) {
  RESOLVE();
  void *result = real_mmap(address, length, prot, flags, fd, offset);
  remember_mapping(result, length, flags, fd, offset);
  return result;
}
void *mmap64(void *address, size_t length, int prot, int flags, int fd,
             off64_t offset) {
  RESOLVE();
  void *result = real_mmap64(address, length, prot, flags, fd, offset);
  remember_mapping(result, length, flags, fd, (off_t)offset);
  return result;
}
void *mremap(void *old, size_t old_length, size_t new_length, int flags, ...) {
  RESOLVE();
  void *result;
  if (flags & MREMAP_FIXED) {
    va_list ap;
    va_start(ap, flags);
    void *target = va_arg(ap, void *);
    va_end(ap);
    result = real_mremap(old, old_length, new_length, flags, target);
  } else {
    result = real_mremap(old, old_length, new_length, flags);
  }
  move_mapping(old, old_length, result, new_length);
  return result;
}
int munmap(void *address, size_t length) {
  RESOLVE();
  int result = real_munmap(address, length);
  if (result == 0) forget_mapping(address, length);
  return result;
}

#endif
