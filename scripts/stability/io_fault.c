/* Test-process-only Linux syscall interposition; never linked into zvec. */
#define _GNU_SOURCE
#include <sys/syscall.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *root;
static const char *arm;
static const char *audit;
static const char *operation;
static long after = 1;
static int once;
static long calls;

__attribute__((constructor)) static void configure(void) {
  root = getenv("ZVEC_FAULT_ROOT");
  arm = getenv("ZVEC_FAULT_ARM");
  audit = getenv("ZVEC_FAULT_AUDIT");
  operation = getenv("ZVEC_FAULT_OPERATION");
  const char *value = getenv("ZVEC_FAULT_AFTER");
  if (value) after = strtol(value, NULL, 10);
  once = getenv("ZVEC_FAULT_ONCE") != NULL;
}

static int matches(const char *path) {
  return root && path && strncmp(path, root, strlen(root)) == 0 &&
         path[strlen(root)] == '/';
}

static void fd_path(int fd, char *path) {
  char link[64];
  snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
  ssize_t size = readlink(link, path, PATH_MAX - 1);
  path[size < 0 ? 0 : size] = '\0';
}

static void record(const char *kind, const char *name, const char *path,
                   int error) {
  if (!audit || !matches(path)) return;
  int saved = errno;
  char line[PATH_MAX + 128];
  int count = snprintf(line, sizeof(line), "%s %s errno=%d %s\n", kind, name,
                       error, path);
  int fd =
      syscall(SYS_openat, AT_FDCWD, audit, O_WRONLY | O_APPEND | O_CREAT, 0600);
  if (fd < 0 || count < 0 || (size_t)count >= sizeof(line) ||
      syscall(SYS_write, fd, line, count) != count) {
    /* An unrecorded injection must fail the run, not look like no fault hit. */
    _exit(125);
  }
  syscall(SYS_close, fd);
  errno = saved;
}

static int inject(const char *name, const char *path) {
  if (!matches(path) || !operation || strcmp(operation, name) != 0 || !arm ||
      access(arm, F_OK) != 0)
    return 0;
  long call = __atomic_add_fetch(&calls, 1, __ATOMIC_SEQ_CST);
  if (call < after || (once && call != after)) return 0;
  record("INJECT", name, path, EIO);
  errno = EIO;
  return 1;
}

#define WRITE_WRAPPER(name, signature, arguments)          \
  ssize_t name signature {                                 \
    ssize_t(*real_fn) signature = dlsym(RTLD_NEXT, #name); \
    char path[PATH_MAX];                                   \
    fd_path(fd, path);                                     \
    if (inject("write", path)) return -1;                  \
    ssize_t result = real_fn arguments;                    \
    if (result < 0) record("ERROR", #name, path, errno);   \
    return result;                                         \
  }

WRITE_WRAPPER(write, (int fd, const void *buf, size_t count), (fd, buf, count))
WRITE_WRAPPER(pwrite, (int fd, const void *buf, size_t count, off_t offset),
              (fd, buf, count, offset))
WRITE_WRAPPER(pwrite64, (int fd, const void *buf, size_t count, off64_t offset),
              (fd, buf, count, offset))

#define SYNC_WRAPPER(name)                               \
  int name(int fd) {                                     \
    int (*real_fn)(int) = dlsym(RTLD_NEXT, #name);       \
    char path[PATH_MAX];                                 \
    fd_path(fd, path);                                   \
    if (inject("sync", path)) return -1;                 \
    int result = real_fn(fd);                            \
    if (result < 0) record("ERROR", #name, path, errno); \
    return result;                                       \
  }
SYNC_WRAPPER(fsync)
SYNC_WRAPPER(fdatasync)

#define OPEN_WRAPPER(name)                                            \
  int name(const char *path, int flags, ...) {                        \
    int (*real_fn)(const char *, int, ...) = dlsym(RTLD_NEXT, #name); \
    mode_t mode = 0;                                                  \
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {      \
      va_list ap;                                                     \
      va_start(ap, flags);                                            \
      mode = va_arg(ap, int);                                         \
      va_end(ap);                                                     \
    }                                                                 \
    int result = real_fn(path, flags, mode);                          \
    if (result < 0) record("ERROR", #name, path, errno);              \
    return result;                                                    \
  }
OPEN_WRAPPER(open)
OPEN_WRAPPER(open64)
