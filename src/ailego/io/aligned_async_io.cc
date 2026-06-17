#include "aligned_async_io.h"

#include <cerrno>
#include <zvec/core/framework/index_logger.h>

#if defined(__linux__) && defined(ZVEC_HAS_LIBAIO)
#include <unistd.h>

namespace zvec {
namespace ailego {

static int execute_io_pread(int fd, std::vector<AlignedRead> &reads) {
  for (auto &r : reads) {
    ssize_t got = ::pread(fd, r.buf, r.len, static_cast<off_t>(r.offset));
    if (got != static_cast<ssize_t>(r.len)) {
      LOG_ERROR(
          "pread fallback failed: fd=%d, offset=%llu, len=%llu, got=%zd, "
          "errno=%d",
          fd, (unsigned long long)r.offset, (unsigned long long)r.len, got,
          errno);
      return -1;
    }
  }
  return 0;
}

static constexpr int kMaxEvents = 128;

ScopedIOContext::ScopedIOContext() {
  ctx_ = 0;
  int ret = io_setup(kMaxEvents, &ctx_);
  if (ret == 0) {
    valid_ = true;
  } else {
    LOG_WARN("io_setup failed (ret=%d, errno=%d); prefetch disabled on thread",
             ret, errno);
    valid_ = false;
  }
}

ScopedIOContext::~ScopedIOContext() {
  if (valid_) {
    io_destroy(ctx_);
    valid_ = false;
  }
}

int execute_aligned_io(IOContext ctx, int fd, std::vector<AlignedRead> &reads,
                       uint64_t n_retries) {
  if (reads.empty()) return 0;

  std::vector<iocb> cbs(reads.size());
  std::vector<iocb *> cb_ptrs(reads.size());
  std::vector<io_event> events(reads.size());

  for (size_t i = 0; i < reads.size(); ++i) {
    io_prep_pread(&cbs[i], fd, reads[i].buf, reads[i].len,
                  static_cast<long long>(reads[i].offset));
    cb_ptrs[i] = &cbs[i];
  }

  int n_submitted = 0;
  int total = static_cast<int>(reads.size());

  for (uint64_t attempt = 0; attempt <= n_retries; ++attempt) {
    while (n_submitted < total) {
      int batch = std::min(total - n_submitted, kMaxEvents);
      int ret = io_submit(ctx, batch, cb_ptrs.data() + n_submitted);
      if (ret >= 0) {
        n_submitted += ret;
      } else if (ret == -EINTR) {
        continue;
      } else if (ret == -EAGAIN) {
        continue;
      } else {
        LOG_WARN("io_submit failed: ret=%d, errno=%d; fallback to pread", ret,
                 errno);
        return execute_io_pread(fd, reads);
      }
    }
    if (n_submitted == total) break;
  }

  if (n_submitted != total) {
    LOG_WARN("io_submit incomplete after retries; fallback to pread");
    return execute_io_pread(fd, reads);
  }

  int n_collected = 0;
  while (n_collected < total) {
    int ret =
        io_getevents(ctx, total - n_collected, total - n_collected, events.data() + n_collected, nullptr);
    if (ret > 0) {
      n_collected += ret;
    } else if (ret == -EINTR) {
      continue;
    } else {
      LOG_WARN("io_getevents failed: ret=%d; fallback to pread", ret);
      return execute_io_pread(fd, reads);
    }
  }

  for (int i = 0; i < total; ++i) {
    if (static_cast<uint64_t>(events[i].res) != reads[i].len) {
      LOG_WARN(
          "aio short read: expected=%llu, got=%lld; fallback to pread",
          (unsigned long long)reads[i].len, (long long)events[i].res);
      return execute_io_pread(fd, reads);
    }
  }

  return 0;
}

}  // namespace ailego
}  // namespace zvec

#else

namespace zvec {
namespace ailego {

ScopedIOContext::ScopedIOContext() { valid_ = false; }
ScopedIOContext::~ScopedIOContext() {}

int execute_aligned_io(IOContext, int fd, std::vector<AlignedRead> &reads,
                       uint64_t) {
  for (auto &r : reads) {
    ssize_t got = ::pread(fd, r.buf, r.len, static_cast<off_t>(r.offset));
    if (got != static_cast<ssize_t>(r.len)) {
      return -1;
    }
  }
  return 0;
}

}  // namespace ailego
}  // namespace zvec

#endif
