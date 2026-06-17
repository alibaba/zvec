#pragma once

#include <cstdint>
#include <vector>

#ifdef ZVEC_HAS_LIBAIO
#include <libaio.h>
#endif

namespace zvec {
namespace ailego {

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;
};

#ifdef ZVEC_HAS_LIBAIO
using IOContext = io_context_t;
#else
using IOContext = uint32_t;
#endif

class ScopedIOContext {
 public:
  ScopedIOContext();
  ~ScopedIOContext();
  ScopedIOContext(const ScopedIOContext &) = delete;
  ScopedIOContext &operator=(const ScopedIOContext &) = delete;

  bool valid() const { return valid_; }
  IOContext &get() { return ctx_; }

 private:
  IOContext ctx_{};
  bool valid_{false};
};

int execute_aligned_io(IOContext ctx, int fd, std::vector<AlignedRead> &reads,
                       uint64_t n_retries = 3);

}  // namespace ailego
}  // namespace zvec
