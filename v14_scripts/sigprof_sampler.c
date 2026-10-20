// LD_PRELOAD leaf-PC sampling profiler (v17-E5).
//
// perf_event_paranoid=4 blocks perf and ptrace_scope=1 blocks gdb attach on
// this server, so hotspot sampling uses an ITIMER_PROF signal handler in the
// target process itself. Each tick records (monotonic ns, RIP) into a
// preallocated buffer; a /proc/self/maps snapshot is taken at the first tick
// so offline symbolization can undo ASLR. The handler only uses
// async-signal-safe calls. bench.cc installs no competing timer.
//
// Env:
//   SIGPROF_OUT     output path for the raw samples (default ./sigprof.bin)
//   SIGPROF_HZ      sampling rate, default 1000
//   SIGPROF_SECONDS stop sampling after this many seconds, 0 = no limit
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

struct Sample {
  uint64_t ns;
  uint64_t rip;
};

static struct Sample *g_samples;
static volatile size_t g_n;
static size_t g_cap;
static int g_maps_dumped;
static uint64_t g_t0;
static uint64_t g_limit_ns;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void dump_maps(void) {
  const char *out = getenv("SIGPROF_OUT");
  if (!out) out = "sigprof.bin";
  size_t len = strlen(out);
  char path[512];
  if (len + 6 >= sizeof(path)) return;
  memcpy(path, out, len);
  memcpy(path + len, ".maps", 6);
  int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) return;
  int mfd = open("/proc/self/maps", O_RDONLY);
  if (mfd >= 0) {
    char buf[4096];
    ssize_t got;
    while ((got = read(mfd, buf, sizeof(buf))) > 0) {
      ssize_t off = 0;
      while (off < got) {
        ssize_t w = write(fd, buf + off, got - off);
        if (w <= 0) break;
        off += w;
      }
    }
    close(mfd);
  }
  close(fd);
}

static void handler(int sig, siginfo_t *si, void *uc) {
  (void)sig;
  (void)si;
  ucontext_t *u = (ucontext_t *)uc;
  const uint64_t rip = (uint64_t)u->uc_mcontext.gregs[REG_RIP];
  const uint64_t ns = now_ns();
  if (!g_maps_dumped) {
    dump_maps();
    g_maps_dumped = 1;
  }
  if (g_limit_ns && ns - g_t0 > g_limit_ns) {
    struct itimerval off = {{0, 0}, {0, 0}};
    setitimer(ITIMER_PROF, &off, NULL);
    return;
  }
  if (g_n < g_cap) {
    g_samples[g_n].ns = ns;
    g_samples[g_n].rip = rip;
    ++g_n;
  }
}

__attribute__((constructor)) static void sampler_init(void) {
  g_cap = 1u << 20;
  g_n = 0;
  g_maps_dumped = 0;
  g_samples = malloc(g_cap * sizeof(struct Sample));
  if (!g_samples) return;
  g_t0 = now_ns();
  long hz = 1000;
  const char *hz_env = getenv("SIGPROF_HZ");
  if (hz_env && *hz_env) {
    char *end = NULL;
    long v = strtol(hz_env, &end, 10);
    if (end && *end == '\0' && v > 0 && v <= 100000) hz = v;
  }
  long secs = 0;
  const char *sec_env = getenv("SIGPROF_SECONDS");
  if (sec_env && *sec_env) {
    char *end = NULL;
    long v = strtol(sec_env, &end, 10);
    if (end && *end == '\0' && v >= 0) secs = v;
  }
  g_limit_ns = (uint64_t)secs * 1000000000ull;
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  sigaction(SIGPROF, &sa, NULL);
  const long us = 1000000L / hz;
  struct itimerval it;
  it.it_interval.tv_sec = us / 1000000L;
  it.it_interval.tv_usec = us % 1000000L;
  it.it_value = it.it_interval;
  setitimer(ITIMER_PROF, &it, NULL);
}

__attribute__((destructor)) static void sampler_fini(void) {
  struct itimerval stop_it = {{0, 0}, {0, 0}};
  setitimer(ITIMER_PROF, &stop_it, NULL);
  if (!g_samples) return;
  const char *out = getenv("SIGPROF_OUT");
  if (!out) out = "sigprof.bin";
  int fd = open(out, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) return;
  const size_t bytes = g_n * sizeof(struct Sample);
  size_t written = 0;
  while (written < bytes) {
    ssize_t w = write(fd, (const char *)g_samples + written, bytes - written);
    if (w <= 0) break;
    written += (size_t)w;
  }
  close(fd);
}
