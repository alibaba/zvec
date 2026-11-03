// iOS test sandbox helper
// Automatically changes the working directory to a writable location
// before any test code runs. On iOS, the app bundle is read-only,
// so tests that write files need a writable CWD.

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IOS || TARGET_OS_SIMULATOR

#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <sys/resource.h>

__attribute__((constructor))
static void ios_test_sandbox_setup() {
    // 1. Raise file descriptor limit (iOS default is very low)
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rlim_t target = 65536;
        if (rl.rlim_cur < target) {
            rl.rlim_cur = (rl.rlim_max >= target) ? target : rl.rlim_max;
            if (setrlimit(RLIMIT_NOFILE, &rl) == 0) {
                fprintf(stderr, "[iOS] File descriptor limit raised to: %llu\n",
                        (unsigned long long)rl.rlim_cur);
            }
        }
    }

    // 2. Change CWD to writable sandbox directory
    // TMPDIR is set by iOS to the app's writable sandbox tmp directory
    const char* tmpdir = getenv("TMPDIR");
    if (tmpdir && chdir(tmpdir) == 0) {
        fprintf(stderr, "[iOS] Working directory set to: %s\n", tmpdir);
    } else {
        // Fallback: try HOME directory
        const char* home = getenv("HOME");
        if (home && chdir(home) == 0) {
            fprintf(stderr, "[iOS] Working directory set to: %s\n", home);
        }
    }
}

#endif // TARGET_OS_IOS || TARGET_OS_SIMULATOR
#endif // __APPLE__
