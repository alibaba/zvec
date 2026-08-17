#include <stdio.h>
#include <zvec/c_api.h>

int main(void) {
  const char *version = zvec_get_version();
  if (!version) {
    fprintf(stderr, "c_smoke: zvec_get_version() returned NULL\n");
    return 1;
  }
  printf("c_smoke: zvec version: %s\n", version);

  int major = zvec_get_version_major();
  int minor = zvec_get_version_minor();
  int patch = zvec_get_version_patch();
  if (!zvec_check_version(major, minor, patch)) {
    fprintf(stderr, "c_smoke: zvec_check_version(%d.%d.%d) failed\n",
            major, minor, patch);
    return 1;
  }
  printf("c_smoke: OK\n");
  return 0;
}
