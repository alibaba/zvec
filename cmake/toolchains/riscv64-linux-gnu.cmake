## Cross-compilation toolchain for RISC-V 64-bit (lp64d ABI) on Ubuntu/Debian.
##
## Ported from faiss (cmake/toolchains/riscv64-linux-gnu.cmake) for the
## riscv64 RVV cross-compile CI job (.github/workflows/08-linux-riscv64-rvv-cross.yml).
##
## Required host packages (Ubuntu 24.04), same as faiss's CI:
##   gcc-14-riscv64-linux-gnu  g++-14-riscv64-linux-gnu
##   libc6:riscv64  libc6-dev:riscv64  libstdc++6:riscv64  libgcc-s1:riscv64
##
## Target libraries are installed via apt multiarch to
## /usr/lib/riscv64-linux-gnu/. CMake's ONLY find-root mode searches
## ${CMAKE_FIND_ROOT_PATH}/usr/lib/riscv64-linux-gnu/ (via the compiler's
## multiarch tuple), so the CI job creates this symlink before configuring:
##   /usr/riscv64-linux-gnu/usr/lib/riscv64-linux-gnu
##     -> /usr/lib/riscv64-linux-gnu
##
## Pin a specific GCC version from the command line if needed:
##   cmake -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc-14 \
##         -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++-14 ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
endif()

# Cross-compiler sysroot provided by the gcc-riscv64-linux-gnu packages.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)

# Never look for host-side tools (cmake, python, ...) inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Look for target libraries/headers/packages only inside the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
