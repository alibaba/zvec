#!/usr/bin/env bash
# Build release dependencies inside the same manylinux image as the wheel/SDK.
set -euo pipefail

if [[ $(uname -m) != x86_64 ]]; then
  echo 'This experimental release toolchain currently supports Linux x86_64.' >&2
  exit 1
fi

ZVEC_LLVM_PREFIX=${ZVEC_LLVM_PREFIX:-/opt/llvm-20}
ZVEC_OPENSSL_PREFIX=${ZVEC_OPENSSL_PREFIX:-/opt/zvec-jit-openssl}
ZVEC_TOOLCHAIN_JOBS=${ZVEC_TOOLCHAIN_JOBS:-$(nproc)}
ZVEC_TOOLCHAIN_WORK=${ZVEC_TOOLCHAIN_WORK:-$(mktemp -d)}
mkdir -p "$ZVEC_TOOLCHAIN_WORK"
cd "$ZVEC_TOOLCHAIN_WORK"

download() {
  local url=$1 name=$2 sha=$3
  curl --fail --location --retry 5 --output "$name" "$url"
  printf '%s  %s\n' "$sha" "$name" | sha256sum --check -
}

download \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/llvm-project-20.1.8.src.tar.xz \
  llvm-project-20.1.8.src.tar.xz \
  6898f963c8e938981e6c4a302e83ec5beb4630147c7311183cf61069af16333d
tar -xf llvm-project-20.1.8.src.tar.xz
cmake -S llvm-project-20.1.8.src/llvm -B llvm-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ZVEC_LLVM_PREFIX" \
  -DCMAKE_C_FLAGS='-march=x86-64 -mtune=generic -ffunction-sections -fdata-sections' \
  -DCMAKE_CXX_FLAGS='-march=x86-64 -mtune=generic -ffunction-sections -fdata-sections' \
  -DCMAKE_EXE_LINKER_FLAGS='-Wl,--gc-sections' \
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_ENABLE_PIC=ON -DLLVM_ENABLE_RTTI=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF -DCLANG_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_CURL=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_FFI=OFF \
  -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_DISTRIBUTION_COMPONENTS='clang;clang-resource-headers;llvm-link;llvm-dis;llvm-config;llvm-headers;llvm-libraries;cmake-exports'
cmake --build llvm-build --target install-distribution --parallel "$ZVEC_TOOLCHAIN_JOBS"

# Gandiva invokes clang separately for its helper bitcode. Select the pinned
# image's GCC C++ headers; LLVM 20 no longer accepts GCC_INSTALL_PREFIX.
ZVEC_GCC_INSTALL_DIR=$(dirname "$(gcc -print-libgcc-file-name)")
printf '%s\n' "--gcc-install-dir=$ZVEC_GCC_INSTALL_DIR" > "$ZVEC_LLVM_PREFIX/bin/clang.cfg"
cp "$ZVEC_LLVM_PREFIX/bin/clang.cfg" "$ZVEC_LLVM_PREFIX/bin/clang++.cfg"
printf '#include <string>\nstd::string toolchain_check() { return "LLVM 20"; }\n' | \
  "$ZVEC_LLVM_PREFIX/bin/clang-20" -std=c++17 -x c++ -emit-llvm -c - -o /tmp/zvec-jit-toolchain-check.bc

download https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz \
  openssl-3.5.9.tar.gz 603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a
tar -xf openssl-3.5.9.tar.gz
pushd openssl-3.5.9
./Configure linux-x86_64 no-shared no-tests no-module no-dso \
  --prefix="$ZVEC_OPENSSL_PREFIX" --libdir=lib \
  -O2 -fPIC -march=x86-64 -mtune=generic -ffunction-sections -fdata-sections
make -j "$ZVEC_TOOLCHAIN_JOBS"
make install_sw
popd

mkdir -p "$ZVEC_LLVM_PREFIX/share/zvec-jit-licenses"
cp llvm-project-20.1.8.src/llvm/LICENSE.TXT \
  "$ZVEC_LLVM_PREFIX/share/zvec-jit-licenses/LLVM-LICENSE.txt"
cp openssl-3.5.9/LICENSE.txt \
  "$ZVEC_LLVM_PREFIX/share/zvec-jit-licenses/OpenSSL-LICENSE.txt"
"$ZVEC_LLVM_PREFIX/bin/llvm-config" --version --targets-built --shared-mode --system-libs
