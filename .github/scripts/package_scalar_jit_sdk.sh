#!/usr/bin/env bash
# Run in Dockerfile.scalar_jit_manylinux_2_28 with the repository at /work.
set -euo pipefail
cd /work
ZVEC_JIT_JOBS=${ZVEC_JIT_JOBS:-$(nproc)}

cmake -S . -B build-jit-sdk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/work/jit-sdk \
  -DBUILD_TOOLS=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=ON \
  -DBUILD_ZVEC_SHARED=ON -DBUILD_ZVEC_CORE_SHARED=ON -DBUILD_ZVEC_AILEGO_SHARED=ON \
  -DZVEC_ENABLE_SCALAR_JIT=ON -DZVEC_SCALAR_JIT_LLVM_SHARED=OFF \
  -DLLVM_ROOT=/opt/llvm-20 -DOPENSSL_ROOT_DIR=/opt/zvec-jit-openssl \
  -DOPENSSL_USE_STATIC_LIBS=ON -DENABLE_NATIVE=OFF
cmake --build build-jit-sdk --parallel "$ZVEC_JIT_JOBS" --target \
  zvec_shared zvec_core_shared zvec_ailego_shared zvec_c_api \
  scalar_doc_filter_test scalar_predicate_expr_test
ctest --test-dir build-jit-sdk --output-on-failure \
  -R '^(scalar_doc_filter_test|scalar_predicate_expr_test)$'
cmake --install build-jit-sdk --strip
# Match _build_prebuilt.yml: the static archives are not self-contained SDKs.
rm -f jit-sdk/lib/*.a
python .github/scripts/check_scalar_jit_artifact.py jit-sdk \
  --bitcode build-jit-sdk/thirdparty/arrow/arrow/src/ARROW.BUILD-build/src/gandiva/irhelpers.bc \
  --llvm-dis /opt/llvm-20/bin/llvm-dis
cmake -S .github/cmake/sdk-smoke -B build-jit-sdk-smoke -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DZVEC_SDK_DIR=/work/jit-sdk
cmake --build build-jit-sdk-smoke --parallel 2
ctest --test-dir build-jit-sdk-smoke --output-on-failure
# Compile against the installed SDK, so singleton state and JIT linkage are
# tested across the real public shared-library boundary.
g++ -std=c++17 -O2 -march=x86-64 -pthread -static-libstdc++ -static-libgcc \
  -Ijit-sdk/include benchmarks/pr889_linux/filter_bench.cc \
  -Ljit-sdk/lib -Wl,-rpath,/work/jit-sdk/lib -lzvec \
  -o build-jit-sdk-smoke/scalar_jit_bench
python .github/scripts/scalar_jit_sdk_smoke.py \
  --binary build-jit-sdk-smoke/scalar_jit_bench \
  --collection build-jit-sdk-smoke/jit-collection \
  --library-directory jit-sdk/lib --output-directory sdk-jit-build-logs
tar czf zvec-sdk-linux-amd64-scalar-jit.tar.gz -C jit-sdk .
