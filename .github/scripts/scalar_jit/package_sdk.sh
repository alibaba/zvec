#!/usr/bin/env bash
# Run in Dockerfile.scalar_jit_manylinux_2_28 with the repository at /work.
set -euo pipefail
cd /work
ZVEC_JIT_JOBS=${ZVEC_JIT_JOBS:-$(nproc)}
export CMAKE_GENERATOR=Ninja

cmake -S . -B build-jit-sdk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/work/jit-sdk \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_TOOLS=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=ON \
  -DBUILD_ZVEC_SHARED=ON -DBUILD_ZVEC_CORE_SHARED=ON -DBUILD_ZVEC_AILEGO_SHARED=ON \
  -DZVEC_ENABLE_SCALAR_JIT=ON -DZVEC_SCALAR_JIT_LLVM_SHARED=OFF \
  -DLLVM_ROOT=/opt/llvm-20 -DOPENSSL_ROOT_DIR=/opt/zvec-jit-openssl \
  -DOPENSSL_USE_STATIC_LIBS=ON -DENABLE_NATIVE=OFF
cmake --build build-jit-sdk --parallel "$ZVEC_JIT_JOBS" --target \
  zvec_shared zvec_core_shared zvec_ailego_shared zvec_c_api \
  scalar_doc_filter_test scalar_predicate_expr_test scalar_jit_runtime_test
ctest --test-dir build-jit-sdk --output-on-failure --timeout 120 \
  --output-junit scalar-tests.xml \
  -R '^(scalar_doc_filter_test|scalar_predicate_expr_test|scalar_jit_runtime_test)$'
cmake --install build-jit-sdk --strip
# Match _build_prebuilt.yml: the static archives are not self-contained SDKs.
rm -f jit-sdk/lib/*.a
python .github/scripts/scalar_jit/check_artifact.py jit-sdk \
  --build-directory build-jit-sdk --report sdk-build-audit.json \
  --llvm-dis /opt/llvm-20/bin/llvm-dis
cmake -S .github/cmake/sdk-smoke -B build-jit-sdk-smoke -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DZVEC_SDK_DIR=/work/jit-sdk \
  -DZVEC_SDK_SCALAR_JIT_SMOKE=ON
cmake --build build-jit-sdk-smoke --parallel 2
ctest --test-dir build-jit-sdk-smoke --output-on-failure
python .github/scripts/scalar_jit/sdk_smoke.py \
  --binary build-jit-sdk-smoke/scalar_jit_smoke \
  --collection build-jit-sdk-smoke/jit-collection \
  --library-directory jit-sdk/lib --output-directory sdk-jit-build-logs
tar czf zvec-sdk-linux-amd64-scalar-jit.tar.gz -C jit-sdk .
