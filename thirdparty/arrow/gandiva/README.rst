Gandiva scalar execution extension
==================================

This directory contains version-specific modifications to Apache Arrow 21.0.0
Gandiva and their build configuration. ``prepare.cmake`` applies the source
overrides to zvec's existing Arrow build. ``PROVENANCE.json`` records the
upstream version and override hashes; ``LICENSE`` and ``NOTICE`` retain its
licensing information.

Dependency boundary
-------------------

The Gandiva extension generates a single-row predicate and binds raw Arrow
buffers. Its ``scalar_predicate_abi.h`` defines only the evaluation function
signature and fixed-width parameter slot. It has no dependency on zvec headers.

The owning ScalarPredicate wrapper, parameter extraction and 128-entry LRU
cache are internal zvec code in ``src/db/sqlengine/scalar_jit/``. The filter
adapter binds immutable snapshots and falls back to Arrow for unsupported
expressions, storage layouts or compilation/binding failures. These internal
runtime headers are not installed as Gandiva APIs.

Build
-----

The backend defaults to OFF. To enable it::

  cmake -S . -B build-scalar-jit -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DZVEC_ENABLE_SCALAR_JIT=ON \
    -DLLVM_ROOT=/path/to/llvm-20
  cmake --build build-scalar-jit --parallel 8

ON requires native 64-bit Linux/macOS, C++17, LLVM 20 with its shared library
and matching ``clang``/``llvm-link``, and OpenSSL development files. LLVM and
OpenSSL remain runtime dependencies. Cross, mobile, Windows and universal
macOS builds are unsupported when ON. OFF adds no LLVM/OpenSSL dependency.
Set ``ZVEC_SCALAR_JIT=0`` before the first query to disable the fast path.

Update the overrides and provenance hashes together. The extension changes
Gandiva's internal ABI and requires rebuilding Gandiva and its consumers.
