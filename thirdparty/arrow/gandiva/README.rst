Gandiva scalar execution extension
==================================

This directory extends Apache Arrow 21.0.0 Gandiva with single-row execution.
``prepare.cmake`` applies ``scalar.patch`` to zvec's existing Arrow build and
copies only new extension files. ``PROVENANCE.json`` records the upstream
version and extension hashes; ``LICENSE`` and ``NOTICE`` retain its licensing
information.

Dependency boundary
-------------------

The patch adds read-only binding metadata, a scalar context for the existing
expression visitor, and build entries for the extension. ``ScalarGenerator``
owns single-row compilation, parameter metadata and buffer binding separately
from ``LLVMGenerator``. The visitor's default null context preserves batch
execution. ``scalar_predicate_abi.h`` defines only the evaluation signature and
parameter slot. Neither the patch nor the extension depends on zvec headers.

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

Update the patch, extension and provenance hashes together. Apply revised
patches to a clean Arrow source tree; conflicts must be reviewed rather than
overwriting upstream files. The extension still uses Gandiva internals:
upgrading Arrow requires rebuilding and validating scalar and batch semantics
before updating the version guard. Clean patch application alone does not
establish compatibility.
