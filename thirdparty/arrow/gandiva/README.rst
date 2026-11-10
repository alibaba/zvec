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

The adapter covers numeric and string comparisons, boolean equality, nullable
boolean logic and scalar null tests, LIKE, and numeric/string IN and NOT IN.
LIKE matchers and IN sets are owned per query and passed as parameters, so
changing a pattern or set reuses the compiled expression. Native helpers use
Arrow's LIKE and set-lookup semantics; they are registered through Gandiva's
external C function API. List operations (CONTAIN and array_length) and other
unsupported expressions continue through Arrow.

Build
-----

The backend defaults to OFF. To enable it::

  cmake -S . -B build-scalar-jit -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DZVEC_ENABLE_SCALAR_JIT=ON \
    -DLLVM_ROOT=/path/to/llvm-20
  cmake --build build-scalar-jit --parallel 8

ON requires native 64-bit Linux/macOS, C++17, LLVM 20 static libraries and
matching ``clang``/``llvm-link``, and OpenSSL development files. The default
link uses Arrow's required LLVM components and the native architecture backend,
with dead-code elimination at the final link. It retains the optimizer and
runtime CPU feature detection; ``native`` selects an architecture, not the
build machine's CPU generation. The wheel does not require an installed LLVM.
OpenSSL and other non-system shared dependencies must still be bundled by the
normal wheel repair step. ``ZVEC_SCALAR_JIT_LLVM_SHARED=ON`` explicitly selects
shared LLVM instead, which must be provided or bundled at runtime.

All linked static libraries and Gandiva bitcode must be built for the wheel's
minimum OS/ABI and CPU baseline. Static linking does not lower a dependency's
deployment target or undo CPU-specific compiler flags. Do not use
``-march=native`` for portable release artifacts. Runtime JIT code uses the
execution machine's CPU features; compiled predicates are process-local and
are not shipped in the wheel. Cross, mobile, Windows and universal macOS builds
are unsupported when ON. OFF adds no LLVM/OpenSSL dependency.
Set ``ZVEC_SCALAR_JIT=0`` before the first query to disable the fast path.

Update the patch, extension and provenance hashes together. Apply revised
patches to a clean Arrow source tree; conflicts must be reviewed rather than
overwriting upstream files. The extension still uses Gandiva internals:
upgrading Arrow requires rebuilding and validating scalar and batch semantics
before updating the version guard. Clean patch application alone does not
establish compatibility.
