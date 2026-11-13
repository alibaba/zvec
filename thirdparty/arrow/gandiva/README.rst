Gandiva scalar execution extension
==================================

This directory extends Apache Arrow 21.0.0 Gandiva with single-row execution.
``prepare.cmake`` applies ``scalar.patch`` to zvec's existing Arrow build and
copies only new extension files. ``PROVENANCE.json`` records the upstream
version, extension hashes and canonical license sources. The complete Apache
Arrow license and third-party acknowledgments are retained in
``../apache-arrow-21.0.0/LICENSE.txt`` and ``../apache-arrow-21.0.0/NOTICE.txt``.
Wheel/SDK installation copies those unmodified texts as ``gandiva/LICENSE``
and ``gandiva/NOTICE`` alongside the provenance; artifact checks verify their
hashes. LLVM/OpenSSL toolchain license packaging is unchanged.

Project-owned files use zvec's Apache 2.0 header. Arrow-derived files retain
their ASF notices and separately identify zvec's local modifications.

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
Set ``ZVEC_SCALAR_JIT=0`` before loading zvec to disable the fast path.

Linux release packaging is exercised by the opt-in
``.github/workflows/scalar-jit-linux.yml`` workflow. Its pinned manylinux_2_28
image builds LLVM 20.1.8 and PIC OpenSSL from verified sources, then builds
and checks both a repaired wheel and the shared SDK. It tests the same artifacts
on the runner CPU, QEMU Haswell-v4 (AVX2, no AVX512) and QEMU Nehalem
(SSE4.2, no AVX), with runtime ON and OFF. Acceptance requires artifact SHA-256
identity, CPUID checks, sampled execution evidence and matching query results.
QEMU is a compatibility check, not a hardware performance benchmark. Arrow's
existing SSE4.2 x86 baseline is unchanged; generic LLVM/OpenSSL flags do not
make the full package support older SSE2-only CPUs. The experimental workflow
does not enable JIT in the normal release matrix.

Initial release qualification is limited to the experimental Linux x86-64
CPython 3.12 wheel and Linux SDK. Local macOS builds are development-only.
The workflow audits final licenses, private LLVM symbols, ABI ceilings and
effective runtime/helper compiler flags, with explicit dispatched-kernel
exemptions. QEMU does not replace physical CPU performance testing or full
release regression qualification. Ordinary wheel and SDK entry points remain
explicitly OFF; this workflow does not publish packages.

The unchanged planner only selects the callback when an entirely inverted
condition becomes entirely forward after optimization. LIKE and IN cannot
currently take that route. Artifact smoke checks their results on ordinary
plans; native scalar differential tests exercise their generated helper calls.
Other smoke predicates prove actual callback execution. A prepared binding is
not, by itself, execution evidence.

Admission and failure policy
----------------------------

Ready entries bind immediately. An admitted miss compiles synchronously outside
the cache lock. In-flight shapes and exhausted capacity bypass JIT immediately,
without waiting, queuing, background compilation or prewarming. Defaults are:

* 128 positive and 128 negative completed entries, bounded independently.
* Two concurrent compilations and 256 live programs, including evicted programs
  retained by active bindings. A live slot is reserved before compilation.
* 1,024 expression nodes, depth 64, and a 64 KiB structural key.
* 1 MiB copied parameter/helper inputs per preparation and 4,096 IN elements.

These are acceleration admission limits, not SQL-language limits. Rejected
expressions continue through Arrow. Checks precede recursive lowering and helper
construction and also cover direct compiler entry points. There is no hard
byte-based cap on LLVM's heap. Unsupported/malformed compiled shapes remain in
the bounded negative cache until eviction. Transient failures retry after
monotonic per-shape exponential backoff from one to 60 seconds. Admission
rejections and binding failures do not poison valid compiled shapes.
Deterministic initialization failures disable further attempts for the process;
allocation failures remain retryable.

Set environment options before loading zvec; they are frozen on first runtime
access and each cache retains immutable compiler options:

* ``ZVEC_SCALAR_JIT=0``: disable acceleration.
* ``ZVEC_SCALAR_JIT_MAX_COMPILATIONS`` and
  ``ZVEC_SCALAR_JIT_MAX_LIVE_PROGRAMS``: positive decimal process limits.
  Empty, malformed, zero and overflowing overrides disable acceleration with
  ``invalid_configuration``.
* ``ZVEC_SCALAR_JIT_DUMP_IR=1``: retain optimized IR, OFF by default. Internal
  compiler options can also explicitly request capture.
* ``ZVEC_SCALAR_JIT_PROFILE=1``: select the separate sampled evaluator at query
  preparation. The ordinary evaluator has no clocks or diagnostic counters.

Diagnostics and ownership
-------------------------

``zvec.scalar_jit_info()`` is present in ON and OFF builds and does not initialize
LLVM. Version 1 returns ``built``, ``enabled``, ``initialized``, ``disable_reason``,
``initialization_error`` (sanitized last status code), effective ``limits``,
``profile``, ``dump_ir``, OS-usable x86 ``cpu_features``, ``cache``, ``timings``,
``selections``, ``execution_samples`` and ``memory``. Counters are process-local,
best-effort diagnostic snapshots, not an atomic transaction across all owners.

Cache fields distinguish hits, misses, in-flight bypasses, admission rejections,
compilation attempts/failures, backoff, evictions and resident counts. Selection
counts describe preparations only. Timing categories contain counts and total
nanoseconds for initialization, compilation, miss lowering, binding/descriptor
construction and total preparation. Execution profiling independently samples
the first and every 1,024th call per thread/backend (JIT and Arrow); these are
sampled counts/times, never exact total evaluations. Unsampled calls only touch
thread-local state. Aggregation survives normal static destruction ordering.

Stable fallback identifiers are ``build_disabled``, ``runtime_disabled``,
``invalid_configuration``, ``forked_process``, ``unsupported_expression``,
``unsupported_snapshot``, ``expression_limit``, ``compilation_busy``,
``live_program_limit``, ``compile_backoff``, ``initialization_failed``,
``compile_failed`` and ``binding_failed``. JIT diagnostic messages use stable
reasons and status codes, not literal values or addresses.

Memory reporting distinguishes cache-resident program bytes from live programs,
bindings, descriptors and helper owners. Emitted object-file bytes are a code-size
proxy, not executable-page measurements. IR capacity and owned wrapper/vector/
parameter storage are counted at their ownership boundaries and released only
with the final owner, even after cache eviction or destruction. Borrowed Arrow
column buffers are excluded; shared snapshot references are reported separately.
LLVM heap, allocator overhead, registry, RE2 internals and hash-table overhead
are unmeasured. Do not interpret these metrics as total JIT RSS. Object-size
observation does not retain an extra object copy and is inactive in ordinary
Gandiva batch generation.

Process lifecycle
-----------------

Use spawn/exec workers: each initializes an independent runtime and cache. A
native PID guard rejects entry to inherited JIT/cache/compiler state in a fork
child before inherited locks or lazy initialization, returning ``forked_process``
for new preparations. Inherited LLVM/cache objects are neither reset nor destroyed
in the child. Diagnostics there report ``accounting_available=False`` and zero
counters rather than inspect inherited owners. Reusing inherited active
collections, bound predicates or ongoing query objects is unsupported. This is
not a claim of general zvec fork safety.

Process-level diagnostic/cache/registry owners intentionally survive static
teardown. Applications must join query threads before orderly shutdown. There is
no unsafe LLVM compilation cancellation during Python interpreter teardown.
Native tests use compiler/clock injection for failures and blocked-compiler fork
coverage; Python lifecycle tests use bounded subprocesses, spawn, quiescent fork,
concurrent first queries, isolated parameters and drained threaded shutdown.
Run lifecycle tests natively, outside QEMU.

Update the patch, extension and provenance hashes together. Apply revised
patches to a clean Arrow source tree; conflicts must be reviewed rather than
overwriting upstream files. The extension still uses Gandiva internals:
upgrading Arrow requires rebuilding and validating scalar and batch semantics
before updating the version guard. Clean patch application alone does not
establish compatibility.
