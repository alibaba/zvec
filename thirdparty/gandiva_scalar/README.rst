Gandiva scalar predicate backend
===============================

This directory vendors the small scalar-predicate facade, runtime-parameter
binding, fixed-capacity LRU cache, and the Arrow 21 Gandiva source changes used
by zvec's optional scalar filter path. ``PROVENANCE.json`` records the snapshot,
upstream pin, and SHA-256 of each copied source. ``LICENSE`` and ``NOTICE``
retain the Apache Arrow licensing information.

The sources are self-contained: building zvec does not require an installation
or checkout of the standalone experiment. The existing Arrow submodule remains
the sole source of Arrow and Gandiva. At configuration time
``prepare_arrow.cmake`` copies the three Gandiva overrides and its scalar ABI
header into that submodule, alongside zvec's existing Arrow patches.

Build
-----

The default ``ZVEC_ENABLE_SCALAR_JIT=OFF`` does not discover LLVM or OpenSSL and
does not build Gandiva. Enable the experimental backend explicitly::

  git submodule update --init --recursive
  cmake -S . -B build-scalar-jit -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DZVEC_ENABLE_SCALAR_JIT=ON \
    -DLLVM_ROOT=/path/to/llvm-20
  cmake --build build-scalar-jit --parallel 8

Requirements are a native 64-bit Linux/macOS C++17 toolchain, LLVM 20.x with its
shared ``LLVM`` library, matching ``clang`` and ``llvm-link`` tools, and OpenSSL
development files. LLVM 20.1.8 was used for the standalone snapshot. Cross,
mobile, Windows, and universal macOS builds are rejected when this option is
enabled. They retain the default backend with the option disabled.

The regular C/C++ compiler is not changed by this option. For example, macOS
can keep ``/usr/bin/clang`` and ``/usr/bin/clang++`` for zvec and Arrow while
setting ``LLVM_ROOT`` to a Homebrew LLVM 20 prefix. Gandiva's bitcode helper
compilation is independently pinned to the ``clang`` and ``llvm-link`` from that
LLVM installation. This avoids loading bitcode from an incompatible LLVM
release. An explicit ``OPENSSL_ROOT_DIR`` can be provided if OpenSSL is outside
CMake's normal search paths.

``ARROW.BUILD`` produces both ``libarrow.a`` and ``libgandiva.a``. The imported
``Gandiva::gandiva_static`` target links the existing ``Arrow::arrow_static``;
there is no second Arrow build or package lookup. LLVM and OpenSSL remain
runtime system dependencies. The vendored facade target is
``zvec_gandiva_scalar`` (alias ``Zvec::gandiva_scalar``), and its consumers get
``ZVEC_ENABLE_SCALAR_JIT=1``. It is linked into both the component index target
and the combined zvec target.

Arrow's bundled dependency download locations are unchanged. Existing downloaded
archives can be copied into the matching ExternalProject download directories,
or supplied using Arrow's ``ARROW_*_URL`` environment variables. Reuse downloaded
source archives rather than an old CMake cache or installed Arrow libraries:
Gandiva and its scalar ABI must be rebuilt with the same Arrow configuration.

Execution contract
------------------

The compiler generates a true single-row function::

  bool predicate(buffers, offsets, holders, row);

Only the requested row is evaluated. There is no batch filter invocation,
selection vector, output bitmap, or temporary storage proportional to batch
length. Intermediate validity uses per-call scalar locals. Parameter buffers
always use element/validity bit zero; data buffers use the requested row and
the Arrow array slice offset.

``ScalarPredicate::MakeParameterized`` compiles a data schema, expression, and
parameter schema. ``Bind(batch, params)`` snapshots typed parameter values and
keeps the batch and compiled code alive. Bound objects can be evaluated
concurrently against immutable Arrow data. ``ScalarPredicateCache`` extracts
numeric/bool literals and reuses compiled expression structure in a bounded
LRU (default capacity 128). Literal values and NULL status are runtime
parameters; numeric IN sets remain compile-time values and participate in the
cache key. The cache is in-memory only and has no lock in ``Evaluate``.

The facade supports a restricted numeric/bool subset. Functions requiring
allocation, execution context, function holders, or error reporting are
rejected. AND/OR/NOT and IF retain intermediate validity; the final predicate
maps an invalid result to false. Numeric IN follows Gandiva 21, which treats
NULL IN (...) as valid false, making NOT(IN(NULL)) true rather than SQL UNKNOWN.
Callers must preserve zvec's original semantics when choosing supported
expressions and retain the original filter path for unsupported or failed
compilation/binding cases.

Updating the snapshot
---------------------

Update the facade/cache sources, all matching overrides, and their provenance
hashes together. The override is version-specific to Arrow 21.0.0. Do not link
an installed unmodified Gandiva library with these headers: ``LLVMGenerator``
contains new members and entry points. Arrow's batch evaluator remains present
and uses the original row-indexed bitmap path when scalar mode is disabled.
