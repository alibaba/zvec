Gandiva scalar predicate backend
===============================

This directory contains the scalar predicate facade, parameter binding and
fixed-capacity LRU cache used by zvec's optional forward filter path.
``prepare_arrow.cmake`` applies the version-specific Gandiva overrides to
zvec's existing Arrow 21.0.0 source; both use the same Arrow build.
``PROVENANCE.json`` records the source versions and hashes. ``LICENSE`` and
``NOTICE`` retain the Apache Arrow licensing information.

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

Execution
---------

Each query binds immutable Arrow buffers and parameter values. The compiled
predicate evaluates only the requested row, with no batch-sized output or
temporary bitmap. Compiled expressions are shared through a 128-entry LRU;
changing literal values or storage addresses does not require recompilation.

The zvec adapter accepts same-type numeric comparisons, Kleene AND/OR, NOT
and null tests on persisted mmap columns. Unsupported expressions, storage
layouts and compilation/binding failures use the existing Arrow evaluator.
IN is excluded because Gandiva's NULL semantics differ from zvec's.

Update the facade, overrides and provenance hashes together. These headers
change Gandiva's internal ABI and must not be linked with an unmodified
installed Gandiva library.
