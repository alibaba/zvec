Scalar predicates in the zvec forward-filter chain
=================================================

This branch starts at upstream/main b0d0760a5df786c6f5ddee47b9c8c0d7dc9a0c18.
The optional Gandiva backend replaces only DocFilter's per-candidate forward
expression evaluation. Delete and inverted filters retain their order, and
the planner still decides whether a query uses inverted or forward filtering.
The feature does not replace bitmap-producing plans or change HNSW parameters.

Query preparation and evaluation
-------------------------------

1. Bind the existing Arrow Compute expression to the original field schema.
2. Obtain an owning snapshot of complete immutable mmap columns from Segment,
   under its shared lock. Mutable writing stores and incomplete layouts cannot
   supply this snapshot and use the original evaluator.
3. Translate an explicit semantic whitelist to Gandiva's typed expression tree.
4. Use a process-local, thread-safe LRU with capacity 128 compiled expressions.
   The key contains structure, operators, fields and types. Scalar literals
   become runtime parameters; their values and storage addresses are excluded.
   Changing a comparison threshold reuses the same compiled code.
5. Bind parameters once per query. Align column chunk boundaries using metadata
   and build flat buffer-pointer/offset descriptors, without copying data or
   creating a RecordBatch for each chunk. Different queries own distinct
   parameter slots and descriptors.
6. For each candidate, locate its batch and call the scalar function at the
   batch-local offset. The callback allocates no Arrow Scalar/ExecBatch, takes
   no forward-store lock, and evaluates only that row.

Uniform chunks use division to find the batch; irregular layouts use binary
search. Array slice offsets apply independently to values and validity bits.
The snapshot owns all buffers, including chunks after the first. The prototype
binding owns parameter storage, holders and compiled code. Cache eviction does
not invalidate an active query. Completed cached programs are bounded by 128;
active queries and in-flight compilations can temporarily retain additional
code, so capacity is not a hard process-memory limit.

Persisted Arrow metadata is validated when its column arrays are published
(segment open or schema refresh). Snapshot acquisition verifies this validation
state, column types and the actual segment-local row count, ``doc_ids_.size()``.
It does not use metadata doc_count, which may also count independent added-column
blocks. The general ScalarDocFilter::Make API still validates arbitrary tables;
only DocFilter can use the private, prevalidated snapshot entry point.

Supported expressions and fallback
----------------------------------

The adapter accepts same-type fixed-width numeric comparisons, boolean
AND/OR with Kleene semantics, NOT, IS NULL and IS VALID, and numeric/bool
fields and scalar literals. It preserves intermediate validity and maps a NULL
final predicate to non-match, as the original forward filter does. The bridge
tests include integer extremes, floating NaN/infinities, typed NULLs and slices.

Unsupported expressions fall back as a whole. This includes strings, arrays,
casts, arithmetic, mixed-type comparisons requiring casts, IN, nested fields,
nonstandard function options and bool comparisons. Gandiva 21's numeric IN
has different NULL behavior, so it is deliberately excluded. General NOT is
supported by the bridge, although the current public SQL grammar exposes only
its more specific NOT IN/CONTAIN forms.

Translation, compilation or binding errors leave the original Arrow evaluator
in place. Mutable stores, non-mmap storage and unavailable snapshots also fall
back. Delete filters and group-by can coexist with compiled forward predicates;
no segment lock is held across the HNSW search.

Build and platform boundary
---------------------------

``ZVEC_ENABLE_SCALAR_JIT`` defaults to OFF. An OFF build neither discovers LLVM
nor builds Gandiva; its filter chain remains the Arrow implementation. For ON,
the existing pinned Arrow 21.0.0 submodule builds both Arrow and Gandiva statically
against the same Arrow instance. The small scalar facade and source overrides
are vendored under thirdparty/gandiva_scalar with licenses and source hashes.

The current integration targets native 64-bit Linux/macOS with LLVM 20's shared
library and matching clang/llvm-link tools. Native C++ code can keep the normal
zvec compiler; only Gandiva bitcode requires the matching clang. Cross builds,
universal binaries, Windows, Android and iOS use the OFF path; explicitly asking
for ON there produces a configuration error. This change has been executed on
macOS arm64; Linux and other platforms have not been runtime-tested here.
An ON binary still requires its LLVM/OpenSSL runtime libraries to load;
expression fallback does not remove that deployment dependency.

See thirdparty/gandiva_scalar/README.rst for dependency/build commands. In an ON
build, ``ZVEC_SCALAR_JIT=0`` selects the old Arrow path for an entire process;
the default enables fast-path attempts. The environment variable is read once
on first use. It is a diagnostic opt-out, not a per-query mutable setting.

Validation and benchmarks
-------------------------

The adjacent README.md documents the public API smoke and 10M replay runners.
Internal counters count successful query bindings and cache activity, never
individual candidate callbacks. The replay verifies activation and that changing
thresholds causes no additional compilation after warmup.

With BUILD_TESTING=ON and the backend enabled, build these targets::

  cmake --build build-scalar-jit --target \
    scalar_predicate_expr_test scalar_doc_filter_test \
    add_column_batch_boundaries_test collection_float_range_filter_test

The optional internal metadata benchmark is built from the main project::

  cmake --build build-scalar-jit --target scalar_filter_binding_benchmark
  build-scalar-jit/bin/scalar_filter_binding_benchmark --iterations 1000

Its public Make timing includes full table validation, unlike the production
prevalidated-snapshot path. Validate is also reported independently; do not add
the two timings. Its uniform-random candidates are synthetic, not a captured
HNSW traversal. End-to-end performance must come from the public replay.

Cold compilation still occurs on the first occurrence of a new structure or
after eviction. Warm-query results do not include this first-use cost; the cache
avoids recompilation when only runtime comparison parameters change.
