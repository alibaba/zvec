# Zvec C++ / C headers

This directory is the SDK's `include/` source tree. Not everything in it carries
the same stability promise, and not everything in it is installed by default, so
read this before including a header or adding one.

## What actually ships

`src/include` is a build directory, not the published surface: it also holds the
engine internals. CMake installs two different sets.

- **Always installed** — the stable API and its transitive include closure,
  listed in `cmake/zvec_public_headers.cmake` (18 headers today). That manifest
  is generated, not hand-edited:

  ```bash
  python3 scripts/check_public_headers.py --write-install-manifest
  ```

- **Installed only with `-DZVEC_INSTALL_ENGINE_HEADERS=ON`** (*not* the default)
  — the remaining `zvec/core/**`, `zvec/ailego/**` and `zvec/turbo/**` headers,
  taking the package from 18 to 95. Turn it on only if you deliberately want to
  build against the engine; see "Advanced, unstable headers" below for what that
  costs you in stability. Building zvec from source needs no switch: the engine
  headers are right here in the source tree.

  `core` and `ailego` are one switch, not two: `zvec/core/**` reaches 31 of the
  35 `zvec/ailego/**` headers transitively, so publishing core without ailego
  would ship headers that do not compile.

## Libraries

A released package carries **two shared libraries**, and the headers above are
served by them:

| Library | Serves |
|---------|--------|
| `libzvec` | the whole C++ surface — `zvec/db/**` and `zvec/c_api.h`'s C++ neighbours, plus the engine headers when they are installed |
| `libzvec_c_api` | the C API |

`libzvec` whole-archives every component and links against a whitelist generated
from these headers (`scripts/generate_export_whitelist.py`), so it exports what a
consumer can name here and nothing else — no `zvec::sqlengine`, no `zvec::fts`,
no `*Impl`. The whitelist is generated at the *engine* surface (all 95 headers)
even though a default package ships 18, so a source-tree consumer that sees the
engine headers can still link the engine API out of `libzvec`; from a prebuilt
package those extra exports are simply unreachable, because nothing there
declares them.

The standalone `libzvec_core` / `libzvec_ailego` are built but **not packaged**
by default; they need `-DZVEC_INSTALL_ENGINE_HEADERS=ON` *and*
`-DZVEC_INSTALL_ENGINE_SHARED_LIBRARIES=ON`. If you linked `-lzvec_core` or
`-lzvec_ailego`, link `-lzvec` instead: same symbols, one library.

The static archives follow the *headers*, not the library switch: `libzvec.a`
always installs (since `zvec/db/**` always does), and `libzvec_core.a` /
`libzvec_ailego.a` / `libzvec_turbo.a` install only with
`ZVEC_INSTALL_ENGINE_HEADERS=ON` — an archive whose declarations were not
installed is a library that cannot be called. The mobile SDK targets
`zvec::static`, `zvec::core_static` and `zvec::ailego_static` are build-tree only
and read headers from this directory rather than from an install tree, so
neither switch reaches them.

## Stable public API

| Header | Use it for |
|--------|-----------|
| `zvec/db/collection.h` | Main entry point: `Collection::CreateAndOpen()` / `Collection::Open()`. Start here. |
| `zvec/db/schema.h` | `CollectionSchema`, `FieldSchema` |
| `zvec/db/index_params.h` | Per-column index configuration (`HnswIndexParams`, `IvfIndexParams`, `FtsIndexParams`, …) |
| `zvec/db/query_params.h` | Per-query tuning (`HnswQueryParams`, `IvfQueryParams`, …) |
| `zvec/db/query.h` | `SearchQuery`, `MultiQuery`, `GroupByVectorQuery`, query clauses |
| `zvec/db/doc.h` | `Doc`: the row/document value type |
| `zvec/db/doc_iterator.h` | Snapshot iteration over a collection |
| `zvec/db/reranker.h` | Hybrid-search rerank strategies |
| `zvec/db/options.h` | `CollectionOptions` and the DDL/maintenance option structs |
| `zvec/db/config.h` | `GlobalConfig`: process-wide logging and thread settings |
| `zvec/db/status.h` | `Status`, `StatusCode`, `Result<T>` |
| `zvec/db/stats.h`, `zvec/db/type.h` | Stats payload and the public enums |
| `zvec/c_api.h` | The C API. This is what every language SDK binds to. |
| `zvec/export.h` | Visibility macros (`ZVEC_API` and friends) |

These follow semantic versioning: they stay source- and ABI-compatible within a
major version.

## Advanced, unstable headers

`zvec/core/**`, `zvec/ailego/**` and `zvec/turbo/**` are the engine framework,
the base library and the SIMD kernels. They are **not installed by default** —
pass `-DZVEC_INSTALL_ENGINE_HEADERS=ON`, or build zvec from source, where
`examples/c++/core` and `examples/c++/ailego` link them out of `libzvec`.

Either way they are **internal APIs**: they may change or disappear in any
release without notice, and they are not covered by the compatibility promise
above. Being exported is not the same as being supported.

Do not include them from application code unless you deliberately want to build
against the engine itself. In particular, `zvec/db/**` must not depend on
`zvec/core/framework/**`.

Three engine headers are the exception: `zvec/core/interface/constants.h`,
`zvec/ailego/pattern/expected.hpp` and `zvec/ailego/utility/float_helper.h` are
reached from `zvec/db/**` itself, so they always ship. They are still not API —
treat them as implementation detail that happens to be visible. Removing them
from the closure is tracked in the header-governance doc (`Result<T>` and
`float16_t` are what keep them there).

## Rules for contributors

1. **Internal helpers do not go in this directory.** Put them next to their
   implementation under `src/**` and name the file `*_internal.h`
   (e.g. `src/db/index/common/query_internal.h`). Existing examples:
   `src/db/doc_iterator_internal.h`, `src/db/collection_query_internal.h`,
   `src/db/index/common/doc_internal.h`,
   `src/db/index/common/hnsw_rabitq_params_internal.h`.
2. **A member that only the engine calls is not public API.** Either make it
   `private` and grant access through a `friend` accessor declared in an
   internal header (the `zvec::internal::DocAccess` and
   `zvec::detail::HnswRabitqRuntimeAccess` pattern), or move it out of the class
   into a free function in `zvec::internal`.
3. **Do not let engine types appear in a stable header.** Opaque members
   (`std::shared_ptr<detail::Something>` with a forward declaration) and Pimpl
   keep `zvec/core/**` and heavy third-party headers out of the transitive
   closure. See `DocIterator::Impl` and `FtsIndexParams`' `detail::FtsState`.
4. **Prefer forward declarations and `<iosfwd>`** over full includes; move
   `to_string()` / `operator==` bodies into `.cc` files when they drag in extra
   headers.
5. **Keep the include budget in check.** `scripts/check_public_headers.py`
   enforces the dependency allowlist, a transitive-include budget for the entry
   headers, and agreement between the include graph and the install manifest. It
   runs in pre-commit and in the Lint workflow:

   ```bash
   python3 scripts/check_public_headers.py --verbose --compile-check
   ```

   After changing a public header, regenerate the manifest (see above) and
   tighten the budgets in the script to the new measured values — they are what
   turns "this header got heavier" into a CI failure.
6. **Adding a header to the stable closure is a decision, not a side effect.**
   Every non-`zvec/db` include reached from a stable header must be listed in
   `ALLOWED_STABLE_DEPS` or `INDIRECT_STABLE_DEPS` with a reason, because it
   then ships in every SDK package forever. Prefer an opaque member or a
   self-contained implementation in `zvec/db/**`.
