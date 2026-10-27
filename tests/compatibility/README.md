# Factory and extension compatibility

The factory facade and `INDEX_FACTORY_REGISTER_*` macros remain supported.
`CreateQuantizer`, `HasQuantizer`, and `AllQuantizers` include custom names and
aliases. Built-in registration inserts defaults without replacing user entries.
Named construction, including index creation and metadata reconstruction, uses
the registry so same-name overrides work on both build and reopen. Explicit
construction of concrete quantizer classes remains available.

The built-in registration roots reference ordinary functions in their archives.
The turbo archive owns its quantizer root independently of core. Generic
`ailego::Factory<T>` access loads the relevant defaults, even before any call to
`IndexFactory`. Shared SDKs also initialize their roots at load time for clients
compiled with the older header.

## Static extensions

For targets built in the same CMake project, legacy registration macros in C/C++
sources are detected automatically. A generated symbol in each registration
translation unit is referenced by the final executable or shared library. This
works with native `add_library`/`add_executable` and the `cc_*` helpers.

For indirect registration (for example a project-specific macro or arbitrary
static initializer), retain the existing `cc_library(... ALWAYS_LINK ...)`
option, or set `ALWAYS_LINK` on a native library target:

```cmake
set_property(TARGET my_extension PROPERTY ALWAYS_LINK TRUE)
target_link_libraries(my_app PRIVATE my_extension)
```

This retains its C/C++ source objects through generated references, without
archive-forcing linker flags. Generated sources that do not exist at configure
time should use an explicit initialization entry point instead.

An opaque, prebuilt static archive cannot be discovered from static
constructors alone. Expose an `extern "C"` initialization function that performs
the registrations (or references every registration object in the archive):

```cpp
extern "C" void RegisterMyQuantizers() {
  static zvec::ailego::Factory<zvec::turbo::Quantizer>::Register<MyQuantizer>
      registration("MyQuantizer");
}
```

Then use the helper when building with zvec's CMake integration:

```cmake
zvec_link_static_extension(TARGET my_app LIBRARY my_extension
                          INITIALIZER RegisterMyQuantizers)
```

Standalone consumers can link the archive normally and call that function
before opening indexes. Archives with no suitable externally referenceable
entry point must be rebuilt; they cannot be retained transparently without
forcing archive extraction.

## ABI and regression coverage

`factory_abi_test` compiles separate translation units against the current
header and the unmodified `ea60a361` header under `legacy/`. It checks singleton
allocation in both orders, layout size, bidirectional registration and lookup,
overrides, and callbacks that reenter the current factory. Synchronization lives
outside the legacy object layout. Already-compiled old registration code still
requires its original external synchronization; it cannot acquire a mutex that
was absent from its header. ABI compatibility assumes matching platform,
compiler ABI, and C++ runtime.

Other tests cover native component and turbo-only consumers, registrar-only
static archives, startup overrides, `ALWAYS_LINK` initializers without factory
headers, imported archives with explicit initializers, custom names/aliases in
persisted Flat indexes, and high-level index creation/reopening with an override.
`ivf_searcher_test` additionally covers custom aliases for scalar and trained PQ
postings and rejects unknown names. The Flat persistence cases cover both the
current quantizer metadata field and the legacy reformer-name field.

Persistence tests generate their indexes at runtime. The suite does not include
checked-in pre-refactor binary fixtures or a cross-version serialized-format
reader test. The legacy-header ABI tests remain independent of index files.
