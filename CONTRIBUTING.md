# Contributing to Zvec

First off, thank you for considering contributing to Zvec! 🙌  
Whether you're reporting a bug, proposing a feature, improving documentation, or submitting code — every contribution helps make Zvec better.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful, collaborative, and inclusive.

---

## Development Setup

> [!TIP]
> **Linux** is the recommended environment for development and performance benchmarking.

### Prerequisites

- Python 3.10 - 3.14 (64-bit only)
- CMake ≥ 3.26, < 4.0 (`cmake --version`)
- A C++17-compatible compiler (e.g., `g++-11+`, `clang++`, Apple Clang on macOS)

### Clone & Initialize

```bash
git clone --recursive https://github.com/alibaba/zvec.git
cd zvec
```

> 💡 **Tip**  
> - Forgot `--recursive`? Run:  
>   ```bash
>   git submodule update --init --recursive
>   ```
> - Set up pre-commit hooks:  
>   ```bash
>   pip install pre-commit && pre-commit install
>   ```

### Build from Source (Editable Install)
```bash
pip install -e ".[dev]"
# This installs dev dependencies (pytest, ruff, etc.) and builds the C++ extension in-place
```

> ✅ Verify:
> ```bash
> python -c "import zvec; print('Success!')"
> ```

---

## Testing

### Run All Tests
```bash
pytest python/tests/ -v
```

### Run with Coverage (Debug/CI)
```bash
pytest python/tests/ --cov=zvec --cov-report=term-missing
```

> 🔎 See full rules in `[tool.ruff]` section of `pyproject.toml`.

---

## Build Customization

You can control build behavior via environment variables or `pyproject.toml`:

| Option | How to Set | Description |
|--------|------------|-------------|
| **Build Type** | `CMAKE_BUILD_TYPE=Debug` | `Debug`, `Release`, or `Coverage` (for gcov/lcov) |
| **Generator** | `CMAKE_GENERATOR="Unix Makefiles"` | Default: `Ninja`; use Make if preferred |
| **AVX-512** | `ENABLE_SKYLAKE_AVX512=ON` | Enable AVX-512 optimizations (x86_64 only) |

Example (Debug + Make):
```bash
CMAKE_BUILD_TYPE=Debug CMAKE_GENERATOR="Unix Makefiles" pip install -v .
```

---

## C++ Public API

Only `src/include/zvec/db/*.h` and `src/include/zvec/c_api.h` are the stable,
versioned API. `src/include/zvec/{core,ailego,turbo}/**` are the engine and may
change at any time; they are installed only because `libzvec_core` /
`libzvec_ailego` are published alongside them, and `-DZVEC_INSTALL_ENGINE_HEADERS=OFF`
produces a stable-API-only SDK. See
[`src/include/zvec/README.md`](src/include/zvec/README.md) for the boundary rules
(where internal helpers go, how to hide engine types, how to keep the include
budget in check).

The rules are enforced by `scripts/check_public_headers.py`, which runs in
pre-commit and in the Lint workflow:

```bash
python scripts/check_public_headers.py --verbose --compile-check
```

It also owns the install manifest. After adding, removing or re-including a
public header, regenerate it and commit the diff:

```bash
python scripts/check_public_headers.py --write-install-manifest
git diff cmake/zvec_public_headers.cmake
```

A manifest diff means the SDK's header set changed, so call it out in the PR
description: every header that enters the closure ships in every package from
then on.

### Exported symbols

`libzvec` whole-archives every component plus rocksdb, arrow, antlr and
protobuf, so it links with an exported-symbol whitelist rather than exporting
everything. `src/exported_symbols_zvec.{txt,lds}` is *generated* from the
published headers -- it is the list of names a consumer can actually write --
and is what keeps engine internals such as `zvec::sqlengine` out of the ABI
surface (measured 10784 -> 2602 exports on macOS/arm64).

The pre-commit hook and the Lint workflow check it for drift. After changing a
published header, regenerate it and commit the diff:

```bash
python scripts/generate_export_whitelist.py --write
git diff src/exported_symbols_zvec.txt
```

This needs `clang++` (the header set is parsed with a clang AST dump); without
it the check skips locally and fails in CI. The standalone engine libraries
(`libzvec_core`, `libzvec_ailego`) keep the coarser namespace-wide
`src/exported_symbols_cpp.*`, since their whole `zvec::core` / `zvec::ailego`
surface is their API. The lists apply to non-Debug builds only, so Debug keeps
full symbols for in-tree debugging and crash symbolication.

---

## Submitting Changes

1. Fork the repo and create a feature branch (`feat/...`, `fix/...`, `docs/...`)
2. Write clear commit messages (e.g., `fix(query): handle null vector in dense_fp32`)
3. Ensure tests pass & linter is clean
4. Open a Pull Request to `main`
5. Link related issue (e.g., `Closes #123`)

✅ **PRs should include**:
- Test coverage for new behavior
- Updates to documentation (if applicable)
- Reasoning behind non-obvious design choices

---

## Documentation

- User guides: `docs/` (built with MkDocs)
- API reference: generated from docstrings (follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings))
- Build & deploy: `mkdocs serve` / `mkdocs build`

---

## Need Help

- Browse [existing issues](https://github.com/alibaba/zvec/issues)
- For sensitive/security issues: email `zvec@alibaba-inc.com`

---

✨ Thanks again for being part of Zvec!
