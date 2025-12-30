# Contributing to Zvec

First off, thank you for considering contributing to Zvec! 🙌  
Whether you're reporting a bug, proposing a feature, improving documentation, or submitting code — every contribution helps make Zvec better.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful, collaborative, and inclusive.

---

## Development Setup

### Prerequisites
- Python ≥ 3.9
- CMake ≥ 3.15 (`cmake --version`)
- A C++17-compatible compiler (e.g., `g++-11+`, `clang++`, Apple Clang on macOS)

### Clone & Initialize
```bash
git clone --recursive https://github.com/your-org/zvec.git
cd zvec
```

> 💡 **Submodule tip**: If you forgot `--recursive`, run:  
> `git submodule update --init --recursive`

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
