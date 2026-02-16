# zvec DoScript Workflows

This folder contains [DoScript](https://github.com/TheServer-lab/DoScript) automation scripts for common zvec developer tasks.

## ⚠️ Important: Run from the project root

All scripts must be run from the **zvec project root directory**, not from inside the `doscript/` folder. This ensures relative paths like `build/`, `data/`, and `bench_results/` all resolve correctly.

```
cd C:\path\to\zvec-0.2.0
do doscript\build.do
```

Each script captures the working directory at startup (`cwd = capture "cd"`) so all file operations use consistent absolute paths.

## Scripts

| Script | Description |
|---|---|
| `build.do` | Configure and build the project |
| `test.do` | Run C++ unit tests, Python tests, or both |
| `format.do` | Check or auto-fix code formatting (clang-format + ruff) |
| `benchmark.do` | Download datasets and run index benchmarks |
| `coverage.do` | Build with coverage flags and generate HTML report |
| `release.do` | Package a release zip with binaries, headers, and optional Python wheel |
| `clean.do` | Remove build artifacts |

## Usage

### Interactive (prompts for input)
```
do doscript\build.do
do doscript\test.do
do doscript\benchmark.do
```

### With CLI args (no prompts)
```
do doscript\build.do Release 8
do doscript\test.do all
do doscript\clean.do all
do doscript\release.do 0.2.0 n
do doscript\benchmark.do hnsw sift 10
```

### Dry-run (preview without executing)
```
do doscript\release.do 0.2.0 n --dry-run
```

## DoScript syntax reference

| Syntax | Meaning |
|---|---|
| `global_variable = a, b` | Declare variables |
| `var = capture "cmd"` | Run command and store output in var |
| `run "cmd"` | Execute shell command |
| `run 'cmd {var}'` | Execute with variable interpolation (single quotes) |
| `copy "src" to "dst"` | Copy file |
| `zip "folder" to "archive.zip"` | Create zip archive |
| `download "url" to "path"` | Download file |
| `if x == "y" / else / end_if` | Conditionals (`==` not `=`) |
| `arg1`..`arg32` | CLI arguments passed after script name |
| `say 'Hello {name}!'` | Print with interpolation |
| `ask varname "prompt"` | Prompt user for input |
| `# comment` or `// comment` | Line comments |
