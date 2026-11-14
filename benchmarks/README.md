# Benchmarks

This directory contains benchmarks for zvec subsystems.

## Build

Benchmarks are disabled by default. From the repository root:

```bash
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCHMARKS=ON

cmake --build build-bench --target <benchmark-target>
```

Replace `<benchmark-target>` with a target from the subsystem's README, which
also documents its inputs and run commands. Executables are written to
`build-bench/bin/`.
