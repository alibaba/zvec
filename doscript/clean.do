# zvec clean
# Usage: do doscript\clean.do [build|all]

global_variable = level, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root:"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask level "Clean level? (build/all):"
else
  level = arg1
end_if

if level == ""
  level = "build"
end_if

if level == "build"
  say "Removing build directory..."
  delete folder "build"
  say "Build directory cleaned"
end_if

if level == "all"
  say "Removing build directory..."
  delete folder "build"
  say "Removing dist directory..."
  delete folder "dist"
  say "Removing data directory..."
  delete folder "data"
  say "Removing bench_results..."
  delete folder "bench_results"
  say "Removing Python cache..."
  // Windows: Remove __pycache__ directories
  run 'for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul'
  // Remove .pyc files
  run 'del /s /q *.pyc 2>nul'
  // Remove .egg-info directories
  run 'for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d" 2>nul'
  say "Full clean complete"
end_if
