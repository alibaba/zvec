# zvec coverage report  
# Usage: do doscript\coverage.do [open|noopen]
# Note: Requires gcov/lcov (typically available in Git Bash or WSL on Windows)

global_variable = open_report, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root:"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask open_report "Open HTML report when done? (y/n):"
else
  open_report = arg1
end_if

say "Cleaning previous build..."
delete folder "build"
run 'mkdir "{root}\build" 2>nul'

say "Configuring with coverage flags..."
run 'cmake -S "{root}" -B "{root}\build" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage -DBUILD_TOOLS=ON'

say "Building..."
run 'cmake --build "{root}\build" --parallel 8'

say "Running unit tests..."
run 'cmake --build "{root}\build" --target unittest --parallel 8'

say "Generating coverage data..."
say "NOTE: Coverage generation requires bash/gcov (Git Bash or WSL)"
// Try Git Bash first, then WSL
run 'bash "{root}\scripts\gcov.sh" -t gcov -o "{root}\build\coverage_html" || wsl bash "{root}/scripts/gcov.sh" -t gcov -o "{root}/build/coverage_html"'

say 'Coverage report ready at {root}\build\coverage_html\index.html'

if open_report == "y"
  open_link 'file:///{root}/build/coverage_html/index.html'
end_if
