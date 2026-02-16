# zvec test runner
# Usage: python doscript.py test.do [cpp|python|all]

global_variable = suite, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root (e.g. C:\Users\User\zvec-0.2.0):"
end_if

say 'Using project root: {root}'



if arg1 == ""
  ask suite "Which tests? (cpp/python/all):"
else
  suite = arg1
end_if

if suite == ""
  suite = "all"
end_if

if suite == "cpp"
  say "Running C++ unit tests..."
  run 'cmake --build "{root}\build" --target unittest --parallel 8'
  say "C++ tests complete"
end_if

if suite == "python"
  say "Installing Python package in dev mode..."
  run "pip install -e python/"
  say "Running Python tests..."
  run "py -m pytest python/tests/ -v"
  say "Python tests complete"
end_if

if suite == "all"
  say "Running C++ unit tests..."
  run 'cmake --build "{root}\build" --target unittest --parallel 8'
  say "Installing Python package in dev mode..."
  run "pip install -e python/"
  say "Running Python tests with coverage..."
  run "py -m pytest python/tests/ --cov=zvec --cov-report=term-missing -v"
  say "All tests complete"
end_if
