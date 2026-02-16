# zvec code formatter
# Usage: do doscript\format.do [check|fix]

global_variable = mode, root, cmake_check

// Auto-detect project root: use CWD if CMakeLists.txt is there, else ask
root = capture "cd"

cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root:"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask mode "Mode? (check/fix):"
else
  mode = arg1
end_if

if mode == ""
  mode = "check"
end_if

if mode == "check"
  say "Checking Python files with ruff..."
  run "ruff check ."
  run "ruff format --check ."
  say "Checking C++ files with clang-format..."
  // Windows: Use for loop to check C++ files
  run 'for /r src %%f in (*.cc *.h *.cpp *.hpp) do @clang-format --dry-run --Werror "%%f"'
  say "Format check passed - no changes needed"
end_if

if mode == "fix"
  say "Fixing Python files with ruff..."
  run "ruff check --fix ."
  run "ruff format ."
  say "Fixing C++ files with clang-format..."
  // Windows: Use for loop to format C++ files
  run 'for /r src %%f in (*.cc *.h *.cpp *.hpp) do @clang-format -i "%%f"'
  say "All formatting applied"
end_if
