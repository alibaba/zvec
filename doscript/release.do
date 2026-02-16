# zvec release packager
# Usage: do doscript\release.do [version] [y|n for python wheel]

global_variable = version, include_python, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root:"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask version "Release version tag? (e.g. 0.2.0):"
else
  version = arg1
end_if

if arg2 == ""
  ask include_python "Include Python wheel? (y/n):"
else
  include_python = arg2
end_if

if version == ""
  version = "dev"
end_if

say "Building release binaries..."
run 'mkdir "{root}\build" 2>nul'
run 'cmake -S "{root}" -B "{root}\build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TOOLS=ON -DBUILD_PYTHON_BINDINGS=OFF'
run 'cmake --build "{root}\build" --config Release --parallel 8'

say 'Preparing dist\zvec-{version}...'
run 'mkdir "{root}\dist\zvec-{version}" 2>nul'
run 'mkdir "{root}\dist\zvec-{version}\lib" 2>nul'
run 'mkdir "{root}\dist\zvec-{version}\include" 2>nul'
run 'mkdir "{root}\dist\zvec-{version}\tools" 2>nul'

say "Copying library files..."
// Windows: Copy lib files from Release or Debug directory
run 'xcopy /Y /Q "{root}\build\src\Release\*.lib" "{root}\dist\zvec-{version}\lib\" 2>nul || xcopy /Y /Q "{root}\build\src\*.lib" "{root}\dist\zvec-{version}\lib\" 2>nul'
run 'xcopy /Y /Q "{root}\build\src\Release\*.dll" "{root}\dist\zvec-{version}\lib\" 2>nul || xcopy /Y /Q "{root}\build\src\*.dll" "{root}\dist\zvec-{version}\lib\" 2>nul'

say "Copying headers..."
run 'xcopy /E /I /Y /Q "{root}\src\include\zvec" "{root}\dist\zvec-{version}\include\zvec\"'

say "Copying tools..."
run 'copy /Y "{root}\build\tools\Release\bench.exe" "{root}\dist\zvec-{version}\tools\" 2>nul || copy /Y "{root}\build\tools\bench.exe" "{root}\dist\zvec-{version}\tools\" 2>nul'
run 'copy /Y "{root}\build\tools\Release\txt2vecs.exe" "{root}\dist\zvec-{version}\tools\" 2>nul || copy /Y "{root}\build\tools\txt2vecs.exe" "{root}\dist\zvec-{version}\tools\" 2>nul'

say "Copying docs..."
copy "README.md" to 'dist\zvec-{version}\README.md'
copy "LICENSE" to 'dist\zvec-{version}\LICENSE'

if include_python == "y"
  say "Building Python wheel..."
  run "pip install build"
  run "py -m build --wheel"
  run 'copy /Y "{root}\dist\*.whl" "{root}\dist\zvec-{version}\" 2>nul'
end_if

say "Creating archive..."
zip 'dist\zvec-{version}' to 'dist\zvec-{version}.zip'

say 'Release package ready: dist\zvec-{version}.zip'
