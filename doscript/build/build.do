# zvec build script
# Usage: do doscript\build.do [Release|Debug] [jobs]

global_variable = build_type, jobs, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root (e.g. C:\Users\User\zvec-0.2.0):"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask build_type "Build type? (Release/Debug/RelWithDebInfo):"
else
  build_type = arg1
end_if

if build_type == ""
  build_type = "Release"
end_if

if arg2 == ""
  ask jobs "Parallel jobs? (e.g. 8):"
else
  jobs = arg2
end_if

if jobs == ""
  jobs = "8"
end_if

say "Preparing build directory..."
run 'mkdir "{root}\build" 2>nul'

say 'Configuring CMake ({build_type})...'
run 'cmake -S "{root}" -B "{root}\build" -DCMAKE_BUILD_TYPE={build_type} -DBUILD_TOOLS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5'

say 'Building with {jobs} parallel jobs...'
run 'cmake --build "{root}\build" --parallel {jobs}'

say 'Build complete! Binaries in {root}\build'
