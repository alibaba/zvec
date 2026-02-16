# zvec build script - FIXED with single quotes
global_variable = build_type, jobs, root, cmake_check, submodule_check

say "═══════════════════════════════════"
say " zvec Build Script"
say "═══════════════════════════════════"

# Auto-detect project root
root = capture "cd"
cmake_check = capture 'if exist "CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
    say "CMakeLists.txt not found in current directory."
    ask root "Enter zvec project root:"
end_if

say 'Using project root: {root}'   # ✅ Single quotes!
say ""

# Get build type
if arg1 == ""
    ask build_type "Build type? (Release/Debug/RelWithDebInfo):"
else
    build_type = arg1
end_if

if build_type == ""
    build_type = "Release"
end_if

# Get parallel jobs
if arg2 == ""
    ask jobs "Parallel jobs? (e.g. 8):"
else
    jobs = arg2
end_if

if jobs == ""
    jobs = "8"
end_if

say ""
say "Configuration:"
say '  Build Type: {build_type}'    # ✅ Single quotes!
say '  Jobs: {jobs}'                # ✅ Single quotes!
say ""

# Check for git submodules
say "Checking dependencies..."
submodule_check = capture 'if exist "{root}\thirdparty\googletest\googletest-1.10.0\CMakeLists.txt" (echo yes) else (echo no)'

if submodule_check == "no"
    say "→ Git submodules not found"
    say "Please run manually:"
    say '  cd {root}'                 # ✅ Single quotes!
    say "  git submodule update --init --recursive"
    say ""
    say "Or download zvec with dependencies included"
    exit 1
else
    say "✓ Dependencies present"
end_if

# Create build directory
say "→ Creating build directory..."
run 'mkdir "{root}\build" 2>nul'

# Configure CMake
say '→ Configuring CMake ({build_type})...'
run 'cmake -S "{root}" -B "{root}\build" -DCMAKE_BUILD_TYPE={build_type} -DBUILD_TOOLS=ON'

# Check if CMake succeeded
cmake_cache_check = capture 'if exist "{root}\build\CMakeCache.txt" (echo yes) else (echo no)'
if cmake_cache_check == "no"
    say ""
    say "✗✗✗ CMake Configuration FAILED ✗✗✗"
    say "Check error messages above for details."
    exit 1
end_if

say "✓ CMake configured successfully"

# Build
say '→ Building with {jobs} parallel jobs...'
run 'cmake --build "{root}\build" --config {build_type} --parallel {jobs}'

say ""
say "═══════════════════════════════════"
say "✓ BUILD COMPLETE!"
say '  Binaries: {root}\build\{build_type}'
say "═══════════════════════════════════"
