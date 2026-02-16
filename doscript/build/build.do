# zvec build script - Cross-platform CMake automation
# Usage: do build.do [Release|Debug] [jobs]

global_variable = build_type, jobs, root

say "═══════════════════════════════════"
say " zvec Build Script (DoScript)"
say "═══════════════════════════════════"

# Auto-detect project root
if_exists "CMakeLists.txt"
    root = capture "cd"
    say "✓ Found CMakeLists.txt in current directory"
else
    say "⚠ CMakeLists.txt not found"
    ask root "Enter zvec project root:"
end_if

say "Project root: {root}"
say ""

# Get build type
if arg1 == ""
    ask build_type "Build type? [Release/Debug/RelWithDebInfo]:"
else
    build_type = arg1
end_if

if build_type == ""
    build_type = "Release"
end_if

# Get parallel jobs
if arg2 == ""
    ask jobs "Parallel jobs? [8]:"
else
    jobs = arg2
end_if

if jobs == ""
    jobs = "8"
end_if

say ""
say "Configuration:"
say "  Build Type: {build_type}"
say "  Jobs: {jobs}"
say ""

# Create build directory
say "→ Creating build directory..."
make folder "{root}/build"

# Configure CMake
say "→ Configuring CMake ({build_type})..."
run 'cmake -S "{root}" -B "{root}/build" -DCMAKE_BUILD_TYPE={build_type} -DBUILD_TOOLS=ON'

# Check CMake succeeded
if_not_exists "{root}/build/CMakeCache.txt"
    say "✗ ERROR: CMake configuration failed!"
    exit 1
end_if

say "✓ CMake configured successfully"

# Build
say "→ Building with {jobs} parallel jobs..."
run 'cmake --build "{root}/build" --config {build_type} --parallel {jobs}'

say ""
say "═══════════════════════════════════"
say "✓ Build complete!"
say "  Binaries: {root}/build"
say "═══════════════════════════════════"
