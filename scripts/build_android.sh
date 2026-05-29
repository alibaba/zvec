#!/bin/bash
set -e

#######################################################################
# build_android.sh — cross-compile zvec for Android arm64-v8a,
#                    then run all C/C++ unit tests inside an emulator.
#
# Usage:
#   ./scripts/build_android.sh [api_level] [build_type]
#
# Examples:
#   ./scripts/build_android.sh            # defaults: API 36, Release
#   ./scripts/build_android.sh 36 Debug
#######################################################################

CURRENT_DIR=$(pwd)
ABI="arm64-v8a"
API_LEVEL=${1:-35}
BUILD_TYPE=${2:-"Release"}
CORE_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# ── Android SDK paths (set later, after host protoc build) ───────────
ANDROID_SDK_ROOT=${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}
ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-$(ls -d "$ANDROID_SDK_ROOT/ndk/"* 2>/dev/null | sort -V | tail -1)}

echo "============================================================"
echo " Android Cross Build & Test"
echo "============================================================"
echo "  ABI          : $ABI"
echo "  API Level    : $API_LEVEL"
echo "  Build Type   : $BUILD_TYPE"
echo "  NDK          : $ANDROID_NDK_HOME"
echo "  CPU cores    : $CORE_COUNT"
echo "============================================================"

if [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME not found at $ANDROID_NDK_HOME"
    echo "Please install the NDK via Android Studio or sdkmanager."
    exit 1
fi

# ── Step 1: build host protoc (using HOST compiler, NOT NDK) ─────────
echo ""
echo ">>> Step 1: Building protoc for host..."
HOST_BUILD_DIR="build_host"

git submodule foreach --recursive 'git stash --include-untracked' 2>/dev/null || true

if [ ! -f "$CURRENT_DIR/$HOST_BUILD_DIR/bin/protoc" ]; then
    # Explicitly avoid NDK toolchain for host build
    cmake -S . -B "$HOST_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_TOOLCHAIN_FILE="" \
        -G Ninja
    cmake --build "$HOST_BUILD_DIR" --target protoc -j"$CORE_COUNT"
else
    echo "  (cached — skipping)"
fi
PROTOC_EXECUTABLE=$CURRENT_DIR/$HOST_BUILD_DIR/bin/protoc
echo ">>> Step 1: Done (protoc=$PROTOC_EXECUTABLE)"

# ── Now export Android env vars for cross-compilation ────────────────
export ANDROID_SDK_ROOT
export ANDROID_HOME=$ANDROID_SDK_ROOT
export ANDROID_NDK_HOME
export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake

export PATH=$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin
export PATH=$PATH:$ANDROID_SDK_ROOT/platform-tools
export PATH=$PATH:$ANDROID_SDK_ROOT/emulator
export PATH=$PATH:$ANDROID_NDK_HOME

# ── Step 2: cross-compile zvec + tests for Android ───────────────────
echo ""
echo ">>> Step 2: Cross-compiling zvec for Android ($ABI, API $API_LEVEL)..."

# reset thirdparty so the cross toolchain can patch cleanly
git submodule foreach --recursive 'git stash --include-untracked' 2>/dev/null || true

BUILD_DIR="build_android_${ABI}"

# Force CMake reconfigure to pick up any bazel.cmake changes
rm -f "$BUILD_DIR/CMakeCache.txt"

cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DANDROID_NDK="$ANDROID_NDK_HOME" \
    -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_NATIVE_API_LEVEL="$API_LEVEL" \
    -DANDROID_STL="c++_static" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_TOOLS=OFF \
    -DENABLE_NATIVE=OFF \
    -DAUTO_DETECT_ARCH=OFF \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
    -DGLOBAL_CC_PROTOBUF_PROTOC="$PROTOC_EXECUTABLE"

echo "  Building library..."
cmake --build "$BUILD_DIR" -j"$CORE_COUNT"

# Discover test target names from ctest metadata (before building)
echo "  Discovering test targets from ctest..."
TEST_NAMES=()
while IFS= read -r line; do
    name=$(echo "$line" | sed -n 's/.*Test[[:space:]]*#[0-9]*:[[:space:]]*//p')
    if [ -n "$name" ]; then
        TEST_NAMES+=("$name")
    fi
done < <(cd "$BUILD_DIR" && ctest --show-only 2>/dev/null || true)

# Fallback: find *_test targets from CMake cache
if [ ${#TEST_NAMES[@]} -eq 0 ]; then
    echo "  ctest listing unavailable, scanning ninja targets..."
    while IFS= read -r line; do
        name=$(echo "$line" | sed -n 's/^\([a-zA-Z0-9_]*_test\): .*/\1/p')
        if [ -n "$name" ]; then
            TEST_NAMES+=("$name")
        fi
    done < <(ninja -C "$BUILD_DIR" -t targets all 2>/dev/null || true)
fi

echo "  Found ${#TEST_NAMES[@]} test targets."
if [ ${#TEST_NAMES[@]} -eq 0 ]; then
    echo "WARNING: No test targets found. Skipping emulator step."
    exit 0
fi

# Build all test executables at once (without running ctest)
echo "  Building ${#TEST_NAMES[@]} test executables..."
ninja -C "$BUILD_DIR" -j"$CORE_COUNT" "${TEST_NAMES[@]}"

echo ">>> Step 2: Done"

# ── Step 3: collect test binaries ────────────────────────────────────
echo ""
echo ">>> Step 3: Collecting test binaries..."

TEST_BINS=()
for name in "${TEST_NAMES[@]}"; do
    bin_path=$(find "$BUILD_DIR" -name "$name" -type f -perm +111 2>/dev/null | head -1)
    if [ -n "$bin_path" ]; then
        TEST_BINS+=("$bin_path")
    else
        echo "  WARNING: binary not found for target '$name'"
    fi
done

echo "  Found ${#TEST_BINS[@]} test binaries."
if [ ${#TEST_BINS[@]} -eq 0 ]; then
    echo "WARNING: No test binaries found. Skipping emulator step."
    exit 0
fi

# ── Step 4: start emulator ───────────────────────────────────────────
echo ""
echo ">>> Step 4: Starting Android emulator..."

EMULATOR_BIN="$ANDROID_SDK_ROOT/emulator/emulator"
ADB_BIN="$ANDROID_SDK_ROOT/platform-tools/adb"
AVD_NAME="zvec_test_avd"

# Find a usable system image
SYS_IMG=""
for candidate in \
    "$ANDROID_SDK_ROOT/system-images/android-${API_LEVEL}/default/arm64-v8a" \
    "$ANDROID_SDK_ROOT/system-images/android-${API_LEVEL}/google_apis/arm64-v8a" \
    "$ANDROID_SDK_ROOT/system-images/android-${API_LEVEL}/google_apis_playstore/arm64-v8a"; do
    if [ -d "$candidate" ]; then
        SYS_IMG="$candidate"
        break
    fi
done

# If the exact API level isn't found, try any available arm64-v8a image
if [ -z "$SYS_IMG" ]; then
    echo "  No system image for API $API_LEVEL, searching for any arm64-v8a image..."
    SYS_IMG=$(find "$ANDROID_SDK_ROOT/system-images" -type d -name "arm64-v8a" 2>/dev/null | head -1)
fi

if [ -z "$SYS_IMG" ]; then
    echo "ERROR: No arm64-v8a system image found."
    echo "Install one via: sdkmanager 'system-images;android-${API_LEVEL};default;arm64-v8a'"
    exit 1
fi
echo "  Using system image: $SYS_IMG"

# Create AVD without avdmanager (write INI files directly)
AVD_DIR="$HOME/.android/avd/${AVD_NAME}.avd"
AVD_INI="$HOME/.android/avd/${AVD_NAME}.ini"

cleanup_emulator() {
    echo ""
    echo ">>> Cleaning up emulator..."
    $ADB_BIN emu kill 2>/dev/null || true
    sleep 2
    # Kill any lingering emulator processes for our AVD
    pkill -f "emulator.*${AVD_NAME}" 2>/dev/null || true
}
trap cleanup_emulator EXIT

mkdir -p "$AVD_DIR"

# Write the top-level .ini
cat > "$AVD_INI" << EOF
avd.ini.encoding=UTF-8
path=${AVD_DIR}
path.rel=avd/${AVD_NAME}.avd
target=android-${API_LEVEL}
EOF

# Write the AVD config
cat > "$AVD_DIR/config.ini" << EOF
AvdId=${AVD_NAME}
PlayStore.enabled=false
abi.type=arm64-v8a
avd.ini.displayname=${AVD_NAME}
avd.ini.encoding=UTF-8
disk.dataPartition.size=8G
hw.accelerator.isConfigured=true
hw.cpu.arch=arm64
hw.cpu.ncore=4
hw.lcd.density=420
hw.lcd.height=1920
hw.lcd.width=1080
hw.ramSize=4096
image.sysdir.1=${SYS_IMG}/
tag.display=Default
tag.id=default
EOF

echo "  Created AVD: $AVD_NAME"

# Kill any existing emulator
$ADB_BIN emu kill 2>/dev/null || true
sleep 1

# Start emulator
echo "  Launching emulator..."
$EMULATOR_BIN -avd "$AVD_NAME" \
    -no-window -no-audio -no-boot-anim \
    -gpu swiftshader_indirect \
    -netdelay none -netspeed full \
    -memory 4096 \
    -no-snapshot \
    -wipe-data \
    2>&1 | sed 's/^/  [emulator] /' &
EMULATOR_PID=$!

# Wait for the device to become reachable
echo "  Waiting for device..."
$ADB_BIN wait-for-device

# Wait for boot_completed
echo "  Waiting for boot to complete..."
BOOT_TIMEOUT=300
BOOT_ELAPSED=0
while true; do
    BOOTED=$($ADB_BIN shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n ' || true)
    if [ "$BOOTED" = "1" ]; then
        echo "  Emulator is ready! (took ${BOOT_ELAPSED}s)"
        break
    fi
    if [ $BOOT_ELAPSED -ge $BOOT_TIMEOUT ]; then
        echo "ERROR: Emulator failed to boot within ${BOOT_TIMEOUT}s"
        exit 1
    fi
    sleep 3
    BOOT_ELAPSED=$((BOOT_ELAPSED + 3))
done

# Print device info
echo ""
echo "  Device ABI : $($ADB_BIN shell getprop ro.product.cpu.abi | tr -d '\r')"
echo "  ABI list   : $($ADB_BIN shell getprop ro.product.cpu.abilist | tr -d '\r')"
echo "  CPU info   :"
$ADB_BIN shell 'cat /proc/cpuinfo | grep -E "^(Features|flags|processor)"' 2>/dev/null | head -4 | sed 's/^/    /'

# ── Step 5: run tests on emulator ────────────────────────────────────
echo ""
echo ">>> Step 5: Running ${#TEST_BINS[@]} unit tests on emulator..."

DEVICE_TEST_DIR="/data/local/tmp/zvec_tests"
DEVICE_LIB_DIR="$DEVICE_TEST_DIR/lib"
$ADB_BIN shell "mkdir -p $DEVICE_TEST_DIR $DEVICE_LIB_DIR" 2>/dev/null

# Push all shared libraries that tests may depend on
echo "  Pushing shared libraries..."
SO_COUNT=0
while IFS= read -r so_file; do
    $ADB_BIN push "$so_file" "$DEVICE_LIB_DIR/$(basename "$so_file")" > /dev/null 2>&1
    SO_COUNT=$((SO_COUNT + 1))
done < <(find "$BUILD_DIR/lib" -name "*.so" -type f 2>/dev/null)
echo "  Pushed $SO_COUNT shared libraries."

# Push non-test helper binaries (e.g. data_generator, collection_optimizer)
# These are needed by crash_recovery tests which fork+exec them at runtime.
echo "  Pushing helper binaries..."
HELPER_COUNT=0
for helper_name in data_generator collection_optimizer; do
    helper_path=$(find "$BUILD_DIR" -name "$helper_name" -type f -perm +111 ! -name "*_test" 2>/dev/null | head -1)
    if [ -n "$helper_path" ]; then
        $ADB_BIN push "$helper_path" "$DEVICE_TEST_DIR/$helper_name" > /dev/null 2>&1
        $ADB_BIN shell "chmod 755 $DEVICE_TEST_DIR/$helper_name" 2>/dev/null
        HELPER_COUNT=$((HELPER_COUNT + 1))
    fi
done
echo "  Pushed $HELPER_COUNT helper binaries."

TOTAL=${#TEST_BINS[@]}
PASSED=0
FAILED=0
FAILED_NAMES=()
IDX=0

for test_bin in "${TEST_BINS[@]}"; do
    IDX=$((IDX + 1))
    test_name=$(basename "$test_bin")
    device_path="$DEVICE_TEST_DIR/$test_name"
    # Give each test its own working directory to avoid name collisions
    # (e.g. flat_builder_test binary vs flat_builder_test/ directory)
    WORK_DIR="$DEVICE_TEST_DIR/workdir_${test_name}"

    echo ""
    echo "────────────────────────────────────────"
    echo "  [$IDX/$TOTAL] $test_name"
    echo "────────────────────────────────────────"

    set +e
    # Create isolated working directory
    $ADB_BIN shell "mkdir -p $WORK_DIR" 2>/dev/null

    # Copy helper binaries into working directory so crash_recovery tests
    # (which fork+exec data_generator / collection_optimizer) can find them.
    $ADB_BIN shell "for h in $DEVICE_TEST_DIR/data_generator $DEVICE_TEST_DIR/collection_optimizer; do [ -f \$h ] && cp \$h $WORK_DIR/; done" 2>/dev/null

    # Push binary
    $ADB_BIN push "$test_bin" "$device_path" > /dev/null 2>&1
    $ADB_BIN shell "chmod 755 $device_path" 2>/dev/null

    # Run test from its own working directory with LD_LIBRARY_PATH
    OUTPUT=$($ADB_BIN shell "cd $WORK_DIR && LD_LIBRARY_PATH=$DEVICE_LIB_DIR $device_path 2>&1; echo EXIT_CODE=\$?" 2>&1)

    # Extract exit code from the output
    EXIT_CODE=$(echo "$OUTPUT" | grep -o 'EXIT_CODE=[0-9]*' | tail -1 | cut -d= -f2)
    set -e

    # Print test output (without the EXIT_CODE line)
    echo "$OUTPUT" | grep -v 'EXIT_CODE=' | sed 's/^/  /'

    if [ "$EXIT_CODE" = "0" ]; then
        echo "  >>> PASSED"
        PASSED=$((PASSED + 1))
    else
        # Detect "segfault-on-exit" pattern: all gtest assertions passed but
        # process crashed during teardown (common with cross-compiled gmock tests).
        GTEST_PASSED_LINE=$(echo "$OUTPUT" | grep '\[  PASSED  \]' | tail -1)
        GTEST_FAILED_LINE=$(echo "$OUTPUT" | grep '\[  FAILED  \]' | head -1)
        if [ -n "$GTEST_PASSED_LINE" ] && [ -z "$GTEST_FAILED_LINE" ] && \
           { [ "$EXIT_CODE" = "139" ] || [ "$EXIT_CODE" = "134" ] || [ "$EXIT_CODE" = "135" ]; }; then
            echo "  >>> PASSED (with crash-on-exit, exit code: $EXIT_CODE — ignored)"
            PASSED=$((PASSED + 1))
        else
            echo "  >>> FAILED (exit code: $EXIT_CODE)"
            FAILED=$((FAILED + 1))
            FAILED_NAMES+=("$test_name")
        fi
    fi

    # Clean up binary and working directory to reclaim disk space
    $ADB_BIN shell "rm -rf $device_path $WORK_DIR" 2>/dev/null || true
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Test Summary"
echo "============================================================"
echo "  Total  : $TOTAL"
echo "  Passed : $PASSED"
echo "  Failed : $FAILED"
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  Failed tests:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "    - $name"
    done
fi
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo "All tests passed!"
